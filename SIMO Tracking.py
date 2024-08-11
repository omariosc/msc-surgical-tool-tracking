import os
import sys
import time
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.spatial import distance

TEST = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration (Set to 'ART' or '6DOF')
dataset_type = sys.argv[1]  # 'ART' or '6DOF'
# dataset_type = "ART"
# Backbone is either vgg, resnet50, or resnet18
BACKBONE = sys.argv[2] if len(sys.argv) > 2 else "vgg"
# BACKBONE = "vgg"

# Configure paths and model settings based on dataset type
if dataset_type == "ART":
    max_bboxes = 2
    n_classes = 4
    data_dir = "data/ART-Net"
    output_dir = "chkpts/SIMO/ART/output"
    weights_folder = "chkpts/SIMO/ART/weights"
    tracking = False
else:  # 6DOF
    max_bboxes = 4
    n_classes = 8
    data_dir = "data/6DOF"
    output_dir = "chkpts/SIMO/6DOF/output"
    weights_folder = "chkpts/SIMO/6DOF/weights"
    tracking = True


# Define the ToolDataset class
class ToolDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, max_bboxes=2):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.max_bboxes = max_bboxes
        self.image_files = [
            f
            for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
            and os.path.exists(os.path.join(label_dir, f.replace(".png", ".txt")))
            and "Neg" not in f  # Exclude negative images
        ]
        self.image_files = sorted(
            self.image_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        # Initialize list to hold valid images and their corresponding labels
        valid_image_files = []
        valid_labels = []

        # Process and filter labels during initialization
        for img_file in self.image_files:
            label_file = img_file.replace(".png", ".txt")
            label_path = os.path.join(self.label_dir, label_file)
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                labels = np.loadtxt(label_path).reshape(-1, 5)

                # Check if labels have the required number of bounding boxes
                if labels.shape[0] == self.max_bboxes:
                    valid_labels.append(labels)
                    valid_image_files.append(img_file)
            # If label file doesn't meet the criteria, skip adding it to valid lists
            # No need to remove files from disk

        # Update image files and labels to only include valid ones
        self.image_files = valid_image_files
        self.labels = valid_labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        labels = self.labels[idx]
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels


class SIMOModel(nn.Module):
    def __init__(self, n_classes, backbone="vgg"):
        super(SIMOModel, self).__init__()
        self.n_classes = n_classes
        self.max_bboxes = 2 if n_classes == 4 else 4

        # Choose the backbone model
        if backbone == "vgg":
            self.backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
            backbone_out_channels = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_out_channels = 2048
        elif backbone == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_out_channels = 512
        else:
            raise ValueError("Invalid backbone. Choose 'vgg' or 'resnet'.")

        # Freeze the backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Feature representation generator (FRG)
        self.frg = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        combined_channels = backbone_out_channels + 512  # For concatenation

        # Decoders for tool and tooltip bounding box regression
        self.tool_1_decoder = self._create_decoder(combined_channels)
        self.tool_2_decoder = self._create_decoder(combined_channels)
        self.tooltip_1_decoder = self._create_decoder(combined_channels)
        self.tooltip_2_decoder = self._create_decoder(combined_channels)

        # Confidence prediction for tool and tooltip
        self.tool_1_confidence = self._create_confidence_head(combined_channels)
        self.tool_2_confidence = self._create_confidence_head(combined_channels)
        self.tooltip_1_confidence = self._create_confidence_head(combined_channels)
        self.tooltip_2_confidence = self._create_confidence_head(combined_channels)

        # After creating the decoders and confidence heads in the SIMOModel class __init__ method
        if n_classes == 4:
            for param in self.tool_2_decoder.parameters():
                param.requires_grad = False
            for param in self.tooltip_2_decoder.parameters():
                param.requires_grad = False
            for param in self.tool_2_confidence.parameters():
                param.requires_grad = False
            for param in self.tooltip_2_confidence.parameters():
                param.requires_grad = False

    def _create_decoder(self, combined_channels):
        return nn.Sequential(
            nn.Conv2d(combined_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 4, kernel_size=1
            ),  # 4 outputs for bounding box coordinates
            nn.AdaptiveAvgPool2d((1, 1)),  # Pooling to get a 1x1 output
            nn.Flatten(),  # Flatten to shape [batch_size, 4]
        )

    def _create_confidence_head(self, combined_channels):
        return nn.Sequential(
            nn.Conv2d(combined_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Sigmoid(),  # Confidence between 0 and 1
        )

    def convert_labels_xywh_to_xyxy(self, labels):
        # Convert xywh to x1y1x2y2 format
        converted_labels = []
        for label in labels:
            x, y, w, h = label
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            converted_labels.append([x1, y1, x2, y2])
        return torch.tensor(converted_labels, dtype=torch.float32)

    def forward(self, x):
        x_backbone = self.backbone(x)

        # Feature representation generator
        frg = self.frg(x)

        # Resize FRG output to match backbone output size
        frg_resized = F.interpolate(
            frg,
            size=(x_backbone.size(2), x_backbone.size(3)),
            mode="bilinear",
            align_corners=True,
        )

        # Concatenate backbone and FRG outputs
        combined_features = torch.cat((x_backbone, frg_resized), dim=1)

        # Decoders for tools and tooltips
        tool_1_pred = self.tool_1_decoder(combined_features)
        tooltip_1_pred = self.tooltip_1_decoder(combined_features)
        tool_1_conf = self.tool_1_confidence(combined_features)
        tooltip_1_conf = self.tooltip_1_confidence(combined_features)

        if n_classes == 8:
            tool_2_pred = self.tool_2_decoder(combined_features)
            tooltip_2_pred = self.tooltip_2_decoder(combined_features)
            tool_2_conf = self.tool_2_confidence(combined_features)
            tooltip_2_conf = self.tooltip_2_confidence(combined_features)
        else:
            tool_2_pred, tool_2_conf, tooltip_2_pred, tooltip_2_conf = tool_1_pred, tool_1_conf, tooltip_1_pred, tooltip_1_conf

        return (
            tool_1_pred,
            tool_1_conf,
            tool_2_pred,
            tool_2_conf,
            tooltip_1_pred,
            tooltip_1_conf,
            tooltip_2_pred,
            tooltip_2_conf,
        )

    def train_model(
        self, train_loader, val_loader, num_epochs=300, lr=0.001, patience=3
    ):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        initial_patience = patience
        start_time = time.time()
        total_time = 0
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        checkpoints_folder = weights_folder
        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)

        best_val_loss = np.inf
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for batch, (images, labels) in enumerate(train_loader):
                images = images.to(device).float()

                # Split labels for tools and tooltips based on max_bboxes
                tool_labels, tooltip_labels = (
                    labels[:, : self.max_bboxes // 2, :],
                    labels[:, self.max_bboxes // 2 :, :],
                )

                tool_labels = tool_labels.to(device).float()
                tooltip_labels = tooltip_labels.to(device).float()

                optimizer.zero_grad()

                preds = self.forward(images)

                if self.max_bboxes == 2:
                    loss = self.compute_losses(
                        preds[
                            :4
                        ],  # Only tool_1_pred, tool_1_conf, tooltip_1_pred, tooltip_1_conf
                        [tool_labels[:, 0], tooltip_labels[:, 0]],
                    )
                else:
                    loss = self.compute_losses(
                        preds,  # All predictions for 4 bounding boxes
                        [
                            tool_labels[:, 0],
                            tool_labels[:, 1],
                            tooltip_labels[:, 0],
                            tooltip_labels[:, 1],
                        ],
                    )
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                print(f"Batch {batch+1}/{len(train_loader)}, Loss: {loss.item()}")
                torch.cuda.empty_cache()

                if TEST:
                    break

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            end_time = time.time()
            epoch_time = end_time - start_time

            total_time += epoch_time
            start_time = end_time

            avg_val_loss = self.validate_model(val_loader)
            val_losses.append(avg_val_loss)

            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss} in {epoch_time:.2f} seconds, Validation Loss: {avg_val_loss}"
            )

            torch.save(self.state_dict(), f"{checkpoints_folder}/{epoch}.pt")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.state_dict(), f"{checkpoints_folder}/best.pt")
                patience = initial_patience
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping")
                    break

            if TEST:
                break

        print(f"Total training time: {total_time:.2f} seconds")
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        return train_losses, val_losses

    def validate_model(self, val_loader):
        self.eval()
        val_loss = 0.0
        start_time = time.time()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device).float()

                # Split labels for tools and tooltips based on max_bboxes
                tool_labels, tooltip_labels = (
                    labels[:, : self.max_bboxes // 2, :],
                    labels[:, self.max_bboxes // 2 :, :],
                )

                tool_labels = tool_labels.to(device).float()
                tooltip_labels = tooltip_labels.to(device).float()

                preds = self.forward(images)

                if self.max_bboxes == 2:
                    loss = self.compute_losses(
                        preds[
                            :4
                        ],  # Only tool_1_pred, tool_1_conf, tooltip_1_pred, tooltip_1_conf
                        [tool_labels[:, 0], tooltip_labels[:, 0]],
                    )
                else:
                    loss = self.compute_losses(
                        preds,  # All predictions for 4 bounding boxes
                        [
                            tool_labels[:, 0],
                            tool_labels[:, 1],
                            tooltip_labels[:, 0],
                            tooltip_labels[:, 1],
                        ],
                    )
                val_loss += loss.mean().item()

            torch.cuda.empty_cache()

        end_time = time.time()
        print(f"Time per image: {(end_time - start_time) / len(val_loader):.2f} seconds")

        return val_loss / len(val_loader)

    def compute_losses(self, preds, labels):
        """
        Compute the combined loss for bounding box regression and confidence prediction.

        Args:
        - preds: List of predictions containing both bounding box coordinates (in xywh) and confidence scores.
        - labels: List of labels where each entry contains the true bounding boxes and confidence scores.

        Returns:
        - total_loss: Combined loss value.
        """
        total_loss = 0.0

        for i in range(self.max_bboxes):
            pred_bbox_xywh = preds[2 * i]  # Bounding box prediction in xywh
            pred_conf = preds[2 * i + 1]  # Confidence prediction

            if pred_bbox_xywh.shape[-1] != 4:
                pred_bbox_xywh = pred_bbox_xywh.view(-1, 4)  # Ensure the shape is correct

            # Move to CPU before converting to numpy
            pred_bbox_xywh_cpu = pred_bbox_xywh.cpu()

            # Convert pred_bbox_xywh and label_bbox from xywh to xyxy format
            pred_bbox = self.convert_labels_xywh_to_xyxy(pred_bbox_xywh_cpu)
            label_bbox = self.convert_labels_xywh_to_xyxy(labels[i][:, 1:].cpu())

            label_conf = (
                labels[i][:, 0].unsqueeze(1).cpu()
            )  # Confidence label (first column), reshape to match pred_conf

            # Compute IoU loss for bounding boxes
            iou_loss_value = self.iou_loss(pred_bbox, label_bbox)

            # Compute confidence loss
            conf_loss_value = F.binary_cross_entropy_with_logits(
                pred_conf.cpu(), label_conf
            )

            total_loss += iou_loss_value + conf_loss_value

        return total_loss

    def visualize_bounding_boxes(
        self,
        image,
        tool_1_preds,
        tool_1_conf,
        tool_2_preds,
        tool_2_conf,
        tooltip_1_preds,
        tooltip_1_conf,
        tooltip_2_preds,
        tooltip_2_conf,
        save_path=None,
    ):
        image = image.cpu().numpy().squeeze()
        image = np.moveaxis(image, 0, -1)
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

        # Convert predictions from xywh to xyxy format for visualization
        tool_1_preds = self.convert_labels_xywh_to_xyxy(tool_1_preds.cpu().numpy())
        tool_2_preds = self.convert_labels_xywh_to_xyxy(tool_2_preds.cpu().numpy())
        tooltip_1_preds = self.convert_labels_xywh_to_xyxy(tooltip_1_preds.cpu().numpy())
        tooltip_2_preds = self.convert_labels_xywh_to_xyxy(tooltip_2_preds.cpu().numpy())

        fig, ax = plt.subplots(1)
        ax.axis("off")
        ax.imshow(image)
        fig.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # Define the data for each tool/tooltip type
        items = [
            (tool_1_preds, tool_1_conf, "red", "Tool 1"),
            (tool_2_preds, tool_2_conf, "blue", "Tool 2"),
            (tooltip_1_preds, tooltip_1_conf, "green", "Tooltip 1"),
            (tooltip_2_preds, tooltip_2_conf, "orange", "Tooltip 2"),
        ]

        # Iterate through the items to add patches and labels
        for preds, confs, color, label in items:
            for pred, conf in zip(preds, confs):
                x1, y1, x2, y2 = map(int, pred)
                w, h = x2 - x1, y2 - y1
                x1 *= image.shape[1]
                x2 *= image.shape[1]
                y1 *= image.shape[0]
                y2 *= image.shape[0]
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (x1, y1),
                        w,
                        h,
                        edgecolor=color,
                        facecolor="none",
                        linewidth=2,
                        label=f"{label}, Conf: {conf.item():.2f}",
                    )
                )
                ax.text(x1, y1, f"{label}, {conf.item():.2f}", color=color)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            ax.imshow(image)

        plt.close()

    def load_best_weights(self, weights_folder="chkpts/SIMO/ART/weights"):
        best_weights = os.path.join(weights_folder, "best.pt")
        if os.path.exists(best_weights):
            self.load_state_dict(torch.load(best_weights))
            print("Loaded best weights")
        else:
            print("No best weights found.")

    def compute_metrics(
        self, pred_boxes, true_boxes, conf_scores, iou_thresholds=[0.5]
    ):
        binary_true_boxes = []
        binary_pred_boxes = []
        valid_conf_scores = []

        for i in range(len(true_boxes)):
            iou_scores = [
                self.iou(pred_boxes[i], true_boxes[j]) for j in range(len(true_boxes))
            ]
            max_iou = max(iou_scores)
            for iou_threshold in iou_thresholds:
                if max_iou > iou_threshold:
                    binary_true_boxes.append(1)
                    binary_pred_boxes.append(1)
                else:
                    binary_true_boxes.append(0)
                    binary_pred_boxes.append(0)
            valid_conf_scores.append(conf_scores[i].cpu().item())

        precisions, recalls, _ = precision_recall_curve(
            binary_true_boxes, valid_conf_scores
        )
        ap50 = average_precision_score(binary_true_boxes, valid_conf_scores)

        aps = []
        for threshold in np.linspace(0.5, 0.95, 10):
            binary_true_boxes = []
            binary_pred_boxes = []
            for i in range(len(true_boxes)):
                iou_scores = [
                    self.iou(pred_boxes[i], true_boxes[j])
                    for j in range(len(true_boxes))
                ]
                max_iou = max(iou_scores)
                if max_iou > threshold:
                    binary_true_boxes.append(1)
                    binary_pred_boxes.append(1)
                else:
                    binary_true_boxes.append(0)
                    binary_pred_boxes.append(0)

            ap = average_precision_score(binary_true_boxes, valid_conf_scores)
            aps.append(ap)

        mAP_50_95 = np.mean(aps)

        return precisions, recalls, ap50, mAP_50_95

    def test(self, input_dir, output_dir, tracking=False):
        os.makedirs(output_dir, exist_ok=True)
        self.eval()
        all_tool_preds, all_tool_labels, all_tool_confs = [], [], []
        all_tooltip_preds, all_tooltip_labels, all_tooltip_confs = [], [], []
        if tracking:
            prev_tool_centres = None
            prev_tooltip_centres = None
            # images will be in format test5_0.png, test5_1.png, test5_2.png, ... so sort them based on number
            images = sorted(
                [
                    os.path.join(input_dir, f)
                    for f in os.listdir(input_dir)
                    if f.endswith(".png")
                ],
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
        else:
            images = sorted(
                [
                    os.path.join(input_dir, f)
                    for f in os.listdir(input_dir)
                    if f.endswith(".png")
                ]
            )
        with torch.no_grad():
            for path in images:
                image = Image.open(path)
                image = functional.to_tensor(image)
                image = functional.resize(image, (512, 512))
                image = image.unsqueeze(0).to(device)

                (
                    tool_1_pred,
                    tool_1_conf,
                    tool_2_pred,
                    tool_2_conf,
                    tooltip_1_pred,
                    tooltip_1_conf,
                    tooltip_2_pred,
                    tooltip_2_conf,
                ) = self.forward(image)

                if tracking:
                    (
                        tool_1_pred,
                        tool_1_conf,
                        tool_2_pred,
                        tool_2_conf,
                        tooltip_1_pred,
                        tooltip_1_conf,
                        tooltip_2_pred,
                        tooltip_2_conf,
                        prev_tool_centres,
                        prev_tooltip_centres,
                    ) = self.track_objects(
                        prev_tool_centres,
                        prev_tooltip_centres,
                        tool_1_pred,
                        tool_1_conf,
                        tool_2_pred,
                        tool_2_conf,
                        tooltip_1_pred,
                        tooltip_1_conf,
                        tooltip_2_pred,
                        tooltip_2_conf,
                    )
                self.visualize_bounding_boxes(
                    image,
                    tool_1_pred,
                    tool_1_conf,
                    tool_2_pred,
                    tool_2_conf,
                    tooltip_1_pred,
                    tooltip_1_conf,
                    tooltip_2_pred,
                    tooltip_2_conf,
                    save_path=os.path.join(output_dir, os.path.basename(path)),
                )

                # Append predictions and confidences for metrics calculation
                all_tool_preds.append(tool_1_pred)
                all_tool_preds.append(tool_2_pred)
                all_tool_labels.append(
                    tool_1_pred
                )  # Ground truth labels should be available to calculate metrics properly
                all_tool_labels.append(tool_2_pred)
                all_tool_confs.append(tool_1_conf)
                all_tool_confs.append(tool_2_conf)

                all_tooltip_preds.append(tooltip_1_pred)
                all_tooltip_preds.append(tooltip_2_pred)
                all_tooltip_labels.append(
                    tooltip_1_pred
                )  # Ground truth labels should be available to calculate metrics properly
                all_tooltip_labels.append(tooltip_2_pred)
                all_tooltip_confs.append(tooltip_1_conf)
                all_tooltip_confs.append(tooltip_2_conf)

            # Compute metrics for tools
            precisions_tool, recalls_tool, mAP_50_tool, mAP_50_95_tool = (
                self.compute_metrics(all_tool_preds, all_tool_labels, all_tool_confs)
            )
            # Compute metrics for tooltips
            precisions_tooltip, recalls_tooltip, mAP_50_tooltip, mAP_50_95_tooltip = (
                self.compute_metrics(
                    all_tooltip_preds, all_tooltip_labels, all_tooltip_confs
                )
            )

        # Print the metrics
        print(
            f"Tool - Precision: {precisions_tool[-1]}, Recall: {recalls_tool[-1]}, mAP@50: {mAP_50_tool}, mAP@50-95: {mAP_50_95_tool}"
        )
        print(
            f"Tooltip - Precision: {precisions_tooltip[-1]}, Recall: {recalls_tooltip[-1]}, mAP@50: {mAP_50_tooltip}, mAP@50-95: {mAP_50_95_tooltip}"
        )

        return (
            precisions_tool,
            recalls_tool,
            mAP_50_tool,
            mAP_50_95_tool,
            precisions_tooltip,
            recalls_tooltip,
            mAP_50_tooltip,
            mAP_50_95_tooltip,
        )

    def track_objects(
        self,
        prev_tool_centres,
        prev_tooltip_centres,
        tool_1_pred,
        tool_1_conf,
        tool_2_pred,
        tool_2_conf,
        tooltip_1_pred,
        tooltip_1_conf,
        tooltip_2_pred,
        tooltip_2_conf,
    ):
        try:
            # Compute centres of the predicted bounding boxes
            tool_centres = []
            tooltip_centres = []

            if tool_1_pred is not None and tool_2_pred is not None:
                tool_centres = [
                    (tool_1_pred[0] + tool_1_pred[2]) / 2,
                    (tool_2_pred[0] + tool_2_pred[2]) / 2,
                ]
            elif tool_1_pred is not None:
                tool_centres = [(tool_1_pred[0] + tool_1_pred[2]) / 2]
            elif tool_2_pred is not None:
                tool_centres = [(tool_2_pred[0] + tool_2_pred[2]) / 2]

            if tooltip_1_pred is not None and tooltip_2_pred is not None:
                tooltip_centres = [
                    (tooltip_1_pred[0] + tooltip_1_pred[2]) / 2,
                    (tooltip_2_pred[0] + tooltip_2_pred[2]) / 2,
                ]
            elif tooltip_1_pred is not None:
                tooltip_centres = [(tooltip_1_pred[0] + tooltip_1_pred[2]) / 2]
            elif tooltip_2_pred is not None:
                tooltip_centres = [(tooltip_2_pred[0] + tooltip_2_pred[2]) / 2]

            if prev_tool_centres is None:
                prev_tool_centres = tool_centres
                prev_tooltip_centres = tooltip_centres
            else:
                if tool_centres:
                    # Compute distances between previous and current centres for tools
                    tool_distances = distance.cdist(
                        prev_tool_centres, tool_centres, "euclidean"
                    )

                    # Assign the closest previous centre to the current one
                    if len(tool_centres) == 2:
                        if tool_distances[0][0] < tool_distances[1][0]:
                            tool_1_pred, tool_2_pred = tool_1_pred, tool_2_pred
                            tool_1_conf, tool_2_conf = tool_1_conf, tool_2_conf
                        else:
                            tool_1_pred, tool_2_pred = tool_2_pred, tool_1_pred
                            tool_1_conf, tool_2_conf = tool_2_conf, tool_1_conf
                    elif len(tool_centres) == 1:
                        tool_1_pred = (
                            tool_1_pred
                            if prev_tool_centres[0] == tool_centres[0]
                            else tool_2_pred
                        )
                        tool_1_conf = (
                            tool_1_conf
                            if prev_tool_centres[0] == tool_centres[0]
                            else tool_2_conf
                        )

                if tooltip_centres:
                    # Compute distances between previous and current centres for tooltips
                    tooltip_distances = distance.cdist(
                        prev_tooltip_centres, tooltip_centres, "euclidean"
                    )

                    # Assign the closest previous centre to the current one
                    if len(tooltip_centres) == 2:
                        if tooltip_distances[0][0] < tooltip_distances[1][0]:
                            tooltip_1_pred, tooltip_2_pred = tooltip_1_pred, tooltip_2_pred
                            tooltip_1_conf, tooltip_2_conf = tooltip_1_conf, tooltip_2_conf
                        else:
                            tooltip_1_pred, tooltip_2_pred = tooltip_2_pred, tooltip_1_pred
                            tooltip_1_conf, tooltip_2_conf = tooltip_2_conf, tooltip_1_conf
                    elif len(tooltip_centres) == 1:
                        tooltip_1_pred = (
                            tooltip_1_pred
                            if prev_tooltip_centres[0] == tooltip_centres[0]
                            else tooltip_2_pred
                        )
                        tooltip_1_conf = (
                            tooltip_1_conf
                            if prev_tooltip_centres[0] == tooltip_centres[0]
                            else tooltip_2_conf
                        )

                prev_tool_centres = tool_centres
                prev_tooltip_centres = tooltip_centres

        except Exception as e:
            print(f"Error during tracking: {e}")

        return (
            tool_1_pred,
            tool_1_conf,
            tool_2_pred,
            tool_2_conf,
            tooltip_1_pred,
            tooltip_1_conf,
            tooltip_2_pred,
            tooltip_2_conf,
            prev_tool_centres,
            prev_tooltip_centres,
        )

    def iou(self, pred, target, smooth=1e-6):
        """
        Compute the Intersection over Union (IoU) between two sets of boxes.
        """
        try:
            # Calculate intersection
            inter_xmin = torch.max(pred[:, 0], target[:, 0])
            inter_ymin = torch.max(pred[:, 1], target[:, 1])
            inter_xmax = torch.min(pred[:, 2], target[:, 2])
            inter_ymax = torch.min(pred[:, 3], target[:, 3])

            inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(
                inter_ymax - inter_ymin, min=0
            )

            # Calculate union
            pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
            target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

            union_area = pred_area + target_area - inter_area

            iou = (inter_area + smooth) / (union_area + smooth)
            return iou
        except Exception as e:
            return 0.0

    def iou_loss(self, pred, target):
        """
        Compute IoU loss.
        """
        iou_score = self.iou(pred, target)
        return 1 - iou_score


def preprocess_background(image1, image2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to both images to reduce noise
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Calculate absolute difference between images
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    # Remove isolated pixels not in connected regions
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity, cv2.CV_32S
    )
    sizes = stats[1:, -1]  # Sizes of connected components, ignoring the background
    min_size = 50  # Minimum size of connected component to keep

    # Create mask to remove small components
    mask = np.zeros_like(thresh)
    for i in range(1, num_labels):
        if sizes[i - 1] >= min_size:
            mask[labels == i] = 255

    # Dilate the remaining components to fill gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)

    # Now using dilated as a map, darken all pixels in image2 to 0.3 times their value not in dilated
    background = image2.copy()
    for i in range(3):
        background[:, :, i] = np.where(
            dilated == 255, image2[:, :, i], 0.3 * image2[:, :, i]
        )

    return background


def process_images(input_dir, output_dir, label_dir, output_dir_labels):
    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_labels):
        os.makedirs(output_dir_labels)

    # Get the list of image files, sorted based on number (test5_1, test5_2, ...)
    image_files = sorted(
        os.listdir(input_dir), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    # Iterate over pairs of images (assuming each pair is consecutive in the sorted list)
    for i in range(1, len(image_files)):
        image1_path = os.path.join(input_dir, image_files[i - 1])
        image2_path = os.path.join(input_dir, image_files[i])

        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        # Preprocess and remove the background
        processed_image = preprocess_background(image1, image2)

        # Save the processed image
        output_path = os.path.join(output_dir, f"processed_{image_files[i]}")
        cv2.imwrite(output_path, processed_image)

        # There is a label file with same name as image in label_dir, copy it to output_dir_labels with same name as processed image
        label_path = os.path.join(label_dir, image_files[i].replace(".png", ".txt"))
        output_label_path = os.path.join(
            output_dir_labels, f"processed_{image_files[i].replace('.png', '.txt')}"
        )
        with open(label_path, "r") as f:
            lines = f.readlines()
        with open(output_label_path, "w") as f:
            f.writelines(lines)

        print(f"Processed {image_files[i]}")

    print(
        f"Processing complete. Processed images saved in {output_dir} and labels in {output_dir_labels}"
    )


def main():
    os.system("set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ToolDataset(
        image_dir=os.path.join(data_dir, "images/train"),
        label_dir=os.path.join(data_dir, "labels/train"),
        transform=transform,
        max_bboxes=max_bboxes,
    )
    val_dataset = ToolDataset(
        image_dir=os.path.join(data_dir, "images/val"),
        label_dir=os.path.join(data_dir, "labels/val"),
        transform=transform,
        max_bboxes=max_bboxes,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = SIMOModel(n_classes=n_classes, backbone=BACKBONE).to(device)

    model.train_model(train_loader, val_loader, num_epochs=300, lr=0.001, patience=10)

    # Load best weights and run evaluation on validation set
    model.load_best_weights(weights_folder)
    results = model.test(
        input_dir=os.path.join(data_dir, "images/val"),
        output_dir=output_dir,
        tracking=tracking,
    )
    print(results)


if __name__ == "__main__":
    main()
