import os
from re import T
import sys
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_curve, average_precision_score


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration (Set to 'ART' or '6DOF')
dataset_type = sys.argv[1]  # 'ART' or '6DOF'
# dataset_type = "ART"
# Backbone is either vgg, resnet50, or resnet18
BACKBONE = sys.argv[2] if len(sys.argv) > 2 else "vgg"
# BACKBONE = "vgg"
TEST = sys.argv[3] if len(sys.argv) > 3 else False

# Configure paths and model settings based on dataset type
if dataset_type == "ART":
    max_bboxes = 2
    data_dir = "data/ART-Net"
    output_dir = f"chkpts/SIMO/ART/{BACKBONE}/output"
    weights_folder = f"chkpts/SIMO/ART/{BACKBONE}/weights"
    tracking = False
else:  # 6DOF
    max_bboxes = 4
    data_dir = "data/6DOF"
    output_dir = f"chkpts/SIMO/6DOF/{BACKBONE}/output"
    weights_folder = f"chkpts/SIMO/6DOF/{BACKBONE}/weights"
    tracking = True


# Define the ToolDataset class
class ToolDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, max_bboxes=2):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.max_bboxes = max_bboxes
        self.image_files = sorted(
            [
                f
                for f in os.listdir(image_dir)
                if os.path.isfile(os.path.join(image_dir, f))
                and os.path.exists(os.path.join(label_dir, f.replace(".png", ".txt")))
                and "Neg" not in f  # Exclude negative images
            ]
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

        # Remove first element of labels (class)
        # if len(labels) == 5:
        #     labels = labels[:, 1:]

        if self.transform:
            image = self.transform(image)

        return image, labels


class SIMOModel(nn.Module):
    def __init__(self, max_bboxes, arch="vgg"):
        super(SIMOModel, self).__init__()
        self.max_bboxes = max_bboxes
        self.arch = arch

        # Choose the backbone model
        if arch == "vgg":
            self.backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
            backbone_out_features = 512
        elif arch == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone_out_features = 2048
        elif arch == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone_out_features = 512
        elif arch == "fcn":
            self.backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            print(self.backbone)
            backbone_out_features = 25088
        else:
            raise ValueError(
                "Invalid backbone. Choose 'vgg', 'resnet18', 'resnet50' or 'fcn'."
            )

        if arch == "fcn":
            # Fully connected layers
            self.fc = nn.Sequential(
                nn.Linear(backbone_out_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(
                    512, self.max_bboxes * 4
                ),  # Each bbox has 5 outputs: x, y, w, h
            )
        else:
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

            combined_channels = backbone_out_features + 512  # For concatenation

            # Decoders for tool and tooltip bounding box regression
            self.tool_1_decoder = self._create_decoder(combined_channels)
            self.tool_2_decoder = self._create_decoder(combined_channels)
            self.tooltip_1_decoder = self._create_decoder(combined_channels)
            self.tooltip_2_decoder = self._create_decoder(combined_channels)

            # After creating the decoders in the SIMOModel class __init__ method
            if self.max_bboxes == 2:
                for param in self.tool_2_decoder.parameters():
                    param.requires_grad = False
                for param in self.tooltip_2_decoder.parameters():
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
            nn.Conv2d(128, 4, kernel_size=1),  # 4 outputs for bounding box coordinates
            nn.AdaptiveAvgPool2d((1, 1)),  # Pooling to get a 1x1 output
            nn.Flatten(),  # Flatten to shape [batch_size, 4]
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
        if self.arch == "fcn":
            if hasattr(self.backbone, "avgpool"):
                x = self.backbone.features(x)  # for vgg
                x = self.backbone.avgpool(x)  # for vgg
                x = torch.flatten(x, 1)
            else:
                x = self.backbone(x)  # for resnet
                x = torch.flatten(x, 1)

            x = self.fc(x)

            tool_1_pred, tooltip_1_pred = (x[:, :4], x[:, 4:8])
            if self.max_bboxes == 4:
                tool_2_pred, tooltip_2_pred = (x[:, 10:14], x[:, 14:19])
            else:
                tool_2_pred, tooltip_2_pred = tool_1_pred, tooltip_1_pred
        else:
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

            if self.max_bboxes == 4:
                tool_2_pred = self.tool_2_decoder(combined_features)
                tooltip_2_pred = self.tooltip_2_decoder(combined_features)
            else:
                tool_2_pred, tooltip_2_pred = (
                    tool_1_pred,
                    tooltip_1_pred,
                )

        return (
            tool_1_pred,
            tooltip_1_pred,
            tool_2_pred,
            tooltip_2_pred,
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
                            :2
                        ],  # Only tool_1_pred, tooltip_1_pred
                        [tool_labels[:, 0], tooltip_labels[:, 0]],
                    )
                else:
                    loss = self.compute_losses(
                        preds,  # All predictions for 4 bounding boxes
                        [
                            tool_labels[:, 0],
                            tooltip_labels[:, 0],
                            tool_labels[:, 1],
                            tooltip_labels[:, 1],
                        ],
                    )
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                print(f"Batch {batch+1}/{len(train_loader)}, Loss: {loss.item()}")
                torch.cuda.empty_cache()
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
                            :2
                        ],  # Only tool_1_pred, tooltip_1_pred
                        [tool_labels[:, 0], tooltip_labels[:, 0]],
                    )
                else:
                    loss = self.compute_losses(
                        preds,  # All predictions for 4 bounding boxes
                        [
                            tool_labels[:, 0],
                            tooltip_labels[:, 0],
                            tool_labels[:, 1],
                            tooltip_labels[:, 1],
                        ],
                    )
                val_loss += loss.mean().item()

            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        end_time = time.time()
        print(
            f"Time per image: {(end_time - start_time) / len(val_loader):.2f} seconds"
        )

        return val_loss / len(val_loader)

    def compute_losses(self, preds, labels):
        """
        Compute the combined loss for bounding box regression, which includes both IoU and MSE losses.

        Args:
        - preds: List of predictions containing both bounding box coordinates (in xywh).
        - labels: List of labels where each entry contains the true bounding boxes.

        Returns:
        - total_loss: Combined loss value.
        """
        total_loss = 0.0

        for i, pred_bbox_xywh in enumerate(preds):
            if pred_bbox_xywh.shape[-1] != 4:
                try:
                    pred_bbox_xywh = pred_bbox_xywh.view(
                        -1, 4
                    )  # Ensure the shape is correct
                except:
                    pred_bbox_xywh = pred_bbox_xywh.reshape(
                        -1, 4
                    )  # Ensure the shape is correct

            # Compute IoU loss for bounding boxes
            iou_loss_value = self.iou_loss(pred_bbox_xywh, labels[i][:, 1:])

            mse_loss = F.mse_loss(pred_bbox_xywh, labels[i][:, 1:])

            total_loss += iou_loss_value + mse_loss

        if TEST:
            print(f"IOU Loss: {iou_loss_value}, MSE Loss: {mse_loss}")

        return total_loss

    def visualize_bounding_boxes(
        self,
        image,
        tool_1_preds,
        tooltip_1_preds,
        tool_2_preds,
        tooltip_2_preds,
        save_path=None,
    ):
        image = image.cpu().numpy().squeeze()
        image = np.moveaxis(image, 0, -1)
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

        fig, ax = plt.subplots(1)
        ax.axis("off")
        ax.imshow(image)
        fig.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # Define the data for each tool/tooltip type
        if self.max_bboxes == 4:
            items = [
                (tool_1_preds.cpu().numpy(), "red", "Tool #1"),
                (tool_2_preds.cpu().numpy(), "blue", "Tool #2"),
                (tooltip_1_preds.cpu().numpy(), "green", "Tooltip #1"),
                (tooltip_2_preds.cpu().numpy(), "orange", "Tooltip #2"),
            ]
        else:
            items = [
                (tool_1_preds.cpu().numpy(), "blue", "Tool"),
                (tooltip_1_preds.cpu().numpy(), "orange", "Tooltip"),
            ]

        # Iterate through the items to add patches and labels
        for preds, color, label in items:
            cx, cy, w, h = preds[:4]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x1 *= image.shape[1]
            y1 *= image.shape[0]
            w *= image.shape[1]
            h *= image.shape[0]
            # if tool is outside image skip
            if (
                x1 < 0
                or y1 < 0
                or x1 + w > image.shape[1]
                or y1 + h > image.shape[0]
            ):
                continue
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (x1, y1),
                    w,
                    h,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=3,
                    label=f"{label}",
                )
            )
            ax.text(
                x1,
                y1 - 10,
                f"{label}",
                color=color,
                fontsize=14,
            )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        else:
            ax.imshow(image)

        plt.close()

    def load_best_weights(self, weights_folder):
        best_weights = os.path.join(weights_folder, "best.pt")
        if os.path.exists(best_weights):
            self.load_state_dict(torch.load(best_weights))
            print("Loaded best weights")
        else:
            print("No best weights found.")

    def compute_metrics(self, pred_boxes, true_boxes):
        """
        Compute metrics including precision, recall, mAP@0.5, and mAP@0.5:0.95.
        """
        # Ensure input arrays are numpy arrays
        pred_boxes = np.array(pred_boxes)
        true_boxes = np.array(true_boxes)

        # Compute mAP@0.5 and mAP@0.5:0.95
        mAP_50, precisions_50, recalls_50 = self.compute_map(
            pred_boxes, true_boxes, iou_thresholds=[0.5]
        )
        mAP_50_95, _, _ = self.compute_map(
            pred_boxes,
            true_boxes,
            iou_thresholds=np.linspace(0.5, 0.95, 10),
        )

        # Use the last value of precision and recall for each list (final threshold)
        # precision = precisions_50[-1][-1] if precisions_50 else 0
        # recall = recalls_50[-1][-1] if recalls_50 else 0

        return precisions_50, recalls_50, mAP_50, mAP_50_95

    def test(self, val_loader, output_dir, tracking=False):
        os.makedirs(output_dir, exist_ok=True)
        self.eval()

        if tracking:
            prev_tool_centres = None
            prev_tooltip_centres = None

        # Initialize accumulators for metrics calculation
        all_tool_preds, all_tool_labels = [], []
        all_tooltip_preds, all_tooltip_labels = [], []

        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device).float()
                labels = labels.to(device).float()
                (
                    tool_1_pred,
                    tooltip_1_pred,
                    tool_2_pred,
                    tooltip_2_pred,
                ) = self.forward(images)
                # Visualize bounding boxes for each image in the batch
                for j in range(images.size(0)):
                    if tracking and self.max_bboxes == 4:
                        (
                            tool_1_pred[j],
                            tooltip_1_pred[j],
                            tool_2_pred[j],
                            tooltip_2_pred[j],
                            prev_tool_centres,
                            prev_tooltip_centres,
                        ) = self.track_objects(
                            prev_tool_centres,
                            prev_tooltip_centres,
                            tool_1_pred[j],
                            tooltip_1_pred[j],
                            tool_2_pred[j],
                            tooltip_2_pred[j],
                        )
                    self.visualize_bounding_boxes(
                        images[j],
                        tool_1_pred[j],
                        tooltip_1_pred[j],
                        tool_2_pred[j],
                        tooltip_2_pred[j],
                        save_path=os.path.join(
                            output_dir, f"batch_{i+1}_img_{j+1}.png"
                        ),
                    )

                    if TEST:
                        break

                # Append predictions for metrics calculation
                all_tool_preds.extend(tool_1_pred.cpu().numpy())
                all_tooltip_preds.extend(tooltip_1_pred.cpu().numpy())
                all_tool_labels.extend(labels[:, 0, :].cpu().numpy())  # Actual labels

                if self.max_bboxes == 4:
                    all_tooltip_labels.extend(
                        labels[:, 2, :].cpu().numpy()
                    )  # Actual labels

                    all_tool_preds.extend(tool_2_pred.cpu().numpy())
                    all_tool_labels.extend(
                        labels[:, 1, :].cpu().numpy()
                    )  # Actual labels

                    all_tooltip_preds.extend(tooltip_2_pred.cpu().numpy())
                    all_tooltip_labels.extend(
                        labels[:, 3, :].cpu().numpy()
                    )  # Actual labels
                else:
                    all_tooltip_labels.extend(labels[:, 1, :].cpu().numpy())

                if TEST:
                    # Compute temp metrics in this batch only
                    (
                        temp_precisions_tool,
                        temp_recalls_tool,
                        temp_mAP_50_tool,
                        temp_mAP_50_95_tool,
                    ) = self.compute_metrics(all_tool_preds[-2:], all_tool_labels[-2:])
                    print(
                        f"Tool: Precision: {temp_precisions_tool}, Recall: {temp_recalls_tool}, mAP@0.5: {temp_mAP_50_tool}, mAP@0.5:0.95: {temp_mAP_50_95_tool}"
                    )
                    (
                        temp_precisions_tooltip,
                        temp_recalls_tooltip,
                        temp_mAP_50_tooltip,
                        temp_mAP_50_95_tooltip,
                    ) = self.compute_metrics(
                        all_tooltip_preds[-2:], all_tooltip_labels[-2:]
                    )
                    print(
                        f"Tooltip: Precision: {temp_precisions_tooltip}, Recall: {temp_recalls_tooltip}, mAP@0.5: {temp_mAP_50_tooltip}, mAP@0.5:0.95: {temp_mAP_50_95_tooltip}"
                    )
                    break

            # Compute metrics for tools
            precisions_tool, recalls_tool, mAP_50_tool, mAP_50_95_tool = (
                self.compute_metrics(all_tool_preds, all_tool_labels)
            )
            print(
                f"Tool: Precision: {precisions_tool}, Recall: {recalls_tool}, mAP@0.5: {mAP_50_tool}, mAP@0.5:0.95: {mAP_50_95_tool}"
            )

            # Compute metrics for tooltips
            precisions_tooltip, recalls_tooltip, mAP_50_tooltip, mAP_50_95_tooltip = (
                self.compute_metrics(all_tooltip_preds, all_tooltip_labels)
            )
            print(
                f"Tooltip: Precision: {precisions_tooltip}, Recall: {recalls_tooltip}, mAP@0.5: {mAP_50_tooltip}, mAP@0.5:0.95: {mAP_50_95_tooltip}"
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
        tooltip_1_pred,
        tool_2_pred,
        tooltip_2_pred
    ):
        # Compute centres of the predicted bounding boxes in xywh format
        tool_centres = []
        tooltip_centres = []

        # Use the x (centre) coordinate from the predictions
        if tool_1_pred is not None and tool_2_pred is not None:
            tool_centres = [
                tool_1_pred[0].item(),  # Centre x of tool 1
                tool_2_pred[0].item(),  # Centre x of tool 2
            ]
        elif tool_1_pred is not None:
            tool_centres = [tool_1_pred[0].item()]
        elif tool_2_pred is not None:
            tool_centres = [tool_2_pred[0].item()]

        if tooltip_1_pred is not None and tooltip_2_pred is not None:
            tooltip_centres = [
                tooltip_1_pred[0].item(),  # Centre x of tooltip 1
                tooltip_2_pred[0].item(),  # Centre x of tooltip 2
            ]
        elif tooltip_1_pred is not None:
            tooltip_centres = [tooltip_1_pred[0].item()]
        elif tooltip_2_pred is not None:
            tooltip_centres = [tooltip_2_pred[0].item()]

        if prev_tool_centres is None:
            prev_tool_centres = tool_centres
            prev_tooltip_centres = tooltip_centres
        else:
            if tool_centres:
                # Compute distances between previous and current centres for tools
                tool_distances = [
                    abs(prev_tool_centres[0] - tool_centres[0]),
                    (
                        abs(prev_tool_centres[1] - tool_centres[0])
                        if len(tool_centres) == 2
                        else np.inf
                    ),
                ]
                # Assign the closest previous centre to the current one
                if len(tool_centres) == 2:
                    if tool_distances[0] < tool_distances[1]:
                        tool_1_pred, tool_2_pred = tool_1_pred, tool_2_pred
                    else:
                        tool_1_pred, tool_2_pred = tool_2_pred, tool_1_pred
                elif len(tool_centres) == 1:
                    tool_1_pred = (
                        tool_1_pred
                        if prev_tool_centres[0] == tool_centres[0]
                        else tool_2_pred
                    )

            if tooltip_centres:
                # Compute distances between previous and current centres for tooltips
                tooltip_distances = [
                    abs(prev_tooltip_centres[0] - tooltip_centres[0]),
                    (
                        abs(prev_tooltip_centres[1] - tooltip_centres[0])
                        if len(tooltip_centres) == 2
                        else np.inf
                    ),
                ]
                # Assign the closest previous centre to the current one
                if len(tooltip_centres) == 2:
                    if tooltip_distances[0] < tooltip_distances[1]:
                        tooltip_1_pred, tooltip_2_pred = tooltip_1_pred, tooltip_2_pred
                    else:
                        tooltip_1_pred, tooltip_2_pred = tooltip_2_pred, tooltip_1_pred
                elif len(tooltip_centres) == 1:
                    tooltip_1_pred = (
                        tooltip_1_pred
                        if prev_tooltip_centres[0] == tooltip_centres[0]
                        else tooltip_2_pred
                    )

            prev_tool_centres = tool_centres
            prev_tooltip_centres = tooltip_centres

        return (
            tool_1_pred,
            tool_2_pred,
            tooltip_1_pred,
            tooltip_2_pred,
            prev_tool_centres,
            prev_tooltip_centres,
        )

    def iou(self, pred, target, smooth=1e-6):
        """
        Compute the Intersection over Union (IoU) between two sets of boxes.
        """
        cx, cy, w, h = pred[0][0], pred[0][1], pred[0][2], pred[0][3]
        target_cx, target_cy, target_w, target_h = (
            target[0][0],
            target[0][1],
            target[0][2],
            target[0][3],
        )
        
        x1 = cx - w / torch.tensor(2)
        y1 = cy - h / torch.tensor(2)
        x2 = cx + w / torch.tensor(2)
        y2 = cy + h / torch.tensor(2)
        
        target_x1 = target_cx - target_w / torch.tensor(2)
        target_y1 = target_cy - target_h / torch.tensor(2)
        target_x2 = target_cx + target_w / torch.tensor(2)
        target_y2 = target_cy + target_h / torch.tensor(2)
        
        # Compute the intersection area
        xA = torch.max(x1, target_x1)
        yA = torch.max(y1, target_y1)
        xB = torch.min(x2, target_x2)
        yB = torch.min(y2, target_y2)
        
        inter_area = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
        
        # Compute the union area
        box1_area = w * h
        box2_area = target_w * target_h
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area

        return min(1.0, iou)

    def iou_loss(self, pred, target):
        """
        Compute IoU loss.
        """
        iou_score = self.iou(pred, target)
        return 1 - iou_score

    def compute_average_precision_at_iou(self, pred_boxes, gt_boxes, iou_threshold):
        """
        Compute Average Precision (AP) at a given IoU threshold.
        """
        num_preds = len(pred_boxes)
        num_gts = len(gt_boxes)

        tp = np.zeros(num_preds)
        fp = np.zeros(num_preds)
        detected_gt = np.zeros(num_gts)

        # Assuming pred_boxes and gt_boxes are aligned and have the same length
        for i in range(len(pred_boxes)):
            iou = self.iou(pred_boxes[i], gt_boxes[i])

            if iou >= iou_threshold and detected_gt[i] == 0:
                tp[i] = 1
                detected_gt[i] = 1  # Mark this ground truth as detected
            else:
                fp[i] = 1

        # Compute precision and recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / num_gts

        # Compute Average Precision (AP)
        ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])

        return ap, precision, recall

    def compute_map(
        self,
        pred_boxes,
        gt_boxes,
        iou_thresholds=np.linspace(0.5, 0.95, 10),
    ):
        aps = []
        precisions_list = []
        recalls_list = []

        for iou_threshold in iou_thresholds:
            ap, precision, recall = self.compute_average_precision_at_iou(
                pred_boxes, gt_boxes, iou_threshold
            )
            aps.append(ap)
            precisions_list.append(precision)
            recalls_list.append(recall)

        map_score = np.mean(aps)
        return map_score, precisions_list, recalls_list


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
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
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

    image_files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".png")
        ]
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

    model = SIMOModel(max_bboxes=max_bboxes, arch=BACKBONE).to(device)

    train_losses, val_losses = model.train_model(train_loader, val_loader, num_epochs=300, lr=0.001, patience=10)
    
    # Plot training and validation losses
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_plot.png")

    # Load best weights and run evaluation on validation set
    model.load_best_weights(weights_folder)
    results = model.test(
        val_loader=val_loader,
        output_dir=output_dir,
        tracking=tracking,
    )
    print(results)


if __name__ == "__main__":
    main()
