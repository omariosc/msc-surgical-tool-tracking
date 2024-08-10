import os
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the ToolDataset class
class ToolDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, max_bboxes=4):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.max_bboxes = max_bboxes
        self.image_files = [
            f
            for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
            and os.path.exists(os.path.join(label_dir, f.replace(".png", ".txt")))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".png", ".txt"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))  # Resize images to 512x512
        image = image / 255.0

        # Load labels and handle cases with empty or partial labels
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            labels = np.loadtxt(label_path, delimiter=" ")
            if labels.ndim == 1:
                labels = labels[np.newaxis, :]
        else:
            labels = np.zeros((0, 5))  # No bounding boxes

        # Pad labels to fixed size, ensuring 2 tools and 2 tool tips (4 total)
        labels = self.pad_labels(labels, self.max_bboxes)

        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

    def pad_labels(self, labels, max_bboxes):
        if len(labels) > max_bboxes:
            labels = labels[:max_bboxes]
        else:
            padding = np.zeros((max_bboxes - len(labels), 5))
            labels = np.vstack((labels, padding))
        return labels


class SIMOModel(nn.Module):
    def __init__(self, n_classes=4, backbone="vgg"):
        super(SIMOModel, self).__init__()

        # Choose the backbone model
        if backbone == "vgg":
            self.backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
            backbone_out_channels = 512
        elif backbone == "resnet":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_out_channels = 2048
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

        # Decoder for tool and tooltip bounding box regression
        self.tool_1_decoder = self._create_decoder(combined_channels, n_classes)
        self.tool_2_decoder = self._create_decoder(combined_channels, n_classes)
        self.tooltip_1_decoder = self._create_decoder(combined_channels, n_classes)
        self.tooltip_2_decoder = self._create_decoder(combined_channels, n_classes)

        # Confidence prediction for tool and tooltip
        self.tool_1_confidence = self._create_confidence_head(combined_channels)
        self.tool_2_confidence = self._create_confidence_head(combined_channels)
        self.tooltip_1_confidence = self._create_confidence_head(combined_channels)
        self.tooltip_2_confidence = self._create_confidence_head(combined_channels)

    def _create_decoder(self, combined_channels, n_classes):
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
                128, n_classes, kernel_size=1
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
        tool_2_pred = self.tool_2_decoder(combined_features)
        tooltip_1_pred = self.tooltip_1_decoder(combined_features)
        tooltip_2_pred = self.tooltip_2_decoder(combined_features)

        # Confidence predictions
        tool_1_conf = self.tool_1_confidence(combined_features)
        tool_2_conf = self.tool_2_confidence(combined_features)
        tooltip_1_conf = self.tooltip_1_confidence(combined_features)
        tooltip_2_conf = self.tooltip_2_confidence(combined_features)

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
        checkpoints_folder = "chkpts/SIMO/ART/weights"
        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)

        best_val_loss = np.inf
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for batch, (images, labels) in enumerate(train_loader):
                images = images.to(device).float()

                # Split labels for tools and tooltips
                tool_labels, tooltip_labels = labels[:, :2, :], labels[:, 2:, :]
                tool_labels = tool_labels.to(device).float()
                tooltip_labels = tooltip_labels.to(device).float()

                optimizer.zero_grad()

                preds = self.forward(images)

                # The preds list has the order:
                # [tool_1_pred, tool_1_conf, tool_2_pred, tool_2_conf, tooltip_1_pred, tooltip_1_conf, tooltip_2_pred, tooltip_2_conf]
                loss = self.compute_losses(
                    preds,
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
                torch.cuda.empty_cache()

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

                # Split labels for tools and tooltips
                tool_labels, tooltip_labels = labels[:, :2, 1:], labels[:, 2:, 1:]
                tool_targets, tooltip_targets = labels[:, :2, 0].unsqueeze(2), labels[
                    :, 2:, 0
                ].unsqueeze(2)

                tool_labels = tool_labels.to(device).float()
                tooltip_labels = tooltip_labels.to(device).float()
                tool_targets = tool_targets.to(device).float()
                tooltip_targets = tooltip_targets.to(device).float()

                preds = self.forward(images)

                loss = self.compute_losses(
                    preds,
                    [
                        tool_labels[:, 0],
                        tool_labels[:, 1],
                        tooltip_labels[:, 0],
                        tooltip_labels[:, 1],
                    ],
                )
                val_loss += loss.item()

            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        end_time = time.time()
        print(
            f"Time per image: {(end_time - start_time) / len(val_loader):.2f} seconds"
        )

        return val_loss / len(val_loader)

    def compute_losses(self, preds, labels):
        """
        Compute the combined loss for bounding box regression and confidence prediction.

        Args:
        - preds: List of predictions containing both bounding box coordinates and confidence scores.
        - labels: List of labels where each entry contains the true bounding boxes and confidence scores.

        Returns:
        - total_loss: Combined loss value.
        """
        total_loss = 0.0

        for i in range(4):  # Iterating over each tool and tooltip (total 4 pairs)
            pred_bbox = preds[
                2 * i
            ]  # Bounding box prediction (tool_1_pred, tool_2_pred, ...)
            pred_conf = preds[
                2 * i + 1
            ]  # Confidence prediction (tool_1_conf, tool_2_conf, ...)

            label_bbox = labels[i][
                :, 1:
            ]  # Bounding box label (skip the first column which is confidence)
            label_conf = labels[i][:, 0].unsqueeze(
                1
            )  # Confidence label (first column), reshape to match pred_conf

            # Compute IoU loss for bounding boxes
            iou_loss_value = self.iou_loss(pred_bbox, label_bbox)

            conf_loss_value = F.binary_cross_entropy_with_logits(
                pred_conf, label_conf
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

        tool_1_preds = tool_1_preds.cpu().numpy().T
        tool_2_preds = tool_2_preds.cpu().numpy().T
        tooltip_1_preds = tooltip_1_preds.cpu().numpy().T
        tooltip_2_preds = tooltip_2_preds.cpu().numpy().T

        fig, ax = plt.subplots(1)
        ax.axis("off")
        ax.imshow(image)
        fig.tight_layout(pad=0)
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
                try:
                    x, y, w, h = map(
                        int, pred
                    )  # Convert all values to integers at once
                    ax.add_patch(
                        matplotlib.patches.Rectangle(
                            (x, y),
                            w,
                            h,
                            edgecolor=color,
                            facecolor="none",
                            linewidth=2,
                            label=f"{label}, Conf: {conf:.2f}",
                        )
                    )
                    ax.text(x, y, f"{label}, {conf:.2f}", color=color)
                except:
                    pass

        if save_path:
            plt.savefig(save_path)
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

    def compute_metrics(pred_boxes, true_boxes, conf_scores, iou_thresholds=[0.5]):
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
                    self.iou(pred_boxes[i], true_boxes[j]) for j in range(len(true_boxes))
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
                        tool_2_pred,
                        tooltip_1_pred,
                        tooltip_2_pred,
                        prev_tool_centres,
                        prev_tooltip_centres,
                    ) = self.track_objects(
                        prev_tool_centres,
                        prev_tooltip_centres,
                        tool_1_pred,
                        tool_2_pred,
                        tooltip_1_pred,
                        tooltip_2_pred,
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
        tool_2_pred,
        tooltip_1_pred,
        tooltip_2_pred,
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
                        tool_1_pred, tool_2_pred = (
                            (
                                tool_1_pred
                                if tool_distances[0][0] < tool_distances[1][0]
                                else tool_2_pred
                            ),
                            (
                                tool_2_pred
                                if tool_distances[0][0] < tool_distances[1][0]
                                else tool_1_pred
                            ),
                        )
                    elif len(tool_centres) == 1:
                        tool_1_pred = (
                            tool_1_pred
                            if prev_tool_centres[0] == tool_centres[0]
                            else tool_2_pred
                        )

                if tooltip_centres:
                    # Compute distances between previous and current centres for tooltips
                    tooltip_distances = distance.cdist(
                        prev_tooltip_centres, tooltip_centres, "euclidean"
                    )

                    # Assign the closest previous centre to the current one
                    if len(tooltip_centres) == 2:
                        tooltip_1_pred, tooltip_2_pred = (
                            (
                                tooltip_1_pred
                                if tooltip_distances[0][0] < tooltip_distances[1][0]
                                else tooltip_2_pred
                            ),
                            (
                                tooltip_2_pred
                                if tooltip_distances[0][0] < tooltip_distances[1][0]
                                else tooltip_1_pred
                            ),
                        )
                    elif len(tooltip_centres) == 1:
                        tooltip_1_pred = (
                            tooltip_1_pred
                            if prev_tooltip_centres[0] == tooltip_centres[0]
                            else tooltip_2_pred
                        )

                prev_tool_centres = tool_centres
                prev_tooltip_centres = tooltip_centres

        except Exception as e:
            print(f"Error during tracking: {e}")

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

    # process_images(
    #     "data/6DOF/images/val",
    #     "data/6DOF/processed_images/val",
    #     "data/6DOF/labels/val",
    #     "data/6DOF/processed_labels/val",
    # )

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = ToolDataset(
        image_dir="data/ART-Net/images/train",
        label_dir="data/ART-Net/labels/train",
        transform=transform,
    )
    val_dataset = ToolDataset(
        image_dir="data/ART-Net/images/val",
        label_dir="data/ART-Net/labels/val",
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = SIMOModel(n_classes=4, backbone="vgg").to(device)

    # model.train_model(train_loader, val_loader, num_epochs=300, lr=0.001, patience=3)

    # Load best weights and run evaluation on validation set
    model.load_best_weights("chkpts/SIMO/ART/weights")
    results = model.test(
        input_dir="data/ART-Net/images/val",
        output_dir="chkpts/SIMO/ART/output",
        tracking=False,
    )
    print(results)


if __name__ == "__main__":
    main()
