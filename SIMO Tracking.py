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
    def __init__(self, n_classes=4, backbone="vgg", use_focal_loss=False):
        super(SIMOModel, self).__init__()
        self.use_focal_loss = use_focal_loss

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
            tool_1_pred, tool_1_conf,
            tool_2_pred, tool_2_conf,
            tooltip_1_pred, tooltip_1_conf,
            tooltip_2_pred, tooltip_2_conf
        )


    def train_model(self, train_loader, val_loader, num_epochs=300, lr=0.001, patience=3):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        initial_patience = patience
        start_time = time.time()
        total_time = 0
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        checkpoints_folder = "chkpts/SIMO/weights"
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
                tool_targets, tooltip_targets = labels[:, :2, 0].unsqueeze(2), labels[:, 2:, 0].unsqueeze(2)

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
        print(f"Time per image: {(end_time - start_time) / len(val_loader):.2f} seconds")

        return val_loss / len(val_loader)

    def compute_losses(self, preds, labels):
        total_loss = 0.0
        
        for i in range(4):  # Iterating over each tool and tooltip (total 4)
            pred_bbox = preds[i*2]     # Bbox prediction (tool_1_pred, tool_2_pred, ...)
            pred_conf = preds[i*2 + 1] # Conf prediction (tool_1_conf, tool_2_conf, ...)
            
            label_bbox = labels[i][:, 1:]  # Bounding box label
            label_conf = labels[i][:, 0].unsqueeze(1)  # Confidence label (reshaped to match pred_conf)
            
            # Compute IoU loss
            iou_loss_value = iou_loss(pred_bbox, label_bbox)
            
            # Compute Confidence loss
            if self.use_focal_loss:
                conf_loss_value = focal_loss(pred_conf, label_conf)
            else:
                conf_loss_value = F.binary_cross_entropy_with_logits(pred_conf, label_conf)

            total_loss += iou_loss_value + conf_loss_value

        return total_loss


    def compute_metrics(
        self,
        tool_preds,
        tool_labels,
        tool_conf,
        tooltip_preds,
        tooltip_labels,
        tooltip_conf,
    ):
        precisions_tool, recalls_tool, mAP_tool, mAP_50_95_tool = compute_metrics(
            tool_preds, tool_labels, tool_conf
        )
        precisions_tip, recalls_tip, mAP_tip, mAP_50_95_tip = compute_metrics(
            tooltip_preds, tooltip_labels, tooltip_conf
        )

        metrics = {
            "tool": {
                "precisions": precisions_tool.tolist(),
                "recalls": recalls_tool.tolist(),
                "mAP": mAP_tool,
                "mAP_50_95": mAP_50_95_tool,
            },
            "tooltip": {
                "precisions": precisions_tip.tolist(),
                "recalls": recalls_tip.tolist(),
                "mAP": mAP_tip,
                "mAP_50_95": mAP_50_95_tip,
            },
        }
        return metrics

    def visualize_bounding_boxes(
        self, image, tool_preds, tooltip_preds, save_path=None
    ):
        image = image.cpu().numpy().squeeze()
        image = np.moveaxis(image, 0, -1)
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

        tool_preds = tool_preds.cpu().numpy().squeeze()
        tooltip_preds = tooltip_preds.cpu().numpy().squeeze()

        fig, ax = plt.subplots()
        plt.axis("off")

        tool_x, tool_y, tool_w, tool_h = tool_preds
        tooltip_x, tooltip_y, tooltip_w, tooltip_h = tooltip_preds

        tool_rect = matplotlib.patches.Rectangle(
            (tool_x, tool_y), tool_w, tool_h, edgecolor="r", facecolor="none"
        )
        tooltip_rect = matplotlib.patches.Rectangle(
            (tooltip_x, tooltip_y),
            tooltip_w,
            tooltip_h,
            edgecolor="g",
            facecolor="none",
        )

        ax.add_patch(tool_rect)
        ax.add_patch(tooltip_rect)

        ax.imshow(image)
        if save_path:
            plt.savefig(save_path)
        else:
            ax.imshow(image)

        plt.close()

    def track_on_images(self, image_folder, save_folder="chkpts/SIMO/tracking"):
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        images = []
        for i in range(1, 10):
            image = cv2.imread(f"{image_folder}/{i}.png")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (512, 512))
            image = image / 255.0
            image = (
                torch.tensor(image, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            images.append(image)

        tracks = self.track_objects(images)
        print(tracks)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for i, image in enumerate(images):
            image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            for box, conf in tracks[i]:
                x, y, w, h = box
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            save_path = os.path.join(save_folder, f"frame_{i+1}.png")
            cv2.imwrite(save_path, image)

    def track_on_videos(self, video_folder, save_folder="chkpts/SIMO/tracking_videos"):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for video_file in os.listdir(video_folder):
            if not video_file.endswith(".mp4"):
                continue

            cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            out = cv2.VideoWriter(
                os.path.join(save_folder, video_file),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (512, 512))
                frame_resized = frame_resized / 255.0
                frame_tensor = (
                    torch.tensor(frame_resized, dtype=torch.float32)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
                frames.append(frame_tensor)

                if len(frames) == 16 or len(frames) == frame_count:
                    tracks = self.track_objects(frames)
                    for i, frame in enumerate(frames):
                        frame = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        for box, conf in tracks[i]:
                            x, y, w, h = box
                            x, y, w, h = int(x), int(y), int(w), int(h)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        out.write(frame)

                    frames = []

            cap.release()
            out.release()

    def track_objects(self, images):
        tracker = ObjectTracker()  # Instantiate the tracker

        for frame in images:
            tool_boxes, tool_confs, tooltip_boxes, tooltip_confs = self.detect_objects(
                frame
            )

            # Update tracker with detected boxes and confidences
            tracker.update(tool_boxes, tool_confs, tooltip_boxes, tooltip_confs)

        return tracker.get_tracks()  # Get the tracked object paths

    def load_best_weights(self, weights_folder="chkpts/SIMO/weights"):
        best_weights = os.path.join(weights_folder, "best.pt")
        if os.path.exists(best_weights):
            self.load_state_dict(torch.load(best_weights))
            print("Loaded best weights")
        else:
            print("No best weights found.")

    def run_on_test_images(
        self, test_image_folder="data/ART-Net/images/val", num_pos=20, num_neg=5
    ):
        test_images = []
        pos_images = [
            os.path.join(test_image_folder, f)
            for f in os.listdir(test_image_folder)
            if "Pos" in f
        ]
        neg_images = [
            os.path.join(test_image_folder, f)
            for f in os.listdir(test_image_folder)
            if "Neg" in f
        ]
        pos_images = np.random.choice(pos_images, num_pos, replace=False).tolist()
        neg_images = np.random.choice(neg_images, num_neg, replace=False).tolist()
        test_images.extend(pos_images + neg_images)

        for image_path in test_images:
            image = Image.open(image_path)
            image = functional.to_tensor(image)
            image = functional.resize(image, (512, 512))
            image = image.unsqueeze(0).to(device)

            self.eval()
            with torch.no_grad():
                tool_preds, tool_conf, tooltip_preds, tooltip_conf = self.forward(image)
            self.visualize_bounding_boxes(
                image,
                tool_preds,
                tooltip_preds,
                save_path=image_path.replace(".png", "_bbox.png"),
            )

    def validate_on_images(self, val_images):
        self.eval()
        with torch.no_grad():
            for val_image in val_images:
                image = Image.open(val_image)
                image = functional.to_tensor(image)
                image = functional.resize(image, (512, 512))
                image = image.unsqueeze(0).to(device)

                tool_preds, tool_conf, tooltip_preds, tooltip_conf = self.forward(image)
                self.visualize_bounding_boxes(image, tool_preds, tooltip_preds)


# Define the ObjectTracker class
class ObjectTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xx1 = max(x1, x1g)
        yy1 = max(y1, y1g)
        xx2 = min(x2, x2g)
        yy2 = min(y2, y2g)

        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)

        inter_area = w * h
        union_area = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - inter_area

        return inter_area / union_area

    def update(self, tool_boxes, tool_confs, tooltip_boxes, tooltip_confs):
        self.frame_count += 1

        # Convert detections to numpy for easier manipulation
        tool_boxes = tool_boxes.cpu().detach().numpy()
        tool_confs = tool_confs.cpu().detach().numpy()
        tooltip_boxes = tooltip_boxes.cpu().detach().numpy()
        tooltip_confs = tooltip_confs.cpu().detach().numpy()

        detections = list(zip(tool_boxes, tool_confs)) + list(
            zip(tooltip_boxes, tooltip_confs)
        )

        # Implement DeepSORT
        if self.frame_count > 1:
            if len(self.trackers) == 0:
                previous_detections = self.trackers[-2]

                cost_matrix = np.zeros((len(previous_detections), len(detections)))
                for i, prev in enumerate(previous_detections):
                    for j, detection in enumerate(detections):
                        cost_matrix[i, j] = 1 - self.iou(prev[0], detection[0])

                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for i, j in zip(row_ind, col_ind):
                    self.trackers[-2][i] = detections[j]
            else:
                self.trackers.append(detections)

        if self.frame_count > self.max_age:
            self.trackers.pop(0)

        self.trackers.append(detections)

    def get_tracks(self):
        return self.trackers


# Define loss functions
def iou(pred, target, smooth=1e-6):
    pred, target = pred.to(device), target.to(device)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def iou_loss(preds, labels):
    """
    Custom loss function to handle different sized predictions and labels.
    Penalizes for missing predictions, unmatched labels, and negative predictions.

    Args:
        preds: Tensor of predicted bounding boxes and confidences [batch_size, 4].
        labels: Tensor of ground truth bounding boxes and confidences [batch_size, 4].

    Returns:
        loss: Computed loss value.
    """
    (
        tool_1_pred,
        tool_1_conf,
        tool_2_pred,
        tool_2_conf,
        tooltip_1_pred,
        tooltip_1_conf,
        tooltip_2_pred,
        tooltip_2_conf,
    ) = preds

    total_iou_loss = 0.0

    for pred, label in zip(preds, labels):
        if label.size(0) > 0:
            iou_loss = 1 - iou(pred, label)
        else:
            iou_loss = torch.tensor(0.0, device=device)

        total_iou_loss += iou_loss
        
    return total_iou_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


focal_loss = FocalLoss()


def compute_metrics(pred_boxes, true_boxes, conf_scores, iou_threshold=0.5):
    binary_true_boxes = []
    binary_pred_boxes = []
    valid_conf_scores = []

    for i in range(len(true_boxes)):
        iou_scores = [
            iou(pred_boxes[i].cpu(), true_boxes[j].cpu())
            for j in range(len(true_boxes))
        ]
        max_iou = max(iou_scores)
        if max_iou > iou_threshold:
            binary_true_boxes.append(1)
            binary_pred_boxes.append(1)
        else:
            binary_true_boxes.append(0)
            binary_pred_boxes.append(0)
        valid_conf_scores.append(conf_scores[i].cpu().item())  # Convert to scalar

    binary_true_boxes = np.array(binary_true_boxes)
    binary_pred_boxes = np.array(binary_pred_boxes)
    valid_conf_scores = np.array(valid_conf_scores)

    precisions, recalls, _ = precision_recall_curve(
        binary_true_boxes, valid_conf_scores
    )
    ap50 = average_precision_score(binary_true_boxes, valid_conf_scores)

    iou_thresholds = np.linspace(0.5, 0.95, 10)
    aps = []
    for threshold in iou_thresholds:
        binary_true_boxes = []
        binary_pred_boxes = []
        for i in range(len(true_boxes)):
            iou_scores = [
                iou(pred_boxes[i].cpu(), true_boxes[j].cpu())
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

    model = SIMOModel(n_classes=4, backbone="vgg", use_focal_loss=False).to(device)

    model.train_model(train_loader, val_loader, num_epochs=300, lr=0.001, patience=10)

    # # Load best weights and run evaluation on validation set
    # model.load_best_weights()
    # val_images = [
    #     "data/ART-Net/images/train/Train_Pos_sample_0001.png",
    #     "data/ART-Net/images/train/Train_Neg_sample_0002.png",
    # ]
    # model.validate_on_images(val_images)
    # model.run_on_test_images(
    #     test_image_folder="data/ART-Net/images/val", num_pos=20, num_neg=5
    # )


if __name__ == "__main__":
    main()
