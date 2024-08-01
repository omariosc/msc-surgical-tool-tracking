import os
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
            and not "Neg" in f
            and os.path.exists(os.path.join(label_dir, f.replace(".png", ".txt")))
            and os.path.getsize(os.path.join(label_dir, f.replace(".png", ".txt"))) > 1
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

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            labels = np.loadtxt(label_path, delimiter=" ")
            if labels.ndim == 1:
                labels = labels[np.newaxis, :]
        else:
            labels = np.zeros((0, 5))  # No bounding boxes

        # Pad labels to fixed size
        if len(labels) > self.max_bboxes:
            labels = labels[: self.max_bboxes]
        else:
            padding = np.zeros((self.max_bboxes - len(labels), 5))
            labels = np.vstack((labels, padding))

        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels


# Define the SIMOModel class
class SIMOModel(nn.Module):
    def __init__(self, n_classes=4):
        super(SIMOModel, self).__init__()
        self.vgg_base = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features

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

        # Decoder for tool bounding box regression
        self.tool_decoder = nn.Sequential(
            nn.Conv2d(
                1024, 512, kernel_size=3, padding=1
            ),  # 1024 channels due to concatenation of VGG and FRG
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

        # Decoder for tooltip bounding box regression
        self.tooltip_decoder = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
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

        # Confidence prediction for tool
        self.tool_confidence = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Sigmoid(),  # Confidence between 0 and 1
        )

        # Confidence prediction for tooltip
        self.tooltip_confidence = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Sigmoid(),  # Confidence between 0 and 1
        )

    def forward(self, x):
        # VGG base layers
        block1_pool = self.vgg_base[:5](x)  # block1_pool
        block2_pool = self.vgg_base[5:10](block1_pool)  # block2_pool
        block3_pool = self.vgg_base[10:17](block2_pool)  # block3_pool
        block4_pool = self.vgg_base[17:24](block3_pool)  # block4_pool
        x_vgg = self.vgg_base[24:](block4_pool)  # block5_pool

        # Feature representation generator
        frg = self.frg(x)

        # Resize FRG output to match VGG output size
        frg_resized = F.interpolate(
            frg,
            size=(x_vgg.size(2), x_vgg.size(3)),
            mode="bilinear",
            align_corners=True,
        )

        # Concatenate VGG and FRG outputs for both tool and tooltip
        combined_features = torch.cat((x_vgg, frg_resized), dim=1)

        # Decoder for tool and tooltip
        tool_output = self.tool_decoder(combined_features)
        tooltip_output = self.tooltip_decoder(combined_features)

        # Confidence predictions
        tool_conf = self.tool_confidence(combined_features)
        tooltip_conf = self.tooltip_confidence(combined_features)

        return tool_output, tool_conf, tooltip_output, tooltip_conf

    def detect_objects(self, x):
        tool_boxes, tool_confs, tooltip_boxes, tooltip_confs = self.forward(x)

        # Filter out boxes with low confidence
        tool_indices = torch.where(tool_confs > 0.5)[0]
        tooltip_indices = torch.where(tooltip_confs > 0.5)[0]

        tool_boxes = tool_boxes[tool_indices]
        tool_confs = tool_confs[tool_indices]

        tooltip_boxes = tooltip_boxes[tooltip_indices]
        tooltip_confs = tooltip_confs[tooltip_indices]

        return tool_boxes, tool_confs, tooltip_boxes, tooltip_confs

    def track_objects(self, images):
        tracker = ObjectTracker()  # Instantiate the tracker

        for frame in images:
            tool_boxes, tool_confs, tooltip_boxes, tooltip_confs = self.detect_objects(
                frame
            )

            # Update tracker with detected boxes and confidences
            tracker.update(tool_boxes, tool_confs, tooltip_boxes, tooltip_confs)

        return tracker.get_tracks()  # Get the tracked object paths


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
            # Get the previous detections
            previous_detections = self.trackers[-2]

            # Create cost matrix
            cost_matrix = np.zeros((len(previous_detections), len(detections)))
            for i, prev in enumerate(previous_detections):
                for j, detection in enumerate(detections):
                    cost_matrix[i, j] = 1 - self.iou(prev[0], detection[0])

            # Perform Hungarian algorithm to assign detections to trackers
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Update the trackers with the new detections
            for i, j in zip(row_ind, col_ind):
                self.trackers[-2][i] = detections[j]

        # Remove old tracks
        if self.frame_count > self.max_age:
            self.trackers.pop(0)

        # Add the detection to trackers (simplified for example)
        self.trackers.append(detections)

    def get_tracks(self):
        return self.trackers


# Define loss functions
def iou(pred, target, smooth=1e-6):
    # Ensure values are in the correct range
    assert torch.all(pred >= 0) and torch.all(pred <= 1), "Predictions out of range"
    assert torch.all(target >= 0) and torch.all(target <= 1), "Targets out of range"

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def iou_loss(pred, target):
    return 1 - iou(pred, target)


def bce_iou_loss(pred, target):
    bce_loss = F.binary_cross_entropy(pred, target, reduction="none")
    iou_loss_value = iou(torch.sigmoid(pred), target)
    return bce_loss + iou_loss_value


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


focal_loss = FocalLoss()


# Function to compute evaluation metrics
def compute_metrics(pred_boxes, true_boxes, conf_scores, iou_threshold=0.5):
    pred_boxes = pred_boxes.cpu().detach().numpy()
    true_boxes = true_boxes.cpu().detach().numpy()
    conf_scores = conf_scores.cpu().detach().numpy()

    precisions, recalls, _ = precision_recall_curve(
        true_boxes.ravel(), conf_scores.ravel()
    )
    ap50 = average_precision_score(true_boxes.ravel(), conf_scores.ravel())

    # mAP calculation (AP for IoU > 0.5)
    mAP = ap50

    # mAP 50-95 calculation (should calculate over IoU thresholds from 0.5 to 0.95)
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    aps = []
    for iou_threshold in iou_thresholds:
        tp, fp, fn = 0, 0, 0
        for i in range(len(pred_boxes)):
            iou_scores = [
                iou(pred_boxes[i], true_boxes[j]) for j in range(len(true_boxes))
            ]
            max_iou = max(iou_scores)
            if max_iou > iou_threshold:
                tp += 1
            else:
                fp += 1

        ap = tp / (tp + fp)
        aps.append(ap)

    mAP_50_95 = np.mean(aps)

    return precisions, recalls, mAP, mAP_50_95


# Main function
def main():
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor()])

    # Create datasets
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

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Move model to the appropriate device
    model = SIMOModel(n_classes=4).to(device)

    # Checkpoints folder
    checkpoints_folder = "chkpts/SIMO/weights"
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    patience = 5
    best_val_loss = np.inf
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch, (images, labels) in enumerate(train_loader):
            images = images.to(device).float()

            # Extract labels for tool and tooltip
            tool_labels = labels[:, 0, 1:].to(device).float()
            tooltip_labels = labels[:, 1, 1:].to(device).float()
            tool_targets = labels[:, 0, 0].unsqueeze(1).to(device).float()
            tooltip_targets = labels[:, 1, 0].unsqueeze(1).to(device).float()

            optimizer.zero_grad()

            # Forward pass
            tool_preds, tool_conf, tooltip_preds, tooltip_conf = model(images)

            if tool_labels.size(0) > 0:
                loss_tool = bce_iou_loss(tool_preds, tool_labels)
                conf_loss_tool = focal_loss(tool_conf, tool_targets)
            else:
                conf_loss_tool = focal_loss(
                    tool_conf, torch.zeros_like(tool_conf, device=device)
                )

            if tooltip_labels.size(0) > 0:
                loss_tooltip = bce_iou_loss(tooltip_preds, tooltip_labels)
                conf_loss_tooltip = focal_loss(tooltip_conf, tooltip_targets)
            else:
                conf_loss_tooltip = focal_loss(
                    tooltip_conf, torch.zeros_like(tooltip_conf, device=device)
                )

            loss = loss_tool + loss_tooltip + conf_loss_tool + conf_loss_tooltip

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            break

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        all_tool_preds, all_tool_labels, all_tool_conf = [], [], []
        all_tooltip_preds, all_tooltip_labels, all_tooltip_conf = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device).float()

                tool_labels = labels[:, 0, 1:].to(device).float()
                tooltip_labels = labels[:, 1, 1:].to(device).float()
                tool_targets = labels[:, 0, 0].unsqueeze(1).to(device).float()
                tooltip_targets = labels[:, 1, 0].unsqueeze(1).to(device).float()

                tool_preds, tool_conf, tooltip_preds, tooltip_conf = model(images)

                if tool_labels.size(0) > 0:
                    loss_tool = bce_iou_loss(tool_preds, tool_labels)
                    conf_loss_tool = focal_loss(tool_conf, tool_targets)
                else:
                    conf_loss_tool = focal_loss(
                        tool_conf, torch.zeros_like(tool_conf, device=device)
                    )

                if tooltip_labels.size(0) > 0:
                    loss_tooltip = bce_iou_loss(tooltip_preds, tooltip_labels)
                    conf_loss_tooltip = focal_loss(tooltip_conf, tooltip_targets)
                else:
                    conf_loss_tooltip = focal_loss(
                        tooltip_conf, torch.zeros_like(tooltip_conf, device=device)
                    )

                loss = loss_tool + loss_tooltip + conf_loss_tool + conf_loss_tooltip
                val_loss += loss.item()

                # Collect predictions and labels for metrics
                all_tool_preds.append(tool_preds)
                all_tool_labels.append(tool_labels)
                all_tool_conf.append(tool_conf)
                all_tooltip_preds.append(tooltip_preds)
                all_tooltip_labels.append(tooltip_labels)
                all_tooltip_conf.append(tooltip_conf)
            
                break
            
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}")

        # Save model checkpoints
        torch.save(model.state_dict(), f"{checkpoints_folder}/epoch{epoch+1}.pt")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{checkpoints_folder}/best.pt")

        # Early stopping
        if avg_val_loss > best_val_loss:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                break
        else:
            patience = 10
            
        break

    # Evaluation Metrics
    tool_preds = torch.cat(all_tool_preds)
    tool_labels = torch.cat(all_tool_labels)
    tool_conf = torch.cat(all_tool_conf)
    tooltip_preds = torch.cat(all_tooltip_preds)
    tooltip_labels = torch.cat(all_tooltip_labels)
    tooltip_conf = torch.cat(all_tooltip_conf)

    precisions_tool, recalls_tool, mAP_tool, mAP_50_95_tool = compute_metrics(
        tool_preds, tool_labels, tool_conf
    )
    print(f"Tool mAP50: {mAP_tool}, mAP50-95: {mAP_50_95_tool}")

    precisions_tip, recalls_tip, mAP_tip, mAP_50_95_tip = compute_metrics(
        tooltip_preds, tooltip_labels, tooltip_conf
    )
    print(f"Tooltip mAP50: {mAP_tip}, mAP50-95: {mAP_50_95_tip}")

    # Store all evaluation metrics in a JSON file in chkpts/SIMO/metrics.json
    with open("chkpts/SIMO/metrics.json", "w") as f:
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
        
        json.dump(metrics, f)        

    # Plot Precision-Recall Curve
    plt.figure(figsize=(12, 8))
    plt.plot(recalls_tool, precisions_tool, label="Tool")
    plt.plot(recalls_tip, precisions_tip, label="Tooltip")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Tool and Tooltip")
    plt.grid(True)
    plt.savefig("chkpts/SIMO/precision_recall_curve.png")
    plt.show()

    # Function to visualize bounding boxes on an image
    def visualize_bounding_boxes(image, tool_preds, tooltip_preds, save_path=None):
        image = image.cpu().numpy().squeeze()
        image = np.moveaxis(image, 0, -1)
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

        tool_preds = tool_preds.cpu().numpy().squeeze()
        tooltip_preds = tooltip_preds.cpu().numpy().squeeze()

        fig, ax = plt.subplots()
        plt.axis("off")
        ax.imshow(image)

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

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    # Test the model on sample image
    image_path = "data/ART-Net/images/train/Train_Pos_sample_0001.png"
    image = Image.open(image_path)
    image = F.to_tensor(image)
    image = functional.resize(image, (512, 512))
    image = image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        tool_preds, tool_conf, tooltip_preds, tooltip_conf = model(image)

    visualize_bounding_boxes(
        image, tool_preds, tooltip_preds, save_path="chkpts/SIMO/bounding_boxes.png"
    )

    # Run tracking on a sequence of images
    image_folder = "data/6DOF/Test 6 png"
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

    tracks = model.track_objects(images)
    print(tracks)

    # Create directory to store tracking results
    tracking_results_dir = "chkpts/SIMO/tracking"
    if not os.path.exists(tracking_results_dir):
        os.makedirs(tracking_results_dir)

    for i, image in enumerate(images):
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for box, conf in tracks[i]:
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        save_path = os.path.join(tracking_results_dir, f"frame_{i+1}.png")
        cv2.imwrite(save_path, image)


if __name__ == "__main__":
    main()
