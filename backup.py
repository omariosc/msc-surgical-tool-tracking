# Check how many images in data/ART-Net/images/test
import os
import cv2

path = "data 2/ART-Net/images/test"
files = os.listdir(path)
count = 0
for file in files:
    if file.endswith(".png") and "Pos" not in file:
        count += 1
print(count)

# 308 test images 154 pos and 154 neg
# 1324 train images 662 pos and 662 neg

# Define the paths
base_path = "data 2/ART-Net/"
train_positive_images_path = os.path.join(base_path, "Train/Train_Positive")
test_positive_images_path = os.path.join(base_path, "Test/Test_Positive")
train_negative_images_path = os.path.join(base_path, "Train/Train_Negative")
test_negative_images_path = os.path.join(base_path, "Test/Test_Negative")

train_label_path = os.path.join(base_path, "labels/train")
test_label_path = os.path.join(base_path, "labels/test")
train_image_dest = os.path.join(base_path, "images/train")
test_image_dest = os.path.join(base_path, "images/test")

val_image_dest = os.path.join(base_path, "images/val")
val_label_dest = os.path.join(base_path, "labels/val")

# Select 154 pos and 154 neg from train images at random to add to val images and labels folder
import random

random.seed(42)

train_pos_files = os.listdir(train_positive_images_path)
train_neg_files = os.listdir(train_negative_images_path)
train_pos_files = random.sample(train_pos_files, 154)
train_neg_files = random.sample(train_neg_files, 154)

# Move the selected images to val images folder (and corresponding label file)
import shutil

for file in train_pos_files:
    print(file)
    shutil.move(
        os.path.join(train_positive_images_path, file),
        os.path.join(val_image_dest, file),
    )
    label_file = file.replace(".png", ".txt")
    shutil.move(
        os.path.join(train_label_path, label_file),
        os.path.join(val_label_dest, label_file),
    )


for file in train_neg_files:
    print(file)
    shutil.move(
        os.path.join(train_negative_images_path, file),
        os.path.join(val_image_dest, file),
    )
    label_file = file.replace(".png", ".txt")
    shutil.move(
        os.path.join(train_label_path, label_file),
        os.path.join(val_label_dest, label_file),
    )

___________


from calendar import c
import os
import cv2
import numpy as np

# Define the paths
base_path = "data 2/ART-Net/"
train_positive_images_path = os.path.join(base_path, "Train/Train_Positive")
test_positive_images_path = os.path.join(base_path, "Test/Test_Positive")
train_negative_images_path = os.path.join(base_path, "Train/Train_Negative")
test_negative_images_path = os.path.join(base_path, "Test/Test_Negative")

train_mask_path = os.path.join(base_path, "Train/Train_Positive_Tool_Mask")
test_mask_path = os.path.join(base_path, "Test/Test_Positive_Tool_Mask")
train_tip_mask_path = os.path.join(base_path, "Train/Train_Positive_TipPoint")
test_tip_mask_path = os.path.join(base_path, "Test/Test_Positive_TipPoint")

train_label_path = os.path.join(base_path, "labels/train")
test_label_path = os.path.join(base_path, "labels/test")
train_image_dest = os.path.join(base_path, "images/train")
test_image_dest = os.path.join(base_path, "images/test")

# Make sure label and image directories exist
os.makedirs(train_label_path, exist_ok=True)
os.makedirs(test_label_path, exist_ok=True)
os.makedirs(train_image_dest, exist_ok=True)
os.makedirs(test_image_dest, exist_ok=True)


def create_bounding_box(mask):
    # Get the indices of non-black pixels
    non_zero_indices = np.nonzero(mask)
    if len(non_zero_indices[0]) == 0:
        return None
    x_min = np.min(non_zero_indices[1])
    y_min = np.min(non_zero_indices[0])
    x_max = np.max(non_zero_indices[1])
    y_max = np.max(non_zero_indices[0])
    return x_min, y_min, x_max - x_min, y_max - y_min


def create_yolo_label(mask_path, label_file, class_id, img_w, img_h):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return
    # Resize the mask to match the image size
    mask = cv2.resize(mask, (img_w, img_h))
    bbox = create_bounding_box(mask)
    if bbox is None:
        return
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    # # Plot the bounding box on image
    # cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # # Validate by plotting the bounding box on the image
    # with open(label_file, "r") as f:
    #     lines = f.readlines()
    #     if class_id == 1:
    #         image_path = mask_path.replace("_Tool_Mask", "")
    #     else:
    #         image_path = mask_path.replace("_TipPoint", "")
    #     image_path = os.path.join("data 2/ART-Net/images/train", os.path.basename(image_path))
    #     image = cv2.imread(image_path)
    #     img_h, img_w = image.shape[:2]
    #     for line in lines:
    #         class_id, x_center, y_center, width, height = map(float, line.strip().split())
    #         x = int((x_center - width / 2) * img_w)
    #         y = int((y_center - height / 2) * img_h)
    #         w = int(width * img_w)
    #         h = int(height * img_h)
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     cv2.imshow("image", image)
    #     cv2.waitKey(0)

    # cv2.destroyAllWindows()


def process_positive_images(image_dir, mask_dir, tip_mask_dir, label_dir, image_dest):
    for image_file in os.listdir(image_dir):
        if image_file.endswith(".png"):
            print(image_file)
            image_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, image_file).replace(
                ".png", "_Tool_Mask.png"
            )
            tip_mask_path = os.path.join(tip_mask_dir, image_file).replace(
                ".png", "_TipPoint.png"
            )
            label_file = os.path.join(label_dir, image_file.replace(".png", ".txt"))
            dest_image_path = os.path.join(image_dest, image_file)

            # Read the image to get its size
            image = cv2.imread(image_path)
            img_h, img_w = image.shape[:2]

            # Copy the image file to the destination
            os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
            cv2.imwrite(dest_image_path, cv2.imread(image_path))

            # Delete label file if it exists
            if os.path.exists(label_file):
                os.remove(label_file)

            # Create label for tool
            create_yolo_label(
                mask_path, label_file, class_id=1, img_w=img_w, img_h=img_h
            )

            # Create label for tool tip
            create_yolo_label(
                tip_mask_path, label_file, class_id=2, img_w=img_w, img_h=img_h
            )


def process_negative_images(image_dir, label_dir, image_dest):
    for image_file in os.listdir(image_dir):
        if image_file.endswith(".png"):
            image_path = os.path.join(image_dir, image_file)
            label_file = os.path.join(label_dir, image_file.replace(".png", ".txt"))
            dest_image_path = os.path.join(image_dest, image_file)

            # Copy the image file to the destination
            os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
            cv2.imwrite(dest_image_path, cv2.imread(image_path))

            # Create an empty label file
            open(label_file, "w").close()


# Process all datasets
process_positive_images(
    train_positive_images_path,
    train_mask_path,
    train_tip_mask_path,
    train_label_path,
    train_image_dest,
)
process_positive_images(
    test_positive_images_path,
    test_mask_path,
    test_tip_mask_path,
    test_label_path,
    test_image_dest,
)
# process_negative_images(train_negative_images_path, train_label_path, train_image_dest)
# process_negative_images(test_negative_images_path, test_label_path, test_image_dest)

print("Processing complete.")


# Iterate over images and plot bounding boxes
def view_bounding_boxes():
    for image_file in os.listdir(train_image_dest):
        if image_file.endswith(".png") and image_file.startswith("Train_Pos"):
            image_path = os.path.join(train_image_dest, image_file)
            label_file = os.path.join(
                train_label_path, image_file.replace(".png", ".txt")
            )
            image = cv2.imread(image_path)
            img_h, img_w = image.shape[:2]
            with open(label_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    class_id, x_center, y_center, width, height = map(
                        float, line.strip().split()
                    )
                    x = int((x_center - width / 2) * img_w)
                    y = int((y_center - height / 2) * img_h)
                    w = int(width * img_w)
                    h = int(height * img_h)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("image", image)
            cv2.waitKey(0)
