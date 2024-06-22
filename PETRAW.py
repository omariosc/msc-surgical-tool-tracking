import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gc  # Garbage collection module


# Function to create YOLO annotations from masks
def create_annotations(
    images_path_base, masks_path, labels_path, batch_size=50, n=None
):
    # For each folder inside images_path, create a corresponding folder inside labels_path
    dirs = os.listdir(images_path_base)
    # Skip specified folders
    folders_to_skip = ["008", "034", "021", "035", "037", "140", "141"]
    dirs = [d for d in dirs if d not in folders_to_skip]
    print(dirs)

    for folder in dirs:
        labels_folder = os.path.join(labels_path, folder)
        images_folder = os.path.join(images_path_base, folder)
        if not os.path.exists(labels_folder):
            os.makedirs(labels_folder)

        if n is not None:
            image_files = sorted(
                [
                    f
                    for f in os.listdir(images_folder)
                    if f.endswith(".png") and not f.startswith(".")
                ]
            )[:n]
        else:
            image_files = sorted(
                [
                    f
                    for f in os.listdir(images_folder)
                    if f.endswith(".png") and not f.startswith(".")
                ]
            )

        # Remove files that already have corresponding annotation files
        image_files = [
            f
            for f in image_files
            if not os.path.exists(
                os.path.join(labels_folder, f.replace(".png", ".txt"))
            )
        ]

        # Process images in batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i : i + batch_size]
            for image_file in batch_files:
                image_path = os.path.join(images_folder, image_file)
                mask_file = os.path.join(masks_path, folder, image_file)
                print(mask_file)
                mask_path = os.path.join(masks_path, mask_file)

                label_file = os.path.join(
                    labels_folder, image_file.replace(".png", ".txt")
                )

                if not os.path.exists(mask_path):
                    continue

                with open(label_file, "w") as lf:
                    mask = cv2.imread(mask_path)

                    # Separate the red and green components
                    mask_hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
                    lower_red = np.array([0, 50, 50])
                    upper_red = np.array([10, 255, 255])
                    red_mask = cv2.inRange(mask_hsv, lower_red, upper_red)

                    lower_green = np.array([50, 50, 50])
                    upper_green = np.array([70, 255, 255])
                    green_mask = cv2.inRange(mask_hsv, lower_green, upper_green)

                    # Process red (left tool)
                    contours_red, _ = cv2.findContours(
                        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for contour in contours_red:
                        contour = contour.squeeze()  # Remove extra dimension
                        if (
                            len(contour.shape) == 1
                        ):  # Single point, duplicate to make a box
                            contour = np.array([contour, contour])
                        normalized_contour = []
                        for point in contour:
                            x = point[0] / red_mask.shape[1]
                            y = point[1] / red_mask.shape[0]
                            normalized_contour.extend([x, y])
                        lf.write(f"0 {' '.join(map(str, normalized_contour))}\n")

                    # Process green (right tool)
                    contours_green, _ = cv2.findContours(
                        green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for contour in contours_green:
                        contour = contour.squeeze()  # Remove extra dimension
                        if (
                            len(contour.shape) == 1
                        ):  # Single point, duplicate to make a box
                            contour = np.array([contour, contour])
                        normalized_contour = []
                        for point in contour:
                            x = point[0] / green_mask.shape[1]
                            y = point[1] / green_mask.shape[0]
                            normalized_contour.extend([x, y])
                        lf.write(f"1 {' '.join(map(str, normalized_contour))}\n")

                # Use matplotlib to draw the images with bounding box
                print(image_path)
                # image = cv2.imread(image_path)
                # for contour in contours_red:
                #     x, y, w, h = cv2.boundingRect(contour)
                #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # for contour in contours_green:
                #     x, y, w, h = cv2.boundingRect(contour)
                #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # plt.axis("off")
                # plt.show()

            # Force garbage collection after each batch
            gc.collect()

        print("Annotations created in YOLO format in", labels_path)


create_annotations(
    images_path_base="/Volumes/Exodus/Data/PETRAW/Training/Training/Images/",
    masks_path="/Volumes/Exodus/Data/PETRAW/Training/Training/Segmentation/",
    labels_path="/Volumes/Exodus/Data/PETRAW/labels/train/",
    n=None,
)
