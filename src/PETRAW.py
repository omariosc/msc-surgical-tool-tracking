import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import gc  # Garbage collection module


# Open folder /Volumes/Exodus/Data/PETRAW/Training/Training/Video and convert all mp4 into images
# with same resolution and save them in /Volumes/Exodus/Data/PETRAW/Training/Training/Images/{vid_name}
def split_video(video_path, vid_name, output_folder):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        # should be 'frame00000000.png'
        cv2.imwrite(
            f"{output_folder}/{vid_name[5:]}frame{str(count).zfill(8)}.png", image
        )  # save frame as PNG file
        success, image = vidcap.read()
        count += 1
    print(f"Extracted {count} frames from {video_path}")


def extract_frames():
    DATA_PATH = (
        "/Volumes/Exodus/Data/PETRAW/"
        if os.path.exists("/Volumes/Exodus")
        else "D:/Data/PETRAW/"
    )

    video_folder = os.path.join(DATA_PATH, "Test/Video")
    output_folder = os.path.join(DATA_PATH, "Test/Images")

    if not os.path.exists(output_folder):

        os.makedirs(output_folder)

    videos = os.listdir(video_folder)
    # remove anything beginning with a dot
    videos = [v for v in videos if not v.startswith(".")]
    # remove video f the name before .mp4 exists in outputfodler already
    videos = [v for v in videos if not os.path.exists(os.path.join(output_folder, os.path.splitext(v)[0]))]
    print(videos)
    videos.append("113.mp4") # temp

    for file in videos:

        if file.endswith(".mp4") and not file.startswith("."):

            vid_name = os.path.splitext(file)[0]

            vid_path = os.path.join(video_folder, file)

            output_path = os.path.join(output_folder, vid_name)

            if not os.path.exists(output_path):

                os.makedirs(output_path)

            split_video(vid_path, vid_name, output_path)

    # Rename every single image in the subfolders here to remove the first _ character
    for root, _, files in os.walk(output_folder):
        for file in files:
            if file.startswith("_"):
                os.rename(os.path.join(root, file), os.path.join(root, file[1:]))


# Function to create YOLO annotations from masks
def create_annotations(
    images_path_base, masks_path, labels_path, batch_size=50, n=None
):
    # For each folder inside images_path, create a corresponding folder inside labels_path
    dirs = os.listdir(images_path_base)
    # # Skip specified folders
    # folders_to_keep = ["113"]
    # # keep all folders, except those in images_path_base, but keep those in folders_to_keep
    # dirs = [f for f in dirs if f not in images_path_base and f not in folders_to_keep]
    # print(dirs)
    # quit()
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
                # print(mask_file)
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
                # print(image_path)
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
                print(f"Processed {image_file}")

            # Force garbage collection after each batch
            gc.collect()
            gc.collect()
            print(f"Processed {len(batch_files)} images in {folder}")

        print("Annotations created in YOLO format in", labels_path)


DATA_PATH = (
    "/Volumes/Exodus/Data/PETRAW/"
    if os.path.exists("/Volumes/Exodus")
    else "D:/Data/PETRAW/"
)

# extract_frames()

create_annotations(
    images_path_base=DATA_PATH + "Training/Training/Images/",
    masks_path=DATA_PATH + "Training/Training/Segmentation/",
    labels_path=DATA_PATH + "labels/train/",
    n=None,
)
