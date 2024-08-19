# python3 setup.py install --user
# python -m wat.run --data-dir ../../data/6DOF --port 1234 --maxtips 4

import os
import json

from cv2 import sort

# Preset width and height for the bounding boxes
PRESET_WIDTH = 0.046875
PRESET_HEIGHT = 0.083


# Function to convert coordinates to YOLO format
def create_yolo_format(coords, img_width, img_height):
    x_center = coords["x"] / img_width
    y_center = coords["y"] / img_height
    return x_center, y_center

def sort_files(input_dir):
    count = 0
    files = os.listdir(input_dir)
    files.sort(key=lambda x: os.path.getctime(os.path.join(input_dir, x)))
    for filename in files:
        if filename.endswith(".json"):
            print(filename)
    return None

def process(input_dir, output_label_dir):
    # Iterate over all JSON files in the directory
    for filename in os.listdir(input_dir):
        left_exists = False
        right_exists = False
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)
            print(f"Processing {json_path}")
            with open(json_path, "r") as f:
                data = json.load(f)
                coords = data["tooltips"]

            # Assume image dimensions (update this according to your real image size)
            img_width = 1920
            img_height = 1080

            # Convert left and right tooltip coordinates to YOLO format
            try:
                left_tool_yolo = create_yolo_format(coords[0], img_width, img_height)
                if all(left_tool_yolo):
                    left_exists = True
            except:
                print(f"{filename} has missing left tool coordinates")

            try:
                right_tool_yolo = create_yolo_format(coords[1], img_width, img_height)
                if all(right_tool_yolo):
                    right_exists = True
            except:
                print(f"{filename} has missing left tool coordinates")

            # Create output filename and path
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"{base_filename}.txt"
            output_path = os.path.join(output_label_dir, output_filename)

            # Append to the existing file
            with open(output_path, "a") as f:
                if left_exists:
                    f.write(
                        f"1 {left_tool_yolo[0]} {left_tool_yolo[1]} {PRESET_WIDTH} {PRESET_HEIGHT}\n"
                    )
                    left_exists = False
                if right_exists:
                    f.write(
                        f"1 {right_tool_yolo[0]} {right_tool_yolo[1]} {PRESET_WIDTH} {PRESET_HEIGHT}\n"
                    )
                    right_exists = False

            # Delete the JSON file
            # os.remove(json_path)

            # Remove the corresponding PNG file
            png_filename = f"{base_filename}_seg.png"
            png_path = os.path.join(input_dir, png_filename)
            if os.path.exists(png_path):
                os.remove(png_path)


if __name__ == "__main__":
    # Directory containing the JSON and PNG files
    input_dir = "data/6DOF/output/"
    output_label_dir = "data/6DOF/labels/val/"

    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    # sort_files(input_dir)
    process(input_dir, output_label_dir)

    print("Processing completed.")
