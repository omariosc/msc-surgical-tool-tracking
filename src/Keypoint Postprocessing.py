# python3 setup.py install --user
# python -m wat.run --data-dir ../../data/6DOF --port 1234 --maxtips 4

import os
import json


# Function to create a bounding box in YOLO format
def create_yolo_format(coords, img_width, img_height):
    x_min = min(coords[0], coords[2])
    y_min = min(coords[1], coords[3])
    x_max = max(coords[0], coords[2])
    y_max = max(coords[1], coords[3])
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height


def check_nulls(input_dir, original_dir):
    # Iterate over all JSON files in the directory
    count = 0
    files = os.listdir(input_dir)
    # sort files by creation time
    files.sort(key=lambda x: os.path.getctime(os.path.join(input_dir, x)))
    for filename in files:
        if filename.endswith(".json"):
            # check if "null" is in the file contents
            with open(os.path.join(input_dir, filename), "r") as f:
                contents = f.read()
                if "null" in contents or "None" in contents or "[]" in contents:
                    print(f"Skipping {filename} due to null")
                    count += 1
                    # print the time created exactly
                    continue
                    # Delete the seg png and json files
                    base_filename = os.path.splitext(filename)[0]
                    png_path = os.path.join(input_dir, f"{base_filename}_seg.png")
                    json_path = os.path.join(input_dir, f"{base_filename}.json")
                    if os.path.exists(png_path):
                        os.remove(png_path)
                    if os.path.exists(json_path):
                        os.remove(json_path)
                    # Move original image to the original folder
                    os.rename(os.path.join(input_dir, f"{base_filename}.png"), os.path.join(original_dir, f"{base_filename}.png"))
    print(f"Skipped {count} files due to null")


def process(input_dir, output_label_dir, output_image_dir):
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

            # Delete the JSON file
            # os.remove(json_path)

            # Assume image dimensions (this should be updated according to your real image size)
            img_width = 1920
            img_height = 1080
            try:
                # Extract bounding boxes for the tools
                left_tool_coords = [
                    coords[0]["x"],
                    coords[0]["y"],
                    coords[1]["x"],
                    coords[1]["y"],
                ]
                # Make sure none of them are [], null or None
                if all(left_tool_coords):
                    left_exists = True
                    # Convert to YOLO format
                    left_tool_yolo = create_yolo_format(left_tool_coords, img_width, img_height)
            except:
                print(f"{filename} has missing left tool coordinates")
            try:
                right_tool_coords = [
                    coords[2]["x"],
                    coords[2]["y"],
                    coords[3]["x"],
                    coords[3]["y"],
                ]
                # Make sure none of them are [], null or None
                if all(right_tool_coords):
                    right_exists = True
                    right_tool_yolo = create_yolo_format(
                        right_tool_coords, img_width, img_height
                    )
            except:
                print(f"{filename} has missing right tool coordinates")

            # Create output filename and path
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"{base_filename}.txt"
            output_path = os.path.join(output_label_dir, output_filename)

            # Overwrite the existing file if exists
            with open(output_path, "w") as f:
                if left_exists:
                    f.write(
                        f"0 {left_tool_yolo[0]} {left_tool_yolo[1]} {left_tool_yolo[2]} {left_tool_yolo[3]}\n"
                    )
                    left_exists = False
                if right_exists:
                    f.write(
                        f"0 {right_tool_yolo[0]} {right_tool_yolo[1]} {right_tool_yolo[2]} {right_tool_yolo[3]}\n"
                    )
                    right_exists = False

            # Remove the corresponding PNG file
            png_filename = f"{base_filename}_seg.png"
            png_path = os.path.join(input_dir, png_filename)
            if os.path.exists(png_path):
                os.remove(png_path)

            # Rename the base_filename.png file
            png_path = os.path.join(input_dir, f"{base_filename}.png")
            new_png_path = os.path.join(output_image_dir, f"{base_filename}.png")

            if os.path.exists(png_path):
                os.rename(png_path, new_png_path)


if __name__ == "__main__":
    TEST = 5
    DATA = "test" if TEST == 5 else "train"

    # Directory containing the JSON and PNG files
    original_dir = "data/6DOF/input/"

    input_dir = "data/6DOF/output/"
    output_dir = "data/6DOF/"
    output_image_dir = f"{output_dir}/images/train/"
    output_label_dir = f"{output_dir}/labels/train/"

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    check_nulls(input_dir, original_dir)
    process(input_dir, output_label_dir, output_image_dir)

    print("Processing completed.")
