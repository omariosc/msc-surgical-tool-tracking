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


def check_nulls(input_dir):
    # Iterate over all JSON files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            # check if "null" is in the file contents
            with open(os.path.join(input_dir, filename), "r") as f:
                contents = f.read()
                if "null" in contents:
                    print(f"Skipping {filename} due to null")
                    continue


def process(input_dir, output_label_dir, output_image_dir):
    # Iterate over all JSON files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)
            print(f"Processing {json_path}")
            with open(json_path, "r") as f:
                data = json.load(f)
                coords = data["tooltips"]

            # Delete the JSON file
            os.remove(json_path)

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
                right_tool_coords = [
                    coords[2]["x"],
                    coords[2]["y"],
                    coords[3]["x"],
                    coords[3]["y"],
                ]
            except:
                print(f"Skipping {filename} due to missing coordinates")
                continue

            # Convert to YOLO format
            left_tool_yolo = create_yolo_format(left_tool_coords, img_width, img_height)
            right_tool_yolo = create_yolo_format(
                right_tool_coords, img_width, img_height
            )

            # Create output filename and path
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"test5_{base_filename}.txt"
            output_path = os.path.join(output_image_dir, output_filename)

            # Overwrite the existing file if exists
            with open(output_path, "w") as f:
                f.write(
                    f"0 {left_tool_yolo[0]} {left_tool_yolo[1]} {left_tool_yolo[2]} {left_tool_yolo[3]}\n"
                )
                f.write(
                    f"0 {right_tool_yolo[0]} {right_tool_yolo[1]} {right_tool_yolo[2]} {right_tool_yolo[3]}\n"
                )

            # Remove the corresponding PNG file
            png_filename = f"{base_filename}_seg.png"
            png_path = os.path.join(input_dir, png_filename)
            if os.path.exists(png_path):
                os.remove(png_path)

            # Rename the base_filename.png file to include the "test5_" prefix
            png_path = os.path.join(input_dir, f"{base_filename}.png")
            new_png_path = os.path.join(output_label_dir, f"test5_{base_filename}.png")

            if os.path.exists(png_path):
                os.rename(png_path, new_png_path)


if __name__ == "__main__":
    TEST = 5
    DATA = "test" if TEST == 5 else "train"

    # Directory containing the JSON and PNG files
    input_dir = "data/6DOF/output copy/"
    output_dir = f"data/6DOF/Test {TEST}/"
    output_image_dir = f"{output_dir}/images/{DATA}/"
    output_label_dir = f"{output_dir}/labels/{DATA}/"

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
        
    check_nulls(input_dir)
    process(input_dir, output_label_dir, output_image_dir)

    print("Processing completed.")
