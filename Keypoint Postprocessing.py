import os
import json

# Directory containing the JSON and PNG files
input_dir = "data/6DOF/output copy/"
output_dir = "data/6DOF/Test 5 labels/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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


# Iterate over all JSON files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(input_dir, filename)
        with open(json_path, "r") as f:
            data = json.load(f)
            coords = data["tooltips"]

        # Assume image dimensions (this should be updated according to your real image size)
        img_width = 1920
        img_height = 1080

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

        # Convert to YOLO format
        left_tool_yolo = create_yolo_format(left_tool_coords, img_width, img_height)
        right_tool_yolo = create_yolo_format(right_tool_coords, img_width, img_height)

        # Create output filename and path
        base_filename = os.path.splitext(filename)[0]
        output_filename = f"test5_{base_filename}.txt"
        output_path = os.path.join(output_dir, output_filename)

        # Write to the file (append if exists)
        with open(output_path, "a") as f:
            f.write(
                f"0 {left_tool_yolo[0]} {left_tool_yolo[1]} {left_tool_yolo[2]} {left_tool_yolo[3]}\n"
            )
            f.write(
                f"1 {right_tool_yolo[0]} {right_tool_yolo[1]} {right_tool_yolo[2]} {right_tool_yolo[3]}\n"
            )

        # Delete the JSON file
        os.remove(json_path)

        # Remove the corresponding PNG file
        png_filename = f"{base_filename}_seg.png"
        png_path = os.path.join(input_dir, png_filename)
        if os.path.exists(png_path):
            os.remove(png_path)

print("Processing completed.")
