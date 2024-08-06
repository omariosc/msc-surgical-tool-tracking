import os


input_dir = "data/6DOF/input"
output_dir = "data/6DOF/output"

# If a file exists in the input directory, which exists in output directory (by name) remove it from input
for file in os.listdir(input_dir):
    if os.path.isfile(os.path.join(output_dir, file)):
        os.remove(os.path.join(input_dir, file))