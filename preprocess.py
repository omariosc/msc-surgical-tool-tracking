# Rename every image inside each folder D:\Data\PETRAW\Training\Training\Images to include its folder name (e.g. 001) to the front (as a string)
import os

# Path to the folder containing the images
path = "D:/Data/PETRAW/Test/Images/"
# path = "D:/Data/PETRAW/labels/test/"
labels = "D:/Data/PETRAW/images/test"
# labels = "D:/Data/PETRAW/labels/test"

# List all the directories in the folder
dirs = os.listdir(path)
# filter out .txt
dirs = [dir for dir in dirs if not dir.endswith(".txt")]
print(dirs)
# quit()

# For each directory
for dir in dirs:
    # List all the files in the directory
    files = os.listdir(path + dir)
    # For each file
    for file in files:
        if file.startswith("frame"):
            # make a copy of the file with the directory name in front in labels folder
            os.rename(path + dir + "/" + file, labels + "/" + dir + "_" + file)
            
    # delete the directory forcefully
    try:
        os.rmdir(path + dir)
    except OSError as e:
        # delete all files beginning with .
        files = os.listdir(path + dir)
        for file in files:
            if file.startswith("."):
                os.remove(path + dir + "/" + file)
        # delete the directory forcefully
        os.rmdir(path + dir)
    print(f"Deleted {dir}")
