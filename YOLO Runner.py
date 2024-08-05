import os

models = ["n", "s", "m", "b", "l", "x"]

for m in models:
    os.mkdir(f"chkpts/6DOF/v10{m}")
    os.system(f'python "YOLO 6DOF".py > "chkpts/6DOF/v10{m}/yolov10{m}-conf.txt"')
    # os.system(f'python YOLO.py > "chkpts/ART/v10{m}/yolov10{m}-conf.txt"')
