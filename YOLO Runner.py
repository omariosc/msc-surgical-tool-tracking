import os

models = ["n", "s", "m", "b", "l", "x"]

for m in models:
    if not os.path.exists(f"chkpts/6DOF/v10{m}"):
        os.mkdir(f"chkpts/6DOF/v10{m}")
    if not os.path.exists(f"chkpts/ART/v10{m}"):
        os.mkdir(f"chkpts/ART/v10{m}")
    os.system(f'python "YOLO 6DOF".py "{m}" > "chkpts/6DOF/v10{m}/yolov10{m}-conf.txt"')
    # os.system(f'python YOLO.py "{m}" > "chkpts/ART/v10{m}/yolov10{m}-val.txt"')
