import os

models = ["n", "s", "m", "l", "x", "b"]
versions = ["8", "10"]

for m in models:
    for v in versions:
        if not os.path.exists(f"chkpts/6DOF/v{v}{m}"):
            os.mkdir(f"chkpts/6DOF/v{v}{m}")
        os.system(
            f'python "YOLO 6DOF.py" "{m}" "{v}" > "chkpts/6DOF/v{v}{m}/yolov{v}{m}-val.txt"'
        )
        if v == "8" and m == "b":
            continue
        if not os.path.exists(f"chkpts/ART/v{v}{m}"):
            os.mkdir(f"chkpts/ART/v{v}{m}")
        os.system(f'python YOLO.py "{m}" "{v}" > "chkpts/ART/v{v}{m}/yolov{v}{m}-val.txt"')
        print(f"Testing model {m} with version {v}")
