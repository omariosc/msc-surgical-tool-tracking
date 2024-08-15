import os

models = ["n", "s", "m", "b", "l", "x"]
models = ["l", "x"]

# for m in models:
#     if not os.path.exists(f"chkpts/6DOF/v8{m}"):
#         os.mkdir(f"chkpts/6DOF/v8{m}")
#     os.system(f'python "YOLO 6DOF".py "{m}" > "chkpts/6DOF/v8{m}/yolov8{m}.txt"')

for m in models:
    if not os.path.exists(f"chkpts/ART/v8{m}"):
        os.mkdir(f"chkpts/ART/v8{m}")
    os.system(f'python YOLO.py "{m}" > "chkpts/ART/v8{m}/yolov8{m}.txt"')
