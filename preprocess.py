import os

models = ["n", "s", "b", "l", "x"]

for m in models:
    os.system(f'python YOLO.py > "chkpts/ART/yolov10{m}-conf.txt"')