import os

models = ["n", "s", "m", "b", "l", "x"]

for m in models:
    os.system(f'python YOLO.py > "chkpts/ART/v10{m}/yolov10{m}-conf.txt"')
