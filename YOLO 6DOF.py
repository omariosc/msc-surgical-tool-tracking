import os
import sys
import time
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics import YOLOv10
from multiprocessing import freeze_support
import torch

# Set seed for reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

dataset_path = "data/6DOF/"

# Create a configuration file for YOLOv10
config_content = f"""
datasets:
train: ../{dataset_path}/images/train
val: ../{dataset_path}/images/val

nc: 2  # number of classes
names: ['tool', 'tip']  # class names
"""

config_path = os.path.join("yaml/6DOF Multiclass.yaml")
with open(config_path, "w") as file:
    file.write(config_content)

config_path = "yaml/6DOF Multiclass.yaml"
# config_path_test = os.path.join(dataset_path, "data-small-test.yaml")
# config_path_final = os.path.join(dataset_path, "data-small-final.yaml")
# config_path_combined = os.path.join("yaml/data-combined.yaml")
# dataset_path = "D:/Data/ART-Net/"
# config_path = os.path.join(dataset_path, "data.yaml")
# config_path_test = os.path.join(dataset_path, "data-test.yaml")
# config_path_final = os.path.join(dataset_path, "data-final.yaml")


def train(n):
    # Load a pre-trained YOLOv8 model
    # model = YOLO("chkpts/YOLOv8/yolov8x-seg.pt")
    model = YOLOv10(f"chkpts/ART/v10{n}/yolov10{n}-detect-art/weights/best.pt")
    # model = YOLOv10(f"chkpts/ART/yolov10{n}-detect-art/weights/last.pt")
    # Put model on GPU
    model.to(device)

    # Train the model
    model.train(
        data=config_path,
        epochs=300,
        imgsz=640,
        # resume=True,
        # single_cls=True,
        save_dir=f"chkpts/6DOF/v10{n}",
        project=f"chkpts/6DOF/v10{n}",
        name=f"yolov10{n}-detect-6dof",
        save_conf=True,
        save_crop=True,
        save_txt=True,
        optimize=True,
        amp=True,
        patience=5,
        save_period=1,
    )

def test(n):
    model = YOLOv10(f"chkpts/6DOF/v10{n}/yolov10{n}-detect-6dof/weights/best.pt")
    model.to(device)
    results = model.val(data=config_path)
    print(results.results_dict)

    # model.track(
    #     # "D:\Data\RARP-45_train/train\Log_D2P280782_2017.11.20_12.20.03_4\DVC\EndoscopeImageMemory_0_sync.avi",
    #     # "D:\Data\PETRAW\Test\Video\/054.mp4",
    #     "data/6DOF/Dataset.mp4",
    #     tracker="bytetrack.yaml",
    #     save=True,
    #     show=False,
    #     # save_dir=f"chkpts/6DOF/v10{n}",
    #     # stream=True,
    # )


def track(n, show=False, save=True):
    model = YOLOv10(f"chkpts/6DOF/v10{n}/yolov10{n}-detect-6dof/weights/best.pt")
    model.to(device)
    model.track(
        # "D:\Data\RARP-45_train/train\Log_D2P280782_2017.11.20_12.20.03_4\DVC\EndoscopeImageMemory_0_sync.avi",
        "data/6DOF/Test 5.mp4",
        # "data/6DOF/Dataset.mp4",
        tracker="bytetrack.yaml",
        save=save,
        show=show,
        # save_dir=f"chkpts/ART/v10{n}",
        # stream=True,
    )


if __name__ == "__main__":
    freeze_support()

    # models = ["n", "s", "m", "b", "l", "x"]

    # # for m in models:
    m = sys.argv[1]
    # m = "x"
    # Save output to a file
    orig_stdout = sys.stdout
    f = open(f"chkpts/6DOF/v10{m}/yolov10{m}-val-out.txt", "w")
    sys.stdout = f

    # # Log time to train
    # print(f"Training model {m}")
    # start = time.time()
    # # Train the model
    # train(m)
    # end = time.time()
    # print(f"Time to train model {m}: {end - start}")

    print(f"Testing model {m}")
    start = time.time()
    # Test the model
    # test(m)
    track(m)
    end = time.time()
    print(f"Time to test model {m}: {end - start}")

    sys.stdout = orig_stdout
    f.close()
