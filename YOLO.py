import os
import sys
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

dataset_path = "data/ART-Net/"

# Create a configuration file for YOLOv8
# config_content = f"""
# datasets:
# train: ../{dataset_path}/images/train
# val: ../{dataset_path}/images/val

# nc: 2  # number of classes
# names: ['tool', 'tip']  # class names
# """

# config_path = os.path.join("yaml/ART-Net Multiclass.yaml")
# with open(config_path, "w") as file:
#     file.write(config_content)

config_path = "yaml/ART-Net Multiclass.yaml"
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
    model = YOLOv10(f"chkpts/YOLOv10/yolov10{n}.pt")
    # model = YOLOv10(f"chkpts/ART/yolov10{n}-detect-art/weights/last.pt")
    # Put model on GPU
    model.to(device)

    # Train the model
    model.train(
        data=config_path,
        epochs=300,
        imgsz=640,
        # resume=True,
        single_cls=False,
        save_dir="chkpts/ART",
        project="chkpts/ART",
        name=f"yolov10{n}-detect-art",
        save_conf=True,
        save_crop=True,
        save_txt=True,
        save_json=True,
        optimize=True,
        amp=True,
        patience=10,
        save_period=1,
    )

    # Save the model checkpoint to a file
    # checkpoint = model.save("chkpts/ART/yolov10{n}-detect-art/1.pt")

    # Train the model for more epochs
    # model = YOLO("chkpts/ART/yolov8x-semiseg-artnet.pt")

    # model.train(
    #     data=config_path_combined,
    #     epochs=50,
    #     imgsz=640,
    #     # resume=True,
    #     single_cls=True,
    #     save_dir="chkpts/combined",
    #     project="chkpts/combined",
    #     name="yolov8x-semiseg-combined",
    #     save_conf=True,
    #     save_crop=True,
    #     optimize=True,
    #     amp=True,
    #     patience=10,
    #     save_period=1,
    # )


def test(n):
    model = YOLOv10(f"chkpts/ART/yolov10{n}-detect-art/weights/best.pt")
    model.to(device)
    results = model.val(data=config_path)
    print(results.results_dict)

    model.track(
        # "D:\Data\RARP-45_train/train\Log_D2P280782_2017.11.20_12.20.03_4\DVC\EndoscopeImageMemory_0_sync.avi",
        # "D:\Data\PETRAW\Test\Video\/054.mp4",
        "data/6DOF/Dataset.mp4",
        tracker="bytetrack.yaml",
        save=True,
        show=False,
    )


if __name__ == "__main__":
    freeze_support()

    models = ["n", "s", "b", "l", "x"]
    
    for m in models:
        train(m)
        test(m)
        
    #     # Save output to a file
    #     orig_stdout = sys.stdout
    #     f = open(f"chkpts/ART/yolov10{m}-output.txt", "w")
    #     sys.stdout = f

    #     # Train the model
    #     train(m)
        
    #     # Test the model
    #     test("m")

    #     sys.stdout = orig_stdout
    #     f.close()
