import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from multiprocessing import freeze_support
import torch

# Set seed for reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# dataset_path = "D:/Data/ART-Net/"
dataset_path = "D:/Data/PETRAW/"
config_path = os.path.join(dataset_path, "data-small.yaml")
config_path_test = os.path.join(dataset_path, "data-small-test.yaml")
config_path_final = os.path.join(dataset_path, "data-small-final.yaml")
config_path_combined = os.path.join("yaml/data-combined.yaml")
# dataset_path = "D:/Data/ART-Net/"
# config_path = os.path.join(dataset_path, "data.yaml")
# config_path_test = os.path.join(dataset_path, "data-test.yaml")
# config_path_final = os.path.join(dataset_path, "data-final.yaml")

def train():
    # Load a pre-trained YOLOv8 model
    # model = YOLO("chkpts/YOLOv8/yolov8x-seg.pt")
    model = YOLO("chkpts/ART/yolov8x-semiseg-artnet2/weights/best.pt")

    # Put model on GPU
    model.to(device)

    # # # Train the model with only 10 images instead of the full dataset
    # # model.train(data=config_path, epochs=1, imgsz=640)

    # # # Save the model checkpoint to a file
    # # checkpoint = model.save("chkpts/ART/yolov8x-semiseg-artnet.pt")

    # # Train the model for more epochs
    # model = YOLO("chkpts/ART/yolov8x-semiseg-artnet.pt")

    model.train(
        data=config_path_combined,
        epochs=50,
        imgsz=640,
        # resume=True,
        single_cls=True,
        save_dir="chkpts/combined",
        project="chkpts/combined",
        name="yolov8x-semiseg-combined",
        save_conf=True,
        save_crop=True,
        optimize=True,
        amp=True,
        patience=10,
        save_period=1,
    )

def test():
    model = YOLO("chkpts/combined/yolov8x-semiseg-combined6/weights/best.pt")
    model.to(device)
    # results = model.val(data=config_path_test)
    # print(results.results_dict)

    model.track(
        # "D:\Data\RARP-45_train/train\Log_D2P280782_2017.11.20_12.20.03_4\DVC\EndoscopeImageMemory_0_sync.avi",
        "D:\Data\PETRAW\Test\Video\/054.mp4",
        # "data/6DOF/Test 1/Task1_stitched_video.mp4",
        tracker="bytetrack.yaml",
        save=True,
        show=True,
    )

if __name__ == "__main__":
    freeze_support()

    # Train the model
    # train()
    
    # Test the model
    test()
