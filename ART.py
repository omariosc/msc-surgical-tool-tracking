import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from multiprocessing import freeze_support
import torch

# Set seed for reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    freeze_support()
    
    print(torch.cuda.is_available())
    print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset_path = "D:/Data/ART-Net/"

    # Create a configuration file for YOLOv8
    config_content = f"""
    train: {dataset_path}/images/train
    val: {dataset_path}/images/val

    nc: 1  # number of classes
    names: ['tool']  # class names
    """

    # config_content = f"""
    # train: {dataset_path}/images/train
    # val: {dataset_path}/images/val

    # nc: 1  # number of classes
    # names: ['tool']  # class names
    # """

    config_path = os.path.join(dataset_path, "data.yaml")
    with open(config_path, "w") as file:
        file.write(config_content)

    # Load a pre-trained YOLOv8 model
    model = YOLO("chkpts/YOLOv8/yolov8x-seg.pt")
    
    # Put model on GPU
    model.to(device)

    # # Train the model with only 10 images instead of the full dataset
    # model.train(data=config_path, epochs=1, imgsz=640)

    # # Save the model checkpoint to a file
    # checkpoint = model.save("chkpts/ART/yolov8x-semiseg-artnet.pt")

    # # Train the model for 50 more epochs but save new checkpoint to a file (and the metrics to a new file)
    # model = YOLO("chkpts/ART/yolov8x-semiseg-artnet.pt")

    model.train(
        data=config_path,
        epochs=50,
        imgsz=640,
        # resume=checkpoint,
        single_cls=True,
        save_dir="chkpts/ART",
        project="chkpts/ART",
        name="yolov8x-semiseg-artnet",
        save_conf=True,
        save_crop=True,
        optimize=True,
        save_period=1
    )

    # Evaluate the model on the validation set
    model.val()

    losses = []
    accuracies = []

    # Losses and accurances are stored in the results.csv file in the save_dir
    with open("chkpts/ART/yolov8x-semiseg-artnet/results.csv") as file:
        lines = file.readlines()
        losses = [float(line.split(",")[1]) for line in lines[1:]]
        accuracies = [float(line.split(",")[7]) for line in lines[1:]]    

    # Plot the training loss and validation mAP
    plt.figure(figsize=(10, 5))
    # Epoch starts at 1 and are all integers
    epochs = range(1, len(losses) + 1)
    epochs = [int(epoch) for epoch in epochs]
    plt.subplot(1, 2, 1)
    # do not plot floats in axis
    plt.xticks(epochs)
    plt.plot(epochs, losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.xticks(epochs)
    plt.plot(epochs, accuracies)
    plt.title("Validation mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.tight_layout()
    plt.savefig("results/ART/metrics.png")
    plt.show()
