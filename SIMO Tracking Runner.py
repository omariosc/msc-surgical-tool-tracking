
import os

data = ["ART", "6DOF"]
data = ["6DOF"]
for d in data:
    # arch = ["vgg", "resnet18", "resnet50"]
    arch = "fcn-vgg"
    os.makedirs(f"chkpts/SIMO/{d}/{arch}", exist_ok=True)
    os.system(
        f'python "SIMO Tracking".py {d} {arch}> "chkpts/SIMO/{d}/{arch}/{arch}.txt"'
    )
