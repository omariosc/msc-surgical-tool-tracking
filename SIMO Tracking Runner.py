import os

data = ["ART", "6DOF"]

for d in data:
    # arch = ["vgg", "resnet18", "resnet50"]
    arch = "vgg"
    os.makedirs(f"chkpts/{d}/{arch}", exist_ok=True)
    os.system(f'python "SIMO Tracking".py {d} {arch}> "chkpts/SIMO/{d}/{arch}.txt"')
