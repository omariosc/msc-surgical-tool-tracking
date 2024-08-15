
import os

data = ["ART", "6DOF"]
for d in data:
    arch = ["resnet50", "vgg", "fcn-resnet", "fcn-vgg"]
    for a in arch:
        if (d == "6DOF" and (a == "fcn-resnet" or a == "vgg")) or (d == "ART" and (a != "resnet50")):
            continue
        print(d, a)
        os.makedirs(f"chkpts/SIMO/{d}/{a}", exist_ok=True)
        os.system(
            f'python "SIMO Tracking".py {d} {a}> "chkpts/SIMO/{d}/{a}/{a}.txt"'
        )
