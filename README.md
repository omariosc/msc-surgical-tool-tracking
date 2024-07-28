# Surgical Skill Improvements through AI-driven Training Enhancements

This is a private repository for my PhD project: "Surgical Skill Improvements through AI-driven Training Enhancements".

## Repository Code

- ART-Net
  - Paper focus.
- ConvLSTM
  - Potential paper focus. Object detection and tracking.
- daVinci tracking
  - YOLOv5.
- Dresden
  - Comparing annotations.
- EndoVis 2017
  - Segmentation code.
- Keypoint Annotation
  - Will be used for annotation.
- Particle Filter
  - For robotic tracking so may not be useful.
- Pose Estimation
  - May be useful for pose estimation visualisation in 3D.
- Segmentation
  - Very useful code for segmentation on EndoVis 2017 dataset.
- surgtoolloc
  - A useful MONAI file
- t-test
  - Can perform paired t-test to establish deep learning model's statistical significance.
- Tool Pose Annotation
  - Can create poses. Did not work on OSX but can try on Windows.

## Datasets

Ideas for datasets include:

- ART-Net (tooltip, masks, bounding box, detection)
  - `/Volumes/Exodus/Data/ART-Net/images` (`train` and `val` folders)
  - `/Volumes/Exodus/Data/ART-Net/labels` (`train` and `val` folders for pixel-wise masks)
  - `/Volumes/Exodus/Data/ART-Net/masks` (`train` and `val` folders for .png masks)
  - `/Volumes/Exodus/Data/ART-Net/tool_tip_masks` (`train` and `val` folders for tool tip pixel-wise masks)
- EndoVis 2015 (tooltip, masks, bounding box)
  - `/Volumes/Exodus/Data/EndoVis2015/Tracking_Rigid_Training (Raw Images)/Training` (folder for each video, `Masks` and `Raw` folders. `Instruments_OPN.csv` for [center points and axes](#other-datasets))
- PETRAW (sensor, masks)
  - `/Volumes/Exodus/Data/PETRAW/Training/Training/Images` (folder for each video)
  - `/Volumes/Exodus/Data/PETRAW/Training/Training/Segmentation` (folder for each video)
  - `/Volumes/Exodus/Data/PETRAW/Training/Training/Kinematic` (1 file for each video)
- 6DOF (sensor) _(In-house dataset)_

### Other Datasets

- EndoVis 2015 (tooltip, pose)
  - `/Volumes/Exodus/Data/EndoVis2015/Testing (Raw Images)` (folder for each video - **nothing to validate**)
  - `/Volumes/Exodus/Data/EndoVis2015/Tracking_Robotic_Training (Pose)/Training` (folder for each video - contains clasper angle, axis for tip and shaft, and center point for each instrument)
- SurgRIPE? (pose, masks)
- m2cai16-tool-locations (bounding boxes)
- EndoVis 2018? (masks)
- ATLAS Dione (bounding boxes)

### Notes

**Excepts for EndoVis2015:**

- `<ins1_center_x>,<ins1_center_y>`: Pixel coordinates of the center point for instrument1, the center point is defined as the intersection between the instrument axis and the border between shaft and manipulator.
- `<ins1_axis_x>,<ins1_axis_y>`: Normalized axis vector of instrument1.
- Manipulator not visible: If the manipulator is not visible in an image, we used the intersection of the shaft axis and the top of the visible section of the shaft as center point.
- Shaft not visible: If the shaft is not visible, neither center point nor axis are computed for the instrument.
