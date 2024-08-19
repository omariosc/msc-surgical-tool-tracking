# TODO

| Day | Date | Tasks |
|-------|------|-------|
| 1 | Mon 29th Jul | ~~Planned a detailed timeline by tasks per day. Download ART-Net Dataset. Convert ART-Net segmentation maps to bounding boxes for the entire tool and tooltip and prepare data in YOLO format. Begin training the YOLOv10 model on detection on the ART-Net dataset.l~~ |
| 2 | Tues 30th Jul | ~~Train a more bulky model on the ART-Net dataset. Adapt the ART-Net method as the anchor-free approach, validate reproducibility and then test on the ART-Net dataset.~~ |
| 3 | Wed 31st Jul | ~~Adapt the annotation software to annotate bounding box and tool tips.~~ |
| 4 | Thurs 1st Aug | ~~Transition the detection problem to tracking. We need a way to consider IDs for each tool and deal with disappearing tools. YOLO should already deal with this, but we would need to adapt the ART-Net method here.~~ |
| 5 | Fri 2nd Aug | ~~Partially labelled video 5.~~ |
| 6 | Sat 3rd Aug | ~~Partially labelled video 5.~~ |
| 7 | Sun 4th Aug | ~~Partially labelled video 5.~~ |
| 8 | Mon 5th Aug | ~~Fully labelled video 5. Ideally, out of 24 videos (107,698 images), we will have 23 semi-labelled with 1 image (1% labelled data) every 100 frames (1,048 total images), where we can employ a cross-validation split of ~80/20 (839/209) and a 2680-image test set (1 video exclusive for testing, completely labelled so that we can evaluate the model on unseen data and be confident in the result and see difference temporally).~~ |
| 9 | Tues 6th Aug | ~~Finish labelling the training set. Apply the YOLO model as the anchor-based approach to the in-house dataset. Also perform some CV techniques to help extract the tool, e.g. background removal, test with different losses and base architecture models (ResNet instead of VGG).~~ |
| 10 | Wed 7th Aug | ~~Ideally, development will be finished by this date, though models may still be training. Labelled tool tips in training data~~. |
| 11 | Thurs 8th Aug | ~~Finished labelling tool tips in test data.~~ |
| 12 | Fri 9th Aug | ~~Train SIMO on ART-Net dataset and implement tracking with YOLO.~~ |
| 13 | Sat 10th Aug | ~~Run YOLO on test set and attempt to fix tracking issues. Begin writing the report with a draft abstract.~~ |
| 14 | Sun 11th Aug | ~~The SIMO (anchor-free model) needs to be trained and evaluated on the ART-Net test sets with key metrics extracted.~~ |
| 15 | Mon 12th Aug | ~~The SIMO (anchor-free model) needs to be trained and evaluated on the 6DOF test sets with key metrics extracted. Add outline document notes.~~ |
| 16 | Tues 13th Aug  | ~~Fix YOLO tracking issue. Train new SIMO model on ART-Net, 6DOF and obtained tracking videos.~~ |
| 17 | Wed 14th Aug   | ~~Generated all annotations using Yolov10X. Create table of results. YOLOv8 models on 6DOF.~~ |
| 18 | Thurs 15th Aug | ~~Processed annotations for RetinaNet anchor-box optimisation. YOLOv8 models on ART-Net. Run RetinaNet (with and without anchor-box optimisation) on 6DOF and ART-Net.~~ |
| 19 | Fri 16th Aug   | ~~Prepared models and data for running future experiments.~~ |
| 20 | Sat 17th Aug   | ~~Run remaining RetinaNet models. Run EfficientDet models. Introduction.~~ |
| 21 | Sun 18th Aug   | ~~Run remaining EfficientDet models and produce videos. Rerun validation on YOLOv8 and YOLOv10 on 6DOF and ART-Net at 0.5 IoU and get non-tracked videos. Improve tracking and get new tracking videos. Scheduled running SIMO models. Introduction.~~ |
| 22 | Mon 19th Aug   | ~~Run DETR models. Introduction. Computer vision and AI methods. PRISMA diagram.~~ |
| 23 | Tues 20th Aug  | Background Research. |
| 24 | Wed 21st Aug   | Background Research. |
| 25 | Thurs 22nd Aug | Background Research. |
| 26 | Fri 23rd Aug   | Methodology. |
| 27 | Sat 24th Aug   | Methodology. |
| 28 | Mon 26th Aug   | Make abstract teaser image. Results. |
| 29 | Tues 27th Aug  | Make combined tracking video. Results. |
| 30 | Wed 28th Aug   | Discusion. |
| 31 | Thurs 29th Aug | Discussion. |
| 32 | Fri 30th Aug | Report deadline. |
| 33 | Sat 31st Aug | Cleanup repository. |
| 34 | Sun 1st Sep  | Presentation. |
