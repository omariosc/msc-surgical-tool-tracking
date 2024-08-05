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
| 8 | Mon 5th Aug | ~~Fully labelled video 5. Ideally, out of 24 videos (107,698 images), we will have 23 semi-labelled with 1 image (1% labelled data) every 100 frames (1,019 total images), where we can employ a cross-validation split of ~80/20 (815/204) and a 2680-image test set (1 video exclusive for testing, completely labelled so that we can evaluate the model on unseen data and be confident in the result and see difference temporally).~~ |
| 9 | Tues 6th Aug | Apply the YOLO model as the anchor-based approach and the ART-Net method as the anchor-free approach (SIMO) to the in-house dataset. Also perform some CV techniques to help extract the tool, e.g. background removal, test with different losses and base architecture models (ResNet instead of VGG). The SIMO model needs to be evaluated on the test set with key metrics extracted, and videos made for the presentation. |
| 10 | Wed 7th Aug | Ideally, development will be finished by this date, though models may still be training. |
| 11 | Thurs 8th Aug | Begin writing the report with a draft abstract. |
| 12 | Fri 9th Aug | Introduction. |
| 13 | Sat 10th Aug | Introduction. |
| 14 | Sun 11th Aug | Background Research. |
| 15 | Mon 12th Aug | Background Research. |
| 16 | Tues 13th Aug | Background Research. |
| 17 | Wed 14th Aug | Methodology. |
| 18 | Thurs 15th Aug | Methodology. |
| 19 | Fri 16th Aug | Methodology. |
| 20 | Sat 17th Aug | Results. |
| 21 | Sun 18th Aug | Results. |
| 22 | Mon 19th Aug | Results. |
| 23 | Tues 20th Aug | Discussion. |
| 24 | Wed 21st Aug | Discussion. |
| 25 | Thurs 22nd Aug | Discussion. |
| 26 | Fri 23rd Aug | Discusion. |
| 27 | Sat 24th Aug | LaTeX formatting for tables, figures and references. |
| 28 | Mon 26th Aug | LaTeX formatting for tables, figures and references. |
| 29 | Tues 27th Aug | Abstract + final changes. |
| 30 | Wed 28th Aug | Reserve day 1 for catchup + final changes. |
| 31 | Thurs 29th Aug | Reserve day 2 for catchup + final changes. |
| 32 | Fri 30th Aug | Report deadline. |
| 33 | Sat 31st Aug | Cleanup repository. |
