# TODO

| Day | Date | Tasks |
|-------|------|-------|
| 1 | Mon 29th Jul | ~~Planned a detailed timeline by tasks per day. Download ART-Net Dataset. Convert ART-Net segmentation maps to bounding boxes for the entire tool and tooltip and prepare data in YOLO format.~~ |
| 2 | Tues 30th Jul | Begin training the YOLOv10 model on detection on the ART-Net dataset. Adapt the ART-Net method as the anchor-free approach, validate reproducibility and then test on the ART-Net dataset. |
| 3 | Wed 31st Jul | Adapt the annotation software to annotate bounding box and tool tips. Label videos 3, 5 and 6 completely. |
| 4 | Thurs 1st Aug | Ideally, out of 23 videos, we have 20 semi-labelled with a sliding window of 10 images every 100 frames, where we can employ a cross-validation split of 80/20 (16 for training and 4 for validation in each fold) and 3 for testing (completely labelled so that we can evaluate the model on unseen data and be confident in the result). We can also then test on 1%, 5%, and 10% labelled data. |
| 5 | Fri 2nd Aug | Reserve day for catchup. Otherwise labelling data. |
| 6 | Sat 3rd Aug | Continue labelling data. |
| 7 | Sun 4th Aug | Apply the YOLO model as the anchor-based approach and the ART-Net method as the anchor-free approach to the in-house dataset. |
| 8 | Mon 5th Aug | Transition the detection problem to tracking. We need a way to consider IDs for each tool and deal with disappearing tools. YOLO should already deal with this, but we would need to adapt the ART-Net method here. |
| 9 | Tues 6th Aug | Ideally, development will be finished by this date. Begin writing the report. Introduction. |
| 10 | Wed 7th Aug | Introduction. |
| 11 | Thurs 8th Aug | Need to recoup if development is not finished and start writing if not already. Background Research. |
| 12 | Fri 9th Aug | Background Research. |
| 13 | Sat 10th Aug | Background Research. |
| 14 | Sun 11th Aug | Methodology. |
| 15 | Mon 12th Aug | Methodology. |
| 16 | Tues 13th Aug | Methodology. |
| 17 | Wed 14th Aug | Results. |
| 18 | Thurs 15th Aug | Results. |
| 19 | Fri 16th Aug | Results. |
| 20 | Sat 17th Aug | Discussion. |
| 21 | Sun 18th Aug | Discussion. |
| 22 | Mon 19th Aug | Discussion. |
| 23 | Tues 20th Aug | Reserve day for catchup. Otherwise, begin working on presentation. |
| 24 | Wed 21st Aug | Continue working on the presentation. |
| 25 | Thurs 22nd Aug | Finish working on the presentation. |
| 26 | Fri 23rd Aug | LaTeX formatting for tables, figures and references. |
| 27 | Sat 24th Aug | LaTeX formatting for tables, figures and references. |
| 28 | Mon 26th Aug | Abstract + final changes. |
| 29 | Tues 27th Aug | Reserve day 1 for catchup + final changes. |
| 30 | Wed 28th Aug | Reserve day 2 for catchup + final changes. |
| 31 | Thurs 29th Aug | Reserve day 3 for catchup + final changes. |
| 32 | Fri 30th Aug | Report deadline. |
