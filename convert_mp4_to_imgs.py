# Write a script which takes in a mp4 file and converts it to a series of images, labelling each one with {frame_number}.png

import cv2
import os


def convert_to_imgs(path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(path)
    # Get the number of frames
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # For each frame
    for i in range(n):
        # Read the frame
        ret, frame = cap.read()
        # Save the frame as a png image
        cv2.imwrite(f"data/6DOF/Test 1/{i}.png", frame)
        print(f"Saved frame {i}")
    # Release the VideoCapture object
    cap.release()


# Path to the video file
path = "data/6DOF/Test 1/Task1_stitched_video.mp4"
convert_to_imgs(path)
