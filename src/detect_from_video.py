# Project: How to Detect and Classify Traffic Lights
# Author: Addison Sears-Collins
# Date created: February 1, 2021
# Description: This program uses a trained neural network to
# detect the color of a traffic light in video.

import cv2  # Computer vision library
import numpy as np
from constants import TRAINED_MODEL_FILE  # Scientific computing library

from object_detection import *  # Custom object detection program
from tensorflow import keras  # Library for neural networks


# Make sure the video file is in the same directory as your code
filename = "vegas_small.mp4"
file_size = (1920, 1080)  # Assumes 1920x1080 mp4
scale_ratio = 1  # Option to scale to fraction of original size.

# We want to save the output to a video file
output_filename = "vegas_small_annotated.mp4"
output_frames_per_second = 20.0

# Load the SSD neural network that is trained on the COCO data set
model_ssd = load_ssd_coco()

# Load the trained neural network
model_traffic_lights_nn = keras.models.load_model(TRAINED_MODEL_FILE)


def main():
    # Load a video
    cap = cv2.VideoCapture(filename)

    # Create a VideoWriter object so we can save the video output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    result = cv2.VideoWriter(
        output_filename, fourcc, output_frames_per_second, file_size
    )

    # Process the video
    while cap.isOpened():
        # Capture one frame at a time
        success, frame = cap.read()

        # Do we have a video frame? If true, proceed.
        if success:
            # Resize the frame
            width = int(frame.shape[1] * scale_ratio)
            height = int(frame.shape[0] * scale_ratio)
            frame = cv2.resize(frame, (width, height))

            # Store the original frame
            original_frame = frame.copy()

            output_frame = perform_object_detection_video(
                model_ssd, frame, model_traffic_lights=model_traffic_lights_nn
            )

            # Write the frame to the output video file
            result.write(output_frame)

        # No more video frames left
        else:
            break

    # Stop when the video is finished
    cap.release()

    # Release the video recording
    result.release()

    # Close all windows
    cv2.destroyAllWindows()


main()
