import cv2, sys
import numpy as np
import glob
import pdb

# path to where you want to save extracted frames
SAVE_PATH = ""

# Path to your video file
filename = ""
 

def video_to_frames(video_filename):
    """
    Convert video to video

    Args:
        video_filename: (str): write your description
    """
    source_video = cv2.VideoCapture(video_filename)
    n_frames = source_video.get(cv2.CAP_PROP_FRAME_COUNT)
    i = 0
    while i <= n_frames:
        ret, frame = source_video.read()
        if ret:
            cv2.imwrite(SAVE_PATH + "frame_" + str(i) + ".jpg", frame)
        else:
            i += 1
            continue

        i += 1

video_to_frames(filename)
