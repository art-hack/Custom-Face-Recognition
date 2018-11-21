import cv2
import os
import time
import imutils
from imutils.video import VideoStream
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="images/*")
ap.add_argument("-o", "--output", required=True,
                help="screenshots/")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
