import cv2
import os
import time
import imutils
from imutils.video import VideoStream
import argparse


print("[Instruction] Press K to take a Screenshot, Press Q to Quit")
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,
                help="screenshots/")
dir_path = os.path.dirname(os.path.realpath(__file__))
args = vars(ap.parse_args())
detector = cv2.CascadeClassifier(dir_path+"\haarcascade_frontalface_default.xml")
print("[Info] Starting Display")
dir_path = dir_path+"\output"
if args["output"] != "":
    dir_path += "\_" + args["output"]
else:
    dir_path += "default"
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0
while True:
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=800)

    rects = detector.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("k"):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        p = os.path.sep.join([dir_path, "{}.png".format(
            str(total).zfill(5))])
        cv2.imwrite(p, orig)
        total += 1
    elif key == ord("q"):
        break

print("[INFO] {} images captured".format(total))
cv2.destroyAllWindows()
vs.stop()
