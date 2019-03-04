# import the necessary packages
from picamera import PiCamera
import picamera
from picamera.array import PiRGBArray
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

camera = PiCamera()
camera.resolution = (360, 240)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(360, 240))

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    orig = image.copy()
    (rects, weights) = hog.detectMultiScale(image, winStride=(6, 6), padding=(20, 20), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    #cv2.imshow('Before NMS', orig)
    cv2.imshow('After NMS', image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
        # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

