import cv2
import numpy as np

# 1. Load the Network
net = cv2.dnn.readNetFromDarknet('cfg/yolov3.cfg', 'weights/yolov3.weights')

with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print(classes)

img = cv2.imread('images/cat.jpg')
height, width, _ = img.shape
