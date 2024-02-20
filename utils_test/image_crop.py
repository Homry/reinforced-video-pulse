import cv2
import numpy as np

image =  cv2.imread('../img.png')
image = image[100:1000, 500:1500]
cv2.imshow('aaa', image)
cv2.waitKey(0)