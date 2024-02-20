import cv2
import numpy as np


class FaceDetector:
    def __init__(self, debug: bool = False):
        self.__face_cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_default.xml")
        self.shape = None
        self.prev_face = None
        self.debug = debug
        self.prev_borders = []

    def find_face(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.__face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            if len(self.prev_borders) == 0:
                raise Exception("No faces detected")
            rect_up_start, rect_up_end, rect_down_start, rect_down_end = self.prev_borders
            up_image = image[rect_up_start[1]: rect_up_end[1], rect_up_start[0]: rect_up_end[0]]
            down_image = image[rect_down_start[1]: rect_down_end[1], rect_down_start[0]: rect_down_end[0]]
            new_image = np.concatenate((up_image, down_image), axis=0)
            if self.shape is not None:
                new_image = cv2.resize(new_image, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
            return self.prev_borders, new_image

        if self.prev_face is None:
            (x, y, w, h) = faces[0]
            self.prev_face = faces[0]
        else:
            (x1, y1, w, h) = self.prev_face
            dist = [np.sqrt(pow(x-x1, 2) + pow(y-y1, 2)) for (x, y, w, h) in faces]
            ind = np.argmin(dist)
            (x, y, w, h) = faces[ind]
            self.prev_face = faces[ind]

        x = int(x+0.2*w)
        y = int(y + 0.05 * h)
        w = int(0.6 * w)
        h = int(0.9 * h)

        rect_up_start, rect_up_end = (x, int(y-0.1*h)), (x + w, int(y + 0.2 * h))
        rect_down_start, rect_down_end = (x, int(y + 0.55 * h)), (x + w, int(y + h))
        self.prev_borders = (rect_up_start, rect_up_end, rect_down_start, rect_down_end)
        up_image = image[rect_up_start[1]: rect_up_end[1], rect_up_start[0]: rect_up_end[0]]
        down_image = image[rect_down_start[1]: rect_down_end[1], rect_down_start[0]: rect_down_end[0]]

        # if self.debug:
        #     cv2.imshow('image', image)
        #     image = cv2.rectangle(image, rect_up_start, rect_up_end, (0, 255, 0), 1)
        #     image = cv2.rectangle(image, rect_down_start, rect_down_end, (0, 255, 0), 1)
        #     cv2.imshow('full_image', image)
        #     if cv2.waitKey(1) & 0xFF == 27:
        #         exit(1)

        new_image = np.concatenate((up_image, down_image), axis=0)
        if self.shape is not None:
            new_image = cv2.resize(new_image, (self.shape[1], self.shape[0]), interpolation = cv2.INTER_AREA)
        self.shape = new_image.shape
        borders = (rect_up_start, rect_up_end, rect_down_start, rect_down_end)

        return borders, new_image

