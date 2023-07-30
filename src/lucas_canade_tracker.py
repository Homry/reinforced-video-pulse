import cv2
import numpy as np

class LucasKanadeTracker:
    def __init__(self):
        self.__old_gray_frame = None
        self.__points = None
        self.__lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.__old_gray_frame is not None:
            self.__points, status, _ = cv2.calcOpticalFlowPyrLK(self.__old_gray_frame, gray, np.array(self.__points, dtype=np.float32), None, **self.__lk_params)
            self.__old_gray_frame = gray.copy()
        else:
            self.__old_gray_frame = gray.copy()
            return None
        return self.__points

    def init_points(self, points):
        self.__points = points
