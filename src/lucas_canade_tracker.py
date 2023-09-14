import cv2
import numpy as np

class LucasKanadeTracker:
    def __init__(self):
        self.__old_gray_frame = None
        self.__lk_points = None
        self.__lk_params = dict(winSize=(15, 15), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        update_points, status, _ = cv2.calcOpticalFlowPyrLK(self.__old_gray_frame, gray, np.array(self.__lk_points, dtype=np.float32), None, **self.__lk_params)
        self.__old_gray_frame = gray.copy()
        self.__lk_points = update_points
        return update_points, status

    def init_points(self, points, image):
        # self.__lk_points = np.array(points).reshape(-1, 1, 2)
        self.__lk_points = points
        self.__old_gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
