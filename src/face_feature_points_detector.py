import time

import cv2
import numpy as np
from src import VideoReader, MediapipeDetector, LucasKanadeTracker, FaceDetector, TimeSeries


class FaceFeaturePointsDetector:
    def __init__(self, path, time_series: TimeSeries, debug: bool = True):
        self.video = VideoReader(path)
        self.video.open()
        self.time_series = time_series
        self.mediapipe = MediapipeDetector()
        self.lukas_kanade = LucasKanadeTracker()
        self.face_detector = FaceDetector(debug)
        self.points_use_vector = []
        self.debug = debug

    def init_detector(self):
        image = self.video.read_frame()
        borders, new_image = self.face_detector.find_face(image)
        points = self.mediapipe.first_use(image, borders)
        self.time_series.init_vector(points)

    def process_video(self):
        while self.video.current_frame != self.video.total_frames:
            image = self.video.read_frame()
            borders, new_image = self.face_detector.find_face(image)

            if new_image is None:
                continue
            points = np.array(self.mediapipe.get_coords_from_face(image, borders))
            if self.debug:
                crop_image = new_image.copy()
                for i in points:
                    cv2.circle(crop_image, (int(i[0]), int(i[1])), 2, (0, 0, 255), -1)
                cv2.imshow('crop_image', crop_image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            # self.time_series.add_in_vector(points, status)
            self.time_series.add_in_vector(points)
        # self.time_series.filter_by_len()
