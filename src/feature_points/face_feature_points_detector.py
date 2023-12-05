import time

import cv2
import tqdm
import numpy as np
from src import VideoReader, MediapipeDetector, LucasKanadeTracker, FaceDetector, TimeSeries


class FaceFeaturePointsDetector:
    def __init__(self, path, time_series: TimeSeries, debug: bool = True, handler='media', time_=0):
        self.handler = handler
        self.video = VideoSkipParser(path, time_)
        # self.video = VideoReader(path)
        self.video.open()
        self.time_series = time_series
        self.mediapipe = MediapipeDetector()
        self.lukas_kanade = LucasKanadeTracker()
        self.face_detector = FaceDetector(debug)
        self.points_use_vector = []
        self.debug = debug
        self.counter = 0
        self.prev_points = []
        self.prev_image = None
        self.init_lk = True

    def init_detector(self):
        image = self.video.read_frame()
        borders, new_image = self.face_detector.find_face(image)
        points = self.mediapipe.first_use(image, borders)
        self.time_series.init_vector(points)

    def process_video(self):
        for _ in tqdm.tqdm(range(1, self.video.total_frames)):
            image = self.video.read_frame()
            borders, new_image = self.face_detector.find_face(image)

            if new_image is None:
                continue
            if self.counter % 150 == 0:

                points = np.array(self.mediapipe.get_coords_from_face(image, borders))
                self.prev_points = points
                self.prev_image = new_image.copy()
                self.init_lk = True
            else:
                if self.handler == 'media':
                    points = np.array(self.mediapipe.get_coords_from_face(image, borders))
                else:
                    if self.init_lk is True:
                        self.init_lk = False
                        self.lukas_kanade.init_points(self.prev_points, self.prev_image)
                    points = self.lukas_kanade.detect(new_image.copy())
                #
            if self.debug:
                crop_image = new_image.copy()
                for i in points:
                    cv2.circle(crop_image, (int(i[0]), int(i[1])), 2, (0, 0, 255), -1)
                cv2.imshow('crop_image', crop_image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            # self.time_series.add_in_vector(points, status)
            self.time_series.add_in_vector(points)
            self.counter += 1
        # self.time_series.filter_by_len()
