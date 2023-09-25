import time

import cv2
import numpy as np
from src import VideoReader, MediapipeDetector, LucasKanadeTracker, FaceDetector, TimeSeries


class FaceFeaturePointsDetector:
    def __init__(self, path, time_series: TimeSeries):
        self.video = VideoReader(path)
        self.video.open()
        self.time_series = time_series
        self.mediapipe = MediapipeDetector()
        self.lukas_kanade = LucasKanadeTracker()
        self.face_detector = FaceDetector()
        self.points_use_vector = []

    def init_detector(self):
        image = self.video.read_frame()
        rect_up_start, rect_up_end, rect_down_start, rect_down_end, new_image = self.face_detector.find_face(image)
        points = []
        y_offset = rect_up_end[1] - rect_up_start[1]
        count = 0
        for index, coords in enumerate(self.mediapipe.get_coords_from_face(image)):
            x, y = coords
            if rect_up_start[0] <= x <= rect_up_end[0] and rect_up_start[1] <= y <= rect_up_end[1]:
                points.append([x - rect_up_start[0], y - rect_up_start[1]])
                self.points_use_vector.append(1)
                # cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            elif rect_down_start[0] <= x <= rect_down_end[0] and rect_down_start[1] <= y <= rect_down_end[1]:
                # cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                points.append([x - rect_down_start[0], y - rect_down_start[1] + y_offset])
                self.points_use_vector.append(1)
            else:
                self.points_use_vector.append(0)

        print(f'count of mediapipe points = {count}')
        self.points_use_vector = np.array(self.points_use_vector)
        self.lukas_kanade.init_points(points, new_image.copy())
        self.time_series.init_vector(points)
        # print(f'init vector {points}')

    def process_video(self):
        while self.video.current_frame != self.video.total_frames:
            image = self.video.read_frame()
            _, _, _, _, new_image = self.face_detector.find_face(image)
            points = np.array(self.mediapipe.get_coords_from_face(image))
            points = points[self.points_use_vector == 1]
            if new_image is None:
                continue
            aaa = new_image.copy()

            # points, status = self.lukas_kanade.detect(new_image)
            for i in points:
                cv2.circle(image, (int(i[0]), int(i[1])), 2, (0, 0, 255), -1)
            # cv2.imshow('uio', image)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break
            # self.time_series.add_in_vector(points, status)
            self.time_series.add_in_vector(points)
        self.time_series.filter_by_len()
