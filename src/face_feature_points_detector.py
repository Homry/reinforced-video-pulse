import time

import cv2
from src import VideoReader, MediapipeDetector, LucasKanadeTracker, FaceDetector, TimeSeries


class FaceFeaturePointsDetector:
    def __init__(self, path, time_series: TimeSeries):
        self.video = VideoReader(path)
        self.video.open()
        self.time_series = time_series
        self.mediapipe = MediapipeDetector()
        self.lukas_kanade = LucasKanadeTracker()
        self.face_detector = FaceDetector()

    def init_detector(self):
        image = self.video.read_frame()
        rect_up_start, rect_up_end, rect_down_start, rect_down_end = self.face_detector.find_face(image)
        points = []
        for i in self.mediapipe.get_coords_from_face(image):
            x, y = i
            if rect_up_start[0] <= x <= rect_up_end[0] and rect_up_start[1] <= y <= rect_up_end[1]:
                points.append([x, y])
                # cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            elif rect_down_start[0] <= x <= rect_down_end[0] and rect_down_start[1] <= y <= rect_down_end[1]:
                # cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                points.append([x, y])
        self.lukas_kanade.init_points(points, image.copy())
        self.time_series.init_vector(points)
        print(f'init vector {points}')

    def process_video(self):
        while self.video.current_frame != self.video.total_frames:
            image = self.video.read_frame()
            points, status = self.lukas_kanade.detect(image)
            self.time_series.add_in_vector(points, status)
        self.time_series.filter_by_len()





