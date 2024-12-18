import traceback

import cv2
from tqdm import tqdm
import os
from video_reader import VideoReader
from face_recognition import FaceDetector
from mediapipe_detector import MediapipeDetector
from parse_time_series import ParseTimeSeries
from ml.ml_pipeline import Pipeline


class Compute:
    def __init__(self, path):
        self.video = VideoReader(path, start_time=0)
        self.face_detector = FaceDetector(debug=True)
        self.first_use = False
        self.vector = None
        self.__current_item = 0
        self.tracker = MediapipeDetector()
        self.time_series = ParseTimeSeries()
        self.pipeline = Pipeline()

    def process(self):
        for k in tqdm(range(0, self.video.total_frames - self.video.current_frame)):
            try:
                frame = self.video()
                points = []
                if not self.first_use:
                    if k % 300 == 0:
                        borders, new_frame = self.face_detector.find_face(frame)
                        points = self.tracker.first_use(frame, borders)
                        if len(points) == 0:
                            self.time_series.set_status()
                            print('here', k)
                            continue
                        self.first_use = True
                else:
                    points = self.tracker.get_coords_from_face(frame)

                if len(points) == 0:
                    self.time_series.set_status()
                    self.first_use = False
                    self.__current_item = 0
                else:
                    data = self.time_series.compute(points)
                    if data is not None:
                        res = self.pipeline.predict(data)
                        print(f'{res = }')
            except Exception as e:
                print('here')
                print(e)
                traceback.print_exc()
                self.first_use = False
                self.time_series.set_status()


if __name__ == '__main__':
    computer = Compute('../videos/test3.mp4')
    computer.process()