import traceback

import cv2
from tqdm import tqdm
import os
from video_reader import VideoReader
from face_recognition import FaceDetector
from mediapipe_detector import MediapipeDetector
from parse_time_series import ParseTimeSeries


class DatasetParser:
    def __init__(self, path, file_name, start_time=1800):
        print(start_time)
        self.video = VideoReader(path, start_time)
        print(self.video.total_frames)
        print(self.video.current_frame)
        self.face_detector = FaceDetector(debug=True)
        self.first_use = False
        self.vector = None
        self.__current_item = 0
        self.tracker = MediapipeDetector()
        self.time_series = ParseTimeSeries(file_name=file_name, current_time=[int(self.video.start_frame / 30),
                                                                              int(self.video.start_frame / 30) + 10],
                                           direct=file_name)

    def __cut_frame(self, frame):
        height, width, _ = frame.shape
        cut_height = height // 2
        cut_frame = frame[:cut_height, :]
        return cut_frame

    def run(self):
        for k in tqdm(range(1, self.video.total_frames - self.video.current_frame)):
            try:
                frame = self.video()
                frame = self.__cut_frame(frame)
                points = []
                if not self.first_use:
                    if k % 300 == 0:
                        borders, new_frame = self.face_detector.find_face(frame)
                        print('face')
                        points = self.tracker.first_use(frame, borders)
                        if len(points) == 0:
                            self.time_series.set_status()
                            continue
                        self.first_use = True
                else:
                    points = self.tracker.get_coords_from_face(frame)

                if len(points) == 0:
                    self.time_series.set_status()
                    self.first_use = False
                    self.__current_item = 0
                else:
                    self.time_series.add_in_vector(points)
            except Exception as e:
                print('here')
                print(e)
                traceback.print_exc()
                self.first_use = False
                self.time_series.set_status()


if __name__ == '__main__':
    pass