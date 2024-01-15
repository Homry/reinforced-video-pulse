import numpy as np
from tqdm import tqdm

from video_reader import VideoReader
from face_recognition import FaceDetector
from mediapipe_detector import MediapipeDetector
from parse_time_series import ParseTimeSeries
import cv2


class DatasetParser:
    def __init__(self, path, file_name):
        self.video = VideoReader(path)
        self.face_detector = FaceDetector(debug=True)
        self.first_use = False
        self.tracker = MediapipeDetector()
        self.time_series = ParseTimeSeries(file_name=file_name, current_time=[int(self.video.start_frame/30),
                                                                              int(self.video.start_frame/30)+10])

    def __cut_frame(self, frame):
        height, width, _ = frame.shape
        cut_height = height // 2
        cut_frame = frame[:cut_height, :]
        return cut_frame

    def run(self):
        for _ in tqdm(range(1, self.video.total_frames - self.video.current_frame)):
            frame = self.video()
            frame = self.__cut_frame(frame)
            if not self.first_use:
                borders, new_frame = self.face_detector.find_face(frame)
                points = self.tracker.first_use(frame, borders)
                self.first_use = True
            else:
                points = self.tracker.get_coords_from_face(frame)
            if len(points) != 0:
                self.time_series.add_in_vector(points)
            else:
                self.time_series.set_status()


            # for i in points:
            #     cv2.circle(frame, (int(i[0]), int(i[1])), 2, (0, 0, 255), -1)
            #
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


if __name__ == '__main__':
    dataset_parser = DatasetParser('D:/датасет/room2115/video/2022-09-10 09-22-51.mp4', '2022-09-10 09-22-51')
    dataset_parser.run()