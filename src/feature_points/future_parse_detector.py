from .IFuture import IFuture
import numpy as np
from tqdm import tqdm
import cv2
from src import VideoReader, FaceDetector, ParseTimeSeries, MediapipeDetector, VideoSkipParser
import traceback
import matplotlib.pyplot as plt


class FutureParseDetector(IFuture):
    def __init__(self, file_path: str, tracker=MediapipeDetector, video_tracker: VideoSkipParser = VideoSkipParser,
                 face_detector: FaceDetector = FaceDetector,
                 debug: bool = False, save_path: str = './dataset_npy'):
        super().__init__(file_path, tracker, video_tracker, face_detector, debug)
        self.time_series = ParseTimeSeries(file_path, save_path, current_time=[int(self.video.start_frame/30), int(self.video.start_frame/30)+10])

    def init_detector(self):
        image = self.video.read_frame()
        borders, new_image = self.face_detector.find_face(image)
        points = self.tracker.first_use(image, borders)
        self.time_series.add_in_vector(points)

    def process(self):
        print(self.video.current_frame)
        for _ in tqdm(range(1, self.video.total_frames-self.video.current_frame)):
            try:
                image = self.video.read_frame()
                borders, new_image = self.face_detector.find_face(image)

                points = np.array(self.tracker.get_coords_from_face(image, borders))
                self.time_series.add_in_vector(points)
                if self.debug:
                    crop_image = new_image.copy()
                    for i in points:
                        cv2.circle(crop_image, (int(i[0]), int(i[1])), 2, (0, 0, 255), -1)
                    cv2.imshow('___', crop_image)
                    cv2.imshow('full', image)
                    if cv2.waitKey(1) & 0xFF == 27:
                        exit(1)
            except Exception as e:
                self.time_series.set_status()
                print(e)
                traceback.print_exc()
