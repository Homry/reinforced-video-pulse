from .IFuture import IFuture
import numpy as np
from tqdm import tqdm
from src import VideoReader, FaceDetector, ParseTimeSeries, MediapipeDetector


class FutureParseDetector(IFuture):
    def __init__(self, file_path: str, tracker=MediapipeDetector, video_tracker: VideoReader = VideoReader,
                 face_detector: FaceDetector = FaceDetector,
                 debug: bool = False, save_path: str = './dataset_npy'):
        super().__init__(file_path, tracker, video_tracker, face_detector, debug)
        self.time_series = ParseTimeSeries(file_path, save_path)

    def init_detector(self):
        image = self.video.read_frame()
        borders, new_image = self.face_detector.find_face(image)
        points = self.tracker.first_use(image, borders)
        self.time_series.add_in_vector(points)

    def process(self):
        for _ in tqdm(range(1, self.video.total_frames)):
            try:
                image = self.video.read_frame()
                borders, new_image = self.face_detector.find_face(image)
                points = np.array(self.tracker.get_coords_from_face(image, borders))
                self.time_series.add_in_vector(points)
            except Exception as e:
                self.time_series.set_status()
