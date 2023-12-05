from src import VideoReader, FaceDetector


class IFuture:
    def __init__(self, path: str, tracker, video_tracker: VideoReader = VideoReader,
                 face_detector: FaceDetector = FaceDetector,
                 debug: bool = False):
        self.video: VideoReader = video_tracker(path)
        self.video.open()
        self.face_detector: FaceDetector = face_detector(debug)
        self.tracker = tracker()

    def process(self):
        pass
