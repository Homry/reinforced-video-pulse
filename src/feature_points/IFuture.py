from src import VideoReader, FaceDetector, VideoSkipParser


class IFuture:
    def __init__(self, path: str, tracker, video_tracker: VideoSkipParser = VideoSkipParser,
                 face_detector: FaceDetector = FaceDetector,
                 debug: bool = False):
        self.video: VideoSkipParser = video_tracker(path)
        self.video.open()
        self.face_detector: FaceDetector = face_detector(debug)
        self.tracker = tracker()
        self.debug = debug

    def process(self):
        pass
