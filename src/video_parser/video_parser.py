import cv2


class VideoReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.video_capture = None
        self.frame_width = None
        self.frame_height = None
        self.frame_rate = None
        self.total_frames = None
        self.current_frame = 0
        self.update(file_path)

    def update(self, file_path):
        self.file_path = file_path
        self.video_capture = None
        self.frame_width = None
        self.frame_height = None
        self.frame_rate = None
        self.total_frames = None
        self.current_frame = 0

    def open(self):
        self.video_capture = cv2.VideoCapture(self.file_path)
        if not self.video_capture.isOpened():
            raise Exception(f"Failed to open video file: {self.file_path}")

        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def close(self):
        if self.video_capture is not None:
            self.video_capture.release()

    def read_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return None

        self.current_frame += 1
        return frame

    def get_frame_count(self):
        return self.total_frames

    def get_frame_rate(self):
        return self.frame_rate

    def get_frame_dimensions(self):
        return self.frame_width, self.frame_height

    def get_current_frame_number(self):
        return self.current_frame
