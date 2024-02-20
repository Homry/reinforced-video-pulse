import cv2
from .video_parser import VideoReader


class VideoSkipParser(VideoReader):
    def __init__(self, path, time_=45000):
        super().__init__(path)
        self.start_frame = time_  # 30 min * 60 sec*30fps
        print("init")
        self.current_frame = self.start_frame#34200

    def open(self):
        self.video_capture = cv2.VideoCapture(self.file_path)
        if not self.video_capture.isOpened():
            raise Exception(f"Failed to open video from webcam")
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        print(self.total_frames)
        print(self.current_frame)
        print(self.total_frames-self.current_frame)


#  2022-09-10 09-22-51.mp4 [1662790981.4648633]


# 1663827841.1197596 reinf gt
