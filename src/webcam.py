import cv2
from src import VideoReader


class WebCam(VideoReader):
    def __init__(self):
        super().__init__(None)

    def open(self):
        self.video_capture = cv2.VideoCapture(2)
        if not self.video_capture.isOpened():
            raise Exception(f"Failed to open video from webcam")

        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.total_frames = 600


if __name__ == '__main__':
    cam = WebCam()
    cam.open()
    for i in range(cam.total_frames):
        img = cam.read_frame()
        cv2.imshow('crop_image', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
