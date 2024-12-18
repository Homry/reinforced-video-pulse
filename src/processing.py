import time

import cv2

from src.face_detector.mediapipe_detector import MediapipeDetector
from src.video_parser.video_parser import VideoReader
from src.video_parser.webcam import WebCam
from src.image_processing.image_processing import crop_image, detect_landmarks, orb_detector, sift_detector, lk_dertector
from src.signal_processing.signal_processing import SignalProcessing
import matplotlib.pyplot as plt

class Processing:
    def __init__(self, path_to_video, debug=False):
        self.video = VideoReader(path_to_video, start_time=0)#WebCam()
        # self.video = WebCam()
        self.face_detector = MediapipeDetector()
        # ##
        self.debug = debug
        self.signal = SignalProcessing(self.video.frame_rate)

    def process_video(self):
        # print(self.video.frame_rate)
        prev_frame = None
        p0 = None
        p1 = None
        for frame in self.video:
            image = frame.copy()
            borders, points = self.face_detector.get_face_borders(frame)
            if p0 is None:
                if len(points) != 0:
                    p0 = points[0]
            if len(borders) != 0:
                face = borders[0]
                face_frame, rectangles = crop_image(frame, face)
                if prev_frame is not None:
                    p1, st = lk_dertector(prev_frame, frame, p0)
                    p0 = p1
                prev_frame = frame

                if self.debug:
                    if p0 is not None:
                        for i, point in enumerate(p0):

                            cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
                    cv2.imshow(f'face_frame', face_frame)
                if self.signal.append(p0):
                    p0 = None


            if self.debug:
                plt.ion()
                cv2.imshow('frame', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == '__main__':
    path_to_video = "../videos/2022-09-10 17-23-17.mp4"#'D:/датасет/room2115/video/2022-09-10 09-22-51.mp4'
    processing = Processing(path_to_video, debug=True)
    processing.process_video()