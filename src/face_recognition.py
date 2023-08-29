import cv2
from src import VideoReader, MediapipeDetector, LucasKanadeTracker

class FaceDetector:
    def __init__(self, path, forehead_time_series, mouth_time_series):
        self.__video = VideoReader(path)

        self.__video.open()
        print(self.__video.total_frames)
        self.__face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.__mp = MediapipeDetector()
        self.__lk_forehead = LucasKanadeTracker()
        self.__lk_mouth = LucasKanadeTracker()
        self.add = False
        self.__forehead_time_series = forehead_time_series
        self.__mouth_time_series = mouth_time_series

    def process_video(self):
        for i in range(self.__video.total_frames):
            image = self.__video.read_frame()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.__face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:

                x = int(x+0.25*w)
                y = int(y+0.05*h)
                w = int(0.5*w)
                h = int(0.9*h)
                rect_up_start, rect_up_end = (x,y),(x+w, int(y+0.2*h))
                rect_down_start, rect_down_end = (x, int(y+0.55*h)), (x + w, int(y + h))
                image = cv2.rectangle(image, rect_up_start, rect_up_end, (0, 255, 0), 1)
                image = cv2.rectangle(image, rect_down_start, rect_down_end, (0, 255, 0), 1)
                bounds = [(rect_up_start, rect_up_end), (rect_down_start, rect_down_end)]

                if self.add is False:
                    points = []
                    for i in self.__mp.get_coords_from_face(image):
                        x, y = i
                        if rect_up_start[0] <= x <= rect_up_end[0] and rect_up_start[1] <= y <= rect_up_end[1]:
                            points.append([x, y])
                            # cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                        elif rect_down_start[0] <= x <= rect_down_end[0] and rect_down_start[1] <= y <= rect_down_end[1]:
                            # cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                            points.append([x, y])
                    self.__lk.init_points(points)
                    self.add = True
                    self.__time.init_vector(points)
                points = self.__lk.detect(image)
                if points is not None:
                    self.__time.add_in_vector(points)

                    for i in points:
                        cv2.circle(image, (int(i[0]), int(i[1])), 2, (0, 0, 255), -1)

                cv2.imshow('Face Landmarks Detection', image)

            # Отображение результата
        self.__video.close()

    def test_process(self):
        for i in range(self.__video.total_frames):
            image = self.__video.read_frame()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.__face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:

                x = int(x+0.25*w)
                y = int(y+0.05*h)
                w = int(0.5*w)
                h = int(0.9*h)
                rect_forehead_start, rect_forehead_end = (x,y),(x+w, int(y+0.2*h))

                rect_mouth_start, rect_mouth_end = (x, int(y+0.55*h)), (x + w, int(y + h))
                forehead_image = image[rect_forehead_start[1]:rect_forehead_end[1], rect_forehead_start[0]:rect_forehead_end[0]]
                mouth_image = image[rect_mouth_start[1]:rect_mouth_end[1], rect_mouth_start[0]:rect_mouth_end[0]]
                image = cv2.rectangle(image, rect_forehead_start, rect_forehead_end, (0, 255, 0), 1)
                image = cv2.rectangle(image, rect_mouth_start, rect_mouth_end, (0, 255, 0), 1)
                if self.add is False:
                    forehead_points = []
                    mouth_points = []
                    for i in self.__mp.get_coords_from_face(image):
                        x, y = i
                        if rect_forehead_start[0] <= x <= rect_forehead_end[0] and rect_forehead_start[1] <= y <= \
                                rect_forehead_end[1]:
                            forehead_points.append([x-rect_forehead_start[0], y-rect_forehead_start[1]])
                            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                        elif rect_mouth_start[0] <= x <= rect_mouth_end[0] and rect_mouth_start[1] <= y <= \
                                rect_mouth_end[1]:
                            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                            mouth_points.append([x-rect_mouth_start[0], y-rect_mouth_start[1]])
                    self.__lk_forehead.init_points(forehead_points)
                    self.__lk_mouth.init_points(mouth_points)
                    self.add = True
                    self.__forehead_time_series.init_vector(forehead_points)
                    self.__mouth_time_series.init_vector(mouth_points)
                else:
                    forehead_points = self.__lk_forehead.detect(forehead_image)
                    print(forehead_points)
                    if forehead_points is not None:
                        self.__forehead_time_series.add_in_vector(forehead_points)
                    for i in forehead_points:
                        cv2.circle(forehead_image, (int(i[0]), int(i[1])), 2, (0, 0, 255), -1)

                    mouth_points = self.__lk_mouth.detect(mouth_image)
                    if mouth_points is not None:
                        self.__mouth_time_series.add_in_vector(mouth_points)
                        for i in mouth_points:
                            cv2.circle(mouth_image, (int(i[0]), int(i[1])), 2, (0, 0, 255), -1)




                cv2.imshow('Face Landmarks Detection 0', image)
                cv2.imshow('Face Landmarks Detection 1', forehead_image)
                cv2.imshow('Face Landmarks Detection 2', mouth_image)
                cv2.waitKey(0)
