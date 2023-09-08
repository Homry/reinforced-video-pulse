import cv2
# from src import VideoReader, MediapipeDetector, LucasKanadeTracker


class FaceDetector:
    def __init__(self):
        self.__face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def find_face(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.__face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        rect_up_start, rect_up_end, rect_down_start, rect_down_end = None, None, None, None
        for (x, y, w, h) in faces:
            x = int(x + 0.25 * w)
            y = int(y + 0.05 * h)
            w = int(0.5 * w)
            h = int(0.9 * h)
            rect_up_start, rect_up_end = (x, y), (x + w, int(y + 0.2 * h))
            rect_down_start, rect_down_end = (x, int(y + 0.55 * h)), (x + w, int(y + h))


        return rect_up_start, rect_up_end, rect_down_start, rect_down_end


    # def process_video(self):
    #     for i in range(self.__video.total_frames):
    #         image = self.__video.read_frame()
    #         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         faces = self.__face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #         for (x, y, w, h) in faces:
    #
    #             x = int(x+0.25*w)
    #             y = int(y+0.05*h)
    #             w = int(0.5*w)
    #             h = int(0.9*h)
    #             rect_up_start, rect_up_end = (x,y),(x+w, int(y+0.2*h))
    #             rect_down_start, rect_down_end = (x, int(y+0.55*h)), (x + w, int(y + h))
    #             image = cv2.rectangle(image, rect_up_start, rect_up_end, (0, 255, 0), 1)
    #             image = cv2.rectangle(image, rect_down_start, rect_down_end, (0, 255, 0), 1)
    #             bounds = [(rect_up_start, rect_up_end), (rect_down_start, rect_down_end)]
    #
    #             if self.add is False:
    #                 points = []
    #                 for i in self.__mp.get_coords_from_face(image):
    #                     x, y = i
    #                     if rect_up_start[0] <= x <= rect_up_end[0] and rect_up_start[1] <= y <= rect_up_end[1]:
    #                         points.append([x, y])
    #                         # cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    #                     elif rect_down_start[0] <= x <= rect_down_end[0] and rect_down_start[1] <= y <= rect_down_end[1]:
    #                         # cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    #                         points.append([x, y])
    #                 self.__lk.init_points(points, image.copy())
    #                 self.add = True
    #                 self.__time.init_vector(points)
    #             else:
    #                 points = self.__lk.detect(image)
    #                 if points is not None:
    #                     self.__time.add_in_vector(points)
    #                     for i in points:
    #                         cv2.circle(image, (int(i[0]), int(i[1])), 2, (0, 0, 255), -1)
    #                 cv2.imshow('Face Landmarks Detection', image)
    #                 if cv2.waitKey(1) & 0xFF == ord('q'):
    #                     break
    #
    #         # Отображение результата
    #     self.__video.close()