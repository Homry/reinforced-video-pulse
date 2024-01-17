import traceback

import cv2
from tqdm import tqdm
import os
from video_reader import VideoReader
from face_recognition import FaceDetector
from mediapipe_detector import MediapipeDetector
from parse_time_series import ParseTimeSeries


class DatasetParser:
    def __init__(self, path, file_name, start_time=1800):
        print(start_time)
        self.video = VideoReader(path, start_time)
        print(self.video.total_frames)
        print(self.video.current_frame)
        self.face_detector = FaceDetector(debug=True)
        self.first_use = False
        self.vector = None
        self.__current_item = 0
        self.tracker = MediapipeDetector()
        self.time_series = ParseTimeSeries(file_name=file_name, current_time=[int(self.video.start_frame / 30),
                                                                              int(self.video.start_frame / 30) + 10],
                                           direct=file_name)

    def __cut_frame(self, frame):
        height, width, _ = frame.shape
        cut_height = height // 2
        cut_frame = frame[:cut_height, :]
        return cut_frame

    def run(self):
        for k in tqdm(range(1, self.video.total_frames - self.video.current_frame)):
            try:
                frame = self.video()
                frame = self.__cut_frame(frame)
                points = []
                if not self.first_use:
                    if k % 300 == 0:
                        borders, new_frame = self.face_detector.find_face(frame)
                        print('face')
                        points = self.tracker.first_use(frame, borders)
                        if len(points) == 0:
                            self.time_series.set_status()
                            continue
                        self.first_use = True
                else:
                    points = self.tracker.get_coords_from_face(frame)

                if len(points) == 0:
                    self.time_series.set_status()
                    self.first_use = False
                    self.__current_item = 0
                else:
                    self.time_series.add_in_vector(points)

                # for i in points:
                #     cv2.circle(frame, (int(i[0]), int(i[1])), 2, (0, 0, 255), -1)
                #
                # cv2.imshow('frame', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            except Exception as e:
                print('here')
                print(e)
                traceback.print_exc()
                self.first_use = False
                self.time_series.set_status()

    # def add_in_vector(self, points):
    #     if self.vector is None:
    #         self.vector = [[t[1]] for t in points]
    #     else:
    #         [self.vector[t].append(points[t][1]) for t, j in enumerate(points)]
    #
    #     self.__current_item += 1
    #     if self.__current_item == 300:
    #         self.__processing()



if __name__ == '__main__':
    # '2022-09-10 09-22-51.mp4'
    # '2022-09-10 13-13-58.mp4'
    # '2022-09-10 17-23-17.mp4',
    # '2022-09-12 17-23-47.mp4',
    # '2022-09-13 09-21-29.mp4',
    # '2022-09-14 17-19-36.mp4',
    # '2022-09-15 09-13-40.mp4',-
    # '2022-09-16 09-50-35.mp4',
    # '2022-09-16 13-47-44.mp4',-
    # '2022-09-16 17-34-55.mp4',-
    # '2022-09-17 09-41-55.mp4',-
    # '2022-09-17 17-42-15.mp4',-
    # '2022-09-22 09-24-26.mp4',-
    # '2022-09-24 09-17-57.mp4',-
    # '2022-09-24 13-20-27.mp4',-
    # '2022-09-24 17-48-42.mp4',
    # '2022-09-26 17-32-26.mp4',
    # '2022-09-27 13-13-21.mp4',
    # '2022-09-27 17-11-41.mp4',
    # '2022-09-09 12-00-59.mp4'] ? 2022-09-09 12-00-59.mp4

    data = [
        # ("D:/датасет/room2115/video/2022-09-27 17-11-41.mp4", '2022-09-27 17-11-41', 25200),
        #     ("D:/датасет/room2115/video/2022-09-27 13-13-21.mp4", '2022-09-27 13-13-21', 39600),
        #     ("D:/датасет/room2115/video/2022-09-26 17-32-26.mp4", '2022-09-26 17-32-26', 19800),
        #     ("D:/датасет/room2115/video/2022-09-24 17-48-42.mp4", '2022-09-24 17-48-42', 39600),
        #     ("D:/датасет/room2115/video/2022-09-24 13-20-27.mp4", '2022-09-24 13-20-27', 27000),
        #     ("D:/датасет/room2115/video/2022-09-24 09-17-57.mp4", '2022-09-24 09-17-57', 30600),
        #     ("D:/датасет/room2115/video/2022-09-22 09-24-26.mp4", '2022-09-22 09-24-26', 30600)
        #  ("D:/датасет/room2115/video/2022-09-09 12-00-59.mp4", ' 2022-09-09 12-00-59', 18000)
        # ("D:/датасет/room2115/video/2022-09-17 17-42-15.mp4", '2022-09-17 17-42-15', 30600),
        # ("D:/датасет/room2115/video/2022-09-16 17-34-55.mp4", '2022-09-16 17-34-55', 30600),
        # ("D:/датасет/room2115/video/2022-09-16 13-47-44.mp4", '2022-09-16 13-47-44', 30600),
        # ("D:/датасет/room2115/video/2022-09-16 09-50-35.mp4", '2022-09-16 09-50-35', 30600),
        # ("D:/датасет/room2115/video/2022-09-15 09-13-40.mp4", '2022-09-15 09-13-40', 30600),
        # ("D:/датасет/room2115/video/2022-09-14 17-19-36.mp4", '2022-09-14 17-19-36', 30600),
        # ("D:/датасет/room2115/video/2022-09-13 09-21-29.mp4", '2022-09-13 09-21-29', 30600),
        # ("D:/датасет/room2115/video/2022-09-12 17-23-47.mp4", '2022-09-12 17-23-47', 30600),#+
        ("D:/датасет/room2115/video/2022-09-10 17-23-17.mp4", '2022-09-10 17-23-17', 30600),
        ("D:/датасет/room2115/video/2022-09-10 13-13-58.mp4", '2022-09-10 13-13-58', 30600),
        ("D:/датасет/room2115/video/2022-09-10 09-22-51.mp4", '2022-09-10 09-22-51', 30600),
            ]
    for i in data:
        path, name, time = i

        dataset_parser = DatasetParser(path, name, time)
        dataset_parser.run()
    # base_path = '/mnt/c081fb1e-5b2e-4e53-a33b-4b322b8c6d90'
    # room_1 = os.listdir(f'{base_path}/part_1')
    # print(room_1)
    # room_2 = os.listdir(f'{base_path}/part_2')
    # print(room_2)
    # for room in room_1:
    #     for filename in os.listdir(f'{base_path}/part_1/{room}/video'):
    #         if filename != 'creation_time.txt':
    #             print(f'{base_path}/part_1/{room}/video/{filename}')
    #             parser = DatasetParser(path=f'{base_path}/part_1/{room}/video/{filename}', file_name=filename)

    # for room in room_2:
    #     for filename in os.listdir(f'{base_path}/part_1/{room}/video'):
    #         if filename != 'creation_time.txt':
    #             parser = DatasetParser(path=f'{base_path}/part_2/{room}/video/{filename}', file_name=filename)