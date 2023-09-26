import numpy as np
import mediapipe as mp
import cv2


class MediapipeDetector:
    def __init__(self):
        self.__mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.use_points = []
        self.points_position = []

    def get_coords(self, image: np.array) -> list:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.__mp_face_mesh.process(image_rgb)
        points = []
        # Draw the face landmarks on the image.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    ih, iw, _ = image.shape
                    x, y = int(landmark.x * iw), int(landmark.y * ih)
                    points.append((x, y))
        return points

    def shift_points(self, tmp_points: np.array, borders):
        rect_up_start, rect_up_end, rect_down_start, rect_down_end = borders
        y_offset = rect_up_end[1] - rect_up_start[1]
        points = tmp_points[self.use_points == 1]
        position = {
            'up': rect_up_start,
            'down': (rect_down_start[0], rect_down_start[1]-y_offset)
        }
        for index, value in enumerate(points):
            points[index] = [value[0] - position[self.points_position[index]][0],
                             value[1]-position[self.points_position[index]][1]]

        return points

    def first_use(self, image, borders):
        tmp_points = self.get_coords(image)
        rect_up_start, rect_up_end, rect_down_start, rect_down_end = borders
        points = []
        y_offset = rect_up_end[1] - rect_up_start[1]
        for index, coords in enumerate(tmp_points):
            x, y = coords
            if rect_up_start[0] <= x <= rect_up_end[0] and rect_up_start[1] <= y <= rect_up_end[1]:
                points.append([x - rect_up_start[0], y - rect_up_start[1]])
                self.use_points.append(1)
                self.points_position.append('up')
            elif rect_down_start[0] <= x <= rect_down_end[0] and rect_down_start[1] <= y <= rect_down_end[1]:
                points.append([x - rect_down_start[0], y - rect_down_start[1] + y_offset])
                self.use_points.append(1)
                self.points_position.append('down')
            else:
                self.use_points.append(0)
        self.use_points = np.array(self.use_points)
        return points

    def get_coords_from_face(self, image: np.array, borders: tuple) -> list:
        points = self.get_coords(image)
        return self.shift_points(np.array(points), borders)


