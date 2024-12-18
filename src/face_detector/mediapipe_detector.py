import numpy as np
import mediapipe as mp
import cv2

points_indexes = np.array(
    [8, 9, 168, 6, 197, 195, 5, 4, 164, 103, 104, 67, 69, 109, 108, 151, 10, 337, 338, 299, 297, 123, 50, 205, 206, 147,
     187, 207, 216, 213, 192, 214, 212, 138, 172, 135, 136, 169, 211, 170, 32, 140, 208, 171, 199, 175, 428, 396, 262,
     369, 431, 395, 430, 394, 266, 280, 425, 426, 411, 427, 436, 416, 434])

eye_1_indexes = np.array(
    [472, 470, 471, 468, 469, 113, 225, 224, 223, 222, 221, 189, 190, 243, 132, 112, 26, 22, 23, 24, 110, 25, 130, 247,
     33, 246, 161, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7, 30, 29, 27, 28, 56, 190])
eye_2_indexes = np.array(
    [413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453, 465, 464, 463, 414, 286, 258, 257, 259,
     260, 467, 359, 255, 339, 254, 253, 252, 256, 341, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380,
     381, 382, 362, 477, 476, 475, 474, 477])

face_indexes = np.array([i for i in range(468) if i not in eye_1_indexes and i not in eye_2_indexes])

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
        ih, iw, _ = image.shape
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                face = np.array([(int(landmark.x * iw), int(landmark.y * ih)) for landmark in face_landmarks.landmark])
                points.append(face[face_indexes])
                # points.append(face)

        return points

    def get_face_borders(self, image: np.array):
        borders = []
        points = self.get_coords(image)
        for face in points:
            x = [point[0] for point in face]
            y = [point[1] for point in face]
            borders.append([(min(x), min(y)), (max(x), max(y))])
        return borders, np.array(points)
