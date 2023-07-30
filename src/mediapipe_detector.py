import numpy as np
import mediapipe as mp
import cv2

class MediapipeDetector:
    def __init__(self):
        self.__mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_coords_from_face(self, image) -> list:
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
