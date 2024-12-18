import cv2
import numpy as np
import dlib

from dlib import rectangles


def crop_image(image: np.array, borders: list, offset: int = 0) -> (np.array, dlib.rectangles):
    x1, y1 = borders[0]
    x2, y2 = borders[1]
    x1, y1 = x1 - offset, y1 - offset
    x2, y2 = x2 + offset, y2 + offset
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image[y1:y2, x1:x2], dlib.rectangles(
        [dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)])


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/shape_predictor_81_face_landmarks.dat")


def detect_landmarks(frame: np.array, faces: dlib.rectangles, debug: bool = False) -> np.array:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = []
    for face in faces:
        landmarks = predictor(gray, face)
        face_points = []
        for n in range(0, 81):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if debug:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            face_points.append((x, y))
        points.append(face_points)
    return points


orb = cv2.ORB_create(nfeatures=10000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2,
                     scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)


def orb_detector(frame: np.array, debug: bool = False):
    face_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp = orb.detect(face_gray, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(face_gray, kp)
    if debug:
        face_frame = cv2.drawKeypoints(frame, kp, frame, color=(0, 255, 0), flags=0)
    return kp


def sift_detector(frame, nfeatures=500, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6,
                  debug=False):
    sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold,
                           edgeThreshold=edgeThreshold, sigma=sigma)

    face_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(face_gray, None)
    face_frame = cv2.drawKeypoints(frame, kp, frame, color=(0, 255, 0), flags=0)
    return kp


lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,
                                                         0.03))


def lk_dertector(old_frame, frame, p0):
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, np.array(p0, dtype=np.float32), None, **lk_params)
    return p1, st
