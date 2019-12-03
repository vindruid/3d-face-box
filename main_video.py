from face_detector import FaceDetector
from mark_detector import MarkDetector

import numpy as np
import cv2
import imutils
from imutils.video import VideoStream , FileVideoStream
import time

def get_headpose(h, w, landmarks_2d, landmarks_3d, lm_2d_index):
    f = w # column size = x axis length (focal length)
    u0, v0 = w / 2, h / 2 # center of image plane
    camera_matrix = np.array(
            [[f, 0, u0],
            [0, f, v0],
            [0, 0, 1]], dtype = np.float64
        )
        
    # Assuming no lens distortion
    dist_coeffs = np.zeros((4,1)) 

    # Filter Landmark 
    coords = []
    for i in lm_2d_index:
        coords += [[landmarks_2d[i,0], landmarks_2d[i,1]]]
    landmarks_2d = np.array(coords, dtype = np.float64)
    # Find rotation, translation
    (_, rotation_vector, translation_vector) = cv2.solvePnP(landmarks_3d, landmarks_2d, camera_matrix, distCoeffs = None)
    return rotation_vector, translation_vector, camera_matrix, dist_coeffs #rvec, tvec, cm, dc

def draw_front_box(image, color, rvec, tvec, cm, dc, b = 10.0):

    h, w, c = image.shape
    fs = ((h + w) / 2) / 500
    ls = round(fs * 2)        

    box = np.array([ #(horizontal, vertical, z)
        ( b,  b ,  b), #upper left (ul)
        ( b, -b ,  b), #bottom left (bl)
        (-b,  b ,  b), #upper right (ur)
        (-b, -b ,  b), #bottom right (br)
    ])

    # Draw from ul-ur > 
    box_lines_seq = np.array([
    (0, 2), (2,3), (3,1), (1,0)
    ])

    (projected_box, _) = cv2.projectPoints(box, rvec, tvec, cm, dc)
    pbox = projected_box[:, 0]
    for p in box_lines_seq:
        p1 = pbox[p[0]].astype(int) #point 1
        p2 = pbox[p[1]].astype(int) #point 2
        p1, p2 = tuple(p1), tuple(p2)

        cv2.line(image, p1, p2, color, ls)

def draw_front_box_corner(image, color, rvec, tvec, cm, dc, scale = 4, b = 10.0):

    h, w, c = image.shape
    fs = ((h + w) / 2) / 500
    ls = round(fs * 2)        

    box = np.array([ #(horizontal, vertical, z)
        ( b,  b ,  b), #upper left (ul)
        ( b, -b ,  b), #bottom left (bl)
        (-b,  b ,  b), #upper right (ur)
        (-b, -b ,  b), #bottom right (br)
    ])

    # Draw from ul-ur > 
    box_lines_seq = np.array([
    (0, 2), (2,3), (3,1), (1,0)
    ])


    (projected_box, _) = cv2.projectPoints(box, rvec, tvec, cm, dc)
    pbox = projected_box[:, 0]
    for p in box_lines_seq:
        p1 = pbox[p[0]].astype(int) #point 1
        p2 = pbox[p[1]].astype(int) #point 2

        extra_len = (((p1 - p2) / scale)).astype(int)

        p1_3 = p1 - extra_len #point 1.3 
        p1_6 = p2 + extra_len #point 1.6

        p1, p1_3, p1_6, p2 = tuple(p1), tuple(p1_3), tuple(p1_6), tuple(p2)
        
        cv2.line(image, p1, p1_3, color, ls)
        cv2.line(image, p1_6, p2, color, ls)


def main():

    # Initiate Class for Face & Mark Detector
    face_detector = FaceDetector()
    mark_detector = MarkDetector()

    # Landmark 3D for projection and landmark 2d index of corresponding mark
    landmarks_3d = np.array([
                [ 0.000000,  0.000000,  6.763430],   # 33 nose bottom edge
                [ 6.825897,  6.760612,  4.402142],   # 17 left brow left corner
                [ 1.330353,  7.122144,  6.903745],   # 21 left brow right corner
                [-1.330353,  7.122144,  6.903745],   # 22 right brow left corner
                [-6.825897,  6.760612,  4.402142],   # 26 right brow right corner
                [ 5.311432,  5.485328,  3.987654],   # 36 left eye left corner
                [ 1.789930,  5.393625,  4.413414],   # 39 left eye right corner
                [-1.789930,  5.393625,  4.413414],   # 42 right eye left corner
                [-5.311432,  5.485328,  3.987654],   # 45 right eye right corner
                [ 2.005628,  1.409845,  6.165652],   # 31 nose left corner
                [-2.005628,  1.409845,  6.165652],   # 35 nose right corner
                [ 2.774015, -2.080775,  5.048531],   # 48 mouth left corner
                [-2.774015, -2.080775,  5.048531],   # 54 mouth right corner
                [ 0.000000, -3.116408,  6.097667],   # 57 mouth central bottom corner
                [ 0.000000, -7.415691,  4.070434]    # 8 chin corner
            ], dtype=np.double)
    lm_2d_index = [33, 17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]

    # Define color for facebox
    color = (244, 134, 66)

    # Initiate Video Streaming
    file_name = 'assets/trump.gif'
    vs = FileVideoStream(file_name).start()
    time.sleep(2.0)

    frame = vs.read()

    while frame is not None:
        frame = imutils.resize(frame, width=800, height=600)
        (H,W) = frame.shape[:2]

        frame = cv2.flip(frame, 1) # Flip if using Webcam
        faceboxes = face_detector.extract_square_facebox(frame)

        if faceboxes is not None:
            for facebox in faceboxes:
                face_img = frame[facebox[1]: facebox[3],
                    facebox[0]: facebox[2]]

                marks = mark_detector.detect_marks(face_img)
                marks *= facebox[2] - facebox[0]
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]

                rvec, tvec, cm, dc = get_headpose(h = H, w = W, landmarks_2d= marks, landmarks_3d= landmarks_3d, lm_2d_index = lm_2d_index)

                draw_front_box(frame, color, rvec, tvec, cm, dc)

        cv2.imshow("3D Face Box", frame)
        # writer.write(frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
            
        frame = vs.read()

if __name__ == "__main__":
    main()