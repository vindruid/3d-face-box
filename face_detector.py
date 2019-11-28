import numpy as np

import cv2
import sys

class FaceDetector:
    """Detect human face from image"""

    def __init__(self,
                 dnn_proto_text='./models/face_detection/deploy.prototxt',
                 dnn_model='./models/face_detection/res10_300x300_ssd_iter_140000.caffemodel',
                 threshold=0.60):
        """Initialization"""
        self.face_ssd = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)
        self.detection_result = None
        self.threshold = threshold

    def get_faceboxes(self, image):
        """
        Get the bounding box of faces in image using dnn.
        """
        H, W, _ = image.shape

        confidences = []
        faceboxes = []
        # cv2.dnn.blobFromImage()
        self.face_ssd.setInput(cv2.dnn.blobFromImage(
            image, 1.0, (150, 150), (104.0, 177.0, 123.0), False, False))
        detections = self.face_ssd.forward()

        for result in detections[0, 0, :, :]:
            confidence = result[2]
            if confidence > self.threshold:
                x_left_bottom = int(result[3] * W)
                y_left_bottom = int(result[4] * H)
                x_right_top = int(result[5] * W)
                y_right_top = int(result[6] * H)
                confidences.append(confidence)
                faceboxes.append(
                    [x_left_bottom, y_left_bottom, x_right_top, y_right_top])

        self.detection_result = [faceboxes, confidences]

        return confidences, faceboxes


