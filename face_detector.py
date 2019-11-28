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
        
    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x 
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:                   # Already a square.
            return box
        elif diff > 0:                  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:                           # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    def extract_square_facebox(self, image):
        """Extract face area from image."""
        _, raw_boxes = self.get_faceboxes(
            image=image)
        faceboxes = []
        for box in raw_boxes:
            # Move box down.
            diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs(diff_height_width / 2))
            box_moved = self.move_box(box, [0, offset_y])
            # Make box square.
            facebox = self.get_square_box(box_moved)

            if self.box_in_image(facebox, image):
                faceboxes.append(facebox)
        return faceboxes
