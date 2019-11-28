# Source: https://github.com/junhwanjang/face_landmark_dnn

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils.generic_utils import custom_object_scope

from utils import smoothL1, relu6, DepthwiseConv2D, mask_weights
import cv2
import sys

# Model File Path #
current_model = "./models/landmark_model/Mobilenet_v1.hdf5"

###
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
###
class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    def __init__(self, mark_model=current_model):
        """Initialization"""
        # A face detector is required for mark detection.
        self.marks = None

        if mark_model.split(".")[1] == "pb":
            # Get a TensorFlow session ready to do landmark detection
            # Load a (frozen) Tensorflow model into memory.
            self.cnn_input_size = 64
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()

                with tf.gfile.GFile(mark_model, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            self.graph = detection_graph
            self.sess = tf.Session(graph=detection_graph, config=config)
            
        
        else:
            self.cnn_input_size = 64
            # with CustomObjectScope({'tf': tf}):
            with custom_object_scope({'smoothL1': smoothL1, 'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D, 'mask_weights': mask_weights, 'tf': tf}):
                self.sess = load_model(mark_model)

    def detect_marks(self, image ):
        """Detect marks from image"""
        face_img_landmark = cv2.resize(image, (self.cnn_input_size, self.cnn_input_size))
        face_img_landmark = cv2.cvtColor(face_img_landmark, cv2.COLOR_BGR2GRAY)

        face_img_landmark = face_img_landmark.reshape(1, self.cnn_input_size, self.cnn_input_size, 1)
        predictions = self.sess.predict_on_batch(face_img_landmark)
            
        # Convert predictions to landmarks.
        marks = np.array(predictions).flatten()
        marks = np.reshape(marks, (-1, 2))
        return marks