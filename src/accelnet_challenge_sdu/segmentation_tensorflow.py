import cv2 as cv
import numpy as np
import os
import rospkg
import tensorflow as tf
import threading


# Force CPU
# tf.config.set_visible_devices([], 'GPU')


# Do not immediately use all available GPU memory, but turn on memory growth
# for gpu in tf.config.experimental.list_physical_devices('GPU'):
#     tf.config.experimental.set_memory_growth(gpu, True)


# Label colors, BGR format
CLASS_COLORS = np.array([
    (0, 0, 0),      # background (black)
    (0, 255, 0),    # tool wrist/gripper (green)
    (0, 0, 255),    # needle (red)
    (255, 0, 0),    # tool shaft (blue)
    (0, 255, 255),  # thread (yellow)
], dtype=np.uint8)


class SegmentationModelV2:
    def __init__(self):
        package_path = rospkg.RosPack().get_path('accelnet_challenge_sdu')
        path = os.path.join(package_path, 'resources', 'segmentation_model_v2')
        self.model = tf.keras.models.load_model(path, compile=False)
        self.lock = threading.Lock()

    def generate_masks(self, img):
        # For a single image, cv.copyMakeBorder is faster than np.pad.
        # For multiple stacked images (a batch), use np.pad instead.
        padded = cv.copyMakeBorder(img, top=4, bottom=4, left=0, right=0, borderType=cv.BORDER_CONSTANT)
        padded = tf.keras.applications.mobilenet_v2.preprocess_input(padded)

        # Make prediction
        # Re-entrancy OK, but allow only a single predict() call at a time to
        # avoid out-of-memory errors.
        with self.lock:
            pred = self.model.predict(padded[np.newaxis,...])  # np.newaxis because batch size = 1

        # Make BGR image with colors corresponding to predicted labels
        pred0 = pred[0][4:-4,...]  # first and only batch; cut away padding
        idx = np.argmax(pred0, axis=-1)
        labels = np.take(CLASS_COLORS, idx, axis=0)  # np.take() is faster than CLASS_COLORS[idx]

        return labels


