from typing import Callable

import numpy as np
import tensorflow as tf
from keras import backend as K

class CellDetectorModel:
    def __init__(self, network_fn: Callable):
        self.network = network_fn(self.data.input_shape, self.data.output_shape)
        self.network.summary()

    def fit(self, dataset, batch_size: int = 16, epochs: int = 50, callbacks: list = []):
        self.network.compile(optimizer='adam', loss='binary_crossentropy', metrics=[])
        # TODO: Replace with fit_generator
        self.network.fit(dataset, batch_size, epochs, callbacks)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)