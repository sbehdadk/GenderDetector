
import logging

import keras.backend as K
from keras import Input, Model
from keras.applications import MobileNet
from keras.layers import Dropout, Dense, GlobalAveragePooling2D


class MobileNetDeepEstimator:
    def __init__(self, image_size, alpha, num_neu, weights=None):

        if K.common.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
        self.alpha = alpha
        self.num_neu = num_neu
        self.weights = weights
        self.FC_LAYER_SIZE = 1024

    def __call__(self):
        logging.debug("Creating model...")

        inputs = Input(shape=self._input_shape)
        model_mobilenet = MobileNet(input_shape=self._input_shape, alpha=self.alpha, depth_multiplier=1, dropout=1e-3,
                                    include_top=False, weights=self.weights, input_tensor=None, pooling=None)
        
        model_mobilenet.trainable=False
        
        x = model_mobilenet(inputs)

        feat_a = GlobalAveragePooling2D()(x)
        feat_a = Dropout(0.5)(feat_a)
        # maybe not needed: (but test ;-) 
        # feat_a = Dense(self.FC_LAYER_SIZE, activation="relu")(feat_a)

        pred_g_softmax = Dense(2, activation='sigmoid', name='gender')(feat_a)
        #pred_a_softmax = Dense(self.num_neu, activation='softmax', name='age')(feat_a)

        model = Model(inputs=inputs, outputs=[pred_g_softmax])

        return model
