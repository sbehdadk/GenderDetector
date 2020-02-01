import logging
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import Input, Model
from keras.applications import MobileNet, VGG16, MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dropout, Dense, GlobalAveragePooling2D


class MobileNetDeepEstimator:
    def __init__(self, image_size, alpha, num_neu, weights=None):
        #modifie the input depends on if we're using tensorflow or thiano in backend
        if K.common.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        #define some parameters
        self.alpha = alpha
        self.num_neu = num_neu
        self.weights = weights
        self.FC_LAYER_SIZE = 1024

    def __call__(self):
        logging.debug("Creating model...")
        #set the backend learning phase on false
        K.set_learning_phase(0)
        #define the input shape
        inputs = Input(shape=self._input_shape, name='input_1')

        #call the model       
        #model_mobilenet = InceptionResNetV2(input_shape=self._input_shape, alpha=self.alpha, depth_multiplier=1, dropout=1e-3,
        #                                   include_top=False, weights=self.weights, input_tensor=None, pooling=None)

        #base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None,
        #                                            input_shape=self._input_shape, pooling=None, classes=1000)

        base_model = MobileNetV2(input_shape=self._input_shape,
                                      include_top=False, weights='imagenet')
        

        #freez some layers
        for layer in base_model.layers[:136]:
           layer.trainable = False
        for layer in base_model.layers[136:]:
           layer.trainable = True
        base_model.summary()
        #set backend learning phase on True
        K.set_learning_phase(1)

        x = base_model(inputs)
        #x = base_model.output
        feat = GlobalAveragePooling2D()(x)
        feat = Dropout(0.5)(feat)
        prediction = Dense(2, activation='softmax', name='gender')(feat)
        
        model = Model(inputs=inputs, outputs=prediction)
        
        
        model.summary()
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name)
            
        return model
