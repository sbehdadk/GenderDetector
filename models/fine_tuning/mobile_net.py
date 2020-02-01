import logging
import os
import logging
import random
import string
from utils import mk_dir
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import Input, Model
from data_loader import DataManager, split_imdb_data
from keras.applications import MobileNet, InceptionResNetV2, inception_v3, vgg16, MobileNetV2
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from data_loader_wiki import DataManager_wiki, split_wiki_data
from config import FINAL_WEIGHTS_PATH, IMG_SIZE
from data_generator import ImageGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau


class MobileNetDeepEstimator:
    def __init__(self, image_size, classes=None, weights=None):
        #depends on backend,if we're using tensorflow or thiano reshaping the input 
        if K.common.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        #defining some parametes 
        self.nb_epochs = 50
        self.patience = 6
        self.model_path = 'models/fine_tuning'
        self.batch_size = 32
        self.weights = weights
        self.learning_rate = 0.0001
        self.FC_LAYER_SIZE = 1024
        
    #reseting graph nodes of the GPU at first    
    def reset_tf_session1(self):
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        s = K.get_session()
        return s

'''
    #preparing the wiki dataset for fiting into the model 
    def __preprocessing__wiki(self, val_split=0.2, batch_size=128):
        input_path = 'datasets/wiki_crop/wiki'
        dataset_name = 'wiki'
        data_loader = DataManager_wiki(dataset_name, dataset_path=input_path)
        ground_truth_data = data_loader.get_data()
        train_keys, val_keys = split_wiki_data(ground_truth_data, val_split)

        print("Samples: Training - {}, Validation - {}".format(len(train_keys), len(val_keys)))
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
        images_path = 'datasets/wiki_crop'

        image_generator_wiki = ImageGenerator(ground_truth_data, batch_size,
                                        input_shape[:2],
                                        train_keys, val_keys,
                                        path_prefix=images_path,
                                        vertical_flip_probability=0
                                        )
        logging.debug('data preprocessing...')
        return image_generator_wiki, train_keys, val_keys



    #preparing the imdb dataset for fiting into the model
    def __preprocessing__imdb(self, val_split=0.2, batch_size=128):
        input_path = 'datasets/imdb_crop/imdb'
        dataset_name = 'imdb'
        data_loader = DataManager(dataset_name, dataset_path=input_path)
        ground_truth_data = data_loader.get_data()
        train_keys_imdb, val_keys_imdb = split_wiki_data(ground_truth_data, val_split)

        print("Samples: Training - {}, Validation - {}".format(len(train_keys_imdb), len(val_keys_imdb)))
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
        images_path = 'datasets/imdb_crop'

        image_generator_imdb = ImageGenerator(ground_truth_data, batch_size,
                                        input_shape[:2],
                                        train_keys_imdb, val_keys_imdb,
                                        path_prefix=images_path,
                                        vertical_flip_probability=0
                                        )
        logging.debug('data preprocessing...')
        return image_generator_imdb, train_keys_imdb, val_keys_imdb

'''

    
    def __call__(self):
       logging.debug("Creating model...")
        K.set_learning_phase(0)

        inputs = Input(shape=self._input_shape, name='input_1')
        base_model = MobileNetV2(input_shape=self._input_shape,
                                      include_top=False, weights='imagenet')
        #base_model.trainable = False    
        #for layer in base_model.layers:
         #       layer.trainable = False
        
        for layer in base_model.layers[:136]:
           layer.trainable = False
        for layer in base_model.layers[136:]:
           layer.trainable = True
        base_model.summary()
        K.set_learning_phase(1)

        x = base_model(inputs)
        #x = base_model.output
        feat_a = GlobalAveragePooling2D()(x)
        feat_a = Dropout(0.5)(feat_a)
        #feat_a = Dense(self.FC_LAYER_SIZE, activation="relu")(feat_a)

        prediction = Dense(2, activation='softmax', name='gender')(feat_a)
        
        model = Model(inputs=inputs, outputs=prediction)
        
        
        model.summary()
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name)
            
        return model