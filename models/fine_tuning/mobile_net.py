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
#from keras_utils import reset_tf_session
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau


class Schedule:
    def __init__(self, nb_epochs):
        self.epochs = nb_epochs

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.10:
            return 0.001
        elif epoch_idx < self.epochs * 0.25:
            return 0.0001
        elif epoch_idx < self.epochs * 0.60:
            return 0.00005
        return 0.00008


class MobileNetDeepEstimator:
    def __init__(self, image_size, classes=None, weights=None):

        if K.common.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
        #self.alpha = alpha
        #self.num_neu = num_neu
        self.nb_epochs = 50
        self.patience = 6
        self.model_path = 'models/fine_tuning'
        self.batch_size = 32
        self.weights = weights
        self.learning_rate = 0.0001
        self.FC_LAYER_SIZE = 1024
        
    def reset_tf_session1(self):
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        s = K.get_session()
        return s
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

    
    
    def __call__(self):
        logging.debug("Creating model...")
        #s = self.reset_tf_session1()
        nb_epochs=self.nb_epochs
        inputs = Input(shape=self._input_shape, name='input_1')
        #base_model = MobileNet(input_shape=self._input_shape, alpha=self.alpha, depth_multiplier=1, dropout=1e-3,
                                    #include_top=False, weights=self.weights, input_tensor=None, pooling=None)
        
        #model_mobilenet = InceptionResNetV2(input_shape=self._input_shape, alpha=self.alpha, depth_multiplier=1, dropout=1e-3,
        #                                   include_top=False, weights=self.weights, input_tensor=None, pooling=None)

        #base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None,
        #                                            input_shape=self._input_shape, pooling=None, classes=1000)

        #base_model = mobilenet_v2.MobileNetv2(input_shape=self._input_shape,
                                                     #weights='imagenet', include_top=False)
        base_model = MobileNet(input_shape=self._input_shape,
                                                       include_top=False,
                                                       weights='imagenet')
        #base_model.trainable = False
        base_model.summary()
        #x = base_model(inputs)
        x = base_model.output
             
        x = GlobalAveragePooling2D()(x)
        #feat_a = Dropout(0.5)(feat_a)
        x = Dense(self.FC_LAYER_SIZE, activation="relu")(x)

        prediction = Dense(2, activation='sigmoid', name='gender')(x)
        
        model = Model(inputs=base_model.input, outputs=prediction)
        model.summary()
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False
        # compile the model (should be done *after* setting layers to non-trainable)
        
        image_generator, train_keys, val_keys = self.__preprocessing__imdb()
        
        model.compile(optimizer=RMSprop(lr=self.learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        
        logging.debug("Running training...")
       
        #print(model.evaluate(image_generator.flow(mode='val'), batch_size=128, verbose=1))
        
        # save weights of best training epoch: monitor either val_loss or val_acc

        top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        callbacks_list = [
            ModelCheckpoint(top_weights_path, monitor='val_loss', verbose=1, save_best_only=True),
        ]
        
        history_imdb1 = model.fit_generator(image_generator.flow(mode='train'),
                                      steps_per_epoch=int(len(train_keys) / self.batch_size),
                                      epochs=nb_epochs/10, verbose=1,
                                      callbacks=callbacks_list,
                                      validation_data=image_generator.flow(mode='val'),
                                      validation_steps=int(len(val_keys) / self.batch_size))
        
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name)
            
        #base_model.trainable = True
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:

        # Fine tune from this layer onwards
        for layer in model.layers[:84]:
            layer.trainable = False
        for layer in model.layers[84:]:
            layer.trainable = True
            
        image_generator_imdb, train_keys_imdb, val_keys_imdb = self.__preprocessing__imdb()
        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        model.compile(optimizer=SGD(lr=self.learning_rate, momentum=0.9),
                      loss='binary_crossentropy',
                      metrics={'gender':'accuracy'})
        logging.debug("Model summary...")
        model.count_params()
        model.summary()

        #len(model.trainable_variables)
        
        #mode_callback
        early_stop = EarlyStopping('val_loss', patience=self.patience)
        reduce_lr = ReduceLROnPlateau(verbose=1, epsilon=0.001,
                                     patience=int(self.patience/2))
                
        logging.debug("Saving model...")
        mk_dir("models")
        with open(os.path.join("models/fine_tuning", "mobileNet.json"), "w") as f:
            f.write(model.to_json())
    
        mk_dir("checkpoints/fine_tuning")
    
        run_id = "MobileNet - " + str(self.batch_size) + " " + '' \
            .join(random
                  .SystemRandom()
                  .choice(string.ascii_uppercase) for _ in
                  range(10)
                  )
        print(run_id)
        model_checkpoint = ModelCheckpoint(
                    os.path.join('checkpoints/fine_tuning', 'mobileNet_imdb_128_10Epoch_Adam.{epoch:02d}-{val_loss:.2f}.hdf5'),
                    monitor="val_loss",
                    verbose=1,
                    save_best_only=True,
                    mode="auto",
                    save_weights_only=False)
        callbacks = [
            LearningRateScheduler(schedule=Schedule(nb_epochs)),
            reduce_lr, model_checkpoint, early_stop,
            TensorBoard(log_dir='logs/' + run_id)
        ]
    
        logging.debug("Running training...")
    
        history_imdb = model.fit_generator(image_generator_imdb.flow(mode='train'),
                                      steps_per_epoch=int(len(train_keys_imdb) / self.batch_size),
                                      epochs=nb_epochs, verbose=1,
                                      callbacks=callbacks,
                                      validation_data=image_generator_imdb.flow(mode='val'),
                                      validation_steps=int(len(val_keys_imdb) / self.batch_size))
        return model, history_imdb
