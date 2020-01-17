from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import matplotlib.pyplot as plt
import logging
import os
import tensorflow as tf
from tensorflow import keras
import random
import string
from keras import backend as K
import pandas as pd
from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
#from keras.optimizers import RMSprop
#from keras.optimizers import Adam
from keras.optimizers import SGD

from config import FINAL_WEIGHTS_PATH, IMG_SIZE
from data_generator import ImageGenerator
from data_loader import DataManager, split_imdb_data
#from models.mobileNet.mobile_net import MobileNetDeepEstimator
#import models.big_exception.cnn_all_Model as CNN 
#from models.fine_tuning.mobile_net import MobileNetDeepEstimator
from models.transfer_learning.inception_v3 import MobileNetDeepEstimator
#import models.fine_tuning.inception_v3_finetune as inception_tuning
#import models.vgg16.vgg16_full as vgg16
#import models.inceptionV3.inceptionmodel as inception
from utils import mk_dir

logging.basicConfig(level=logging.DEBUG)

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



def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    '''parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to history h5 file")'''
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")
    parser.add_argument("--patience", type=int, default=20,
                        help="patience_epochs")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    #input_path = args.input
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split
    patience = args.patience
    input_path = 'datasets/imdb_crop/imdb'
    logging.debug("Loading data...")

    dataset_name = 'imdb'
    data_loader = DataManager(dataset_name, dataset_path=input_path)
    ground_truth_data = data_loader.get_data()
    train_keys, validation_keys = split_imdb_data(ground_truth_data, validation_split)

    print("Samples: Training - {}, Validation - {}".format(len(train_keys), len(validation_keys)))
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    images_path = 'datasets/imdb_crop'

    image_generator = ImageGenerator(ground_truth_data, batch_size,
                                     input_shape[:2],
                                     train_keys, validation_keys,
                                     path_prefix=images_path,
                                     vertical_flip_probability=0
                                     )
    n_age_bins = 21
    alpha = 1
    model = MobileNetDeepEstimator(input_shape[0], alpha, n_age_bins, weights='imagenet')()
    #weights_path = 'models/fine_tuing/top_model_weights.h5'
    #Create the base model from the pre-trained model inception V3
    #model = inception.MobileNetDeepEstimator(input_shape[0], classes=1000, weights='imagenet')()
    #model = MobileNetDeepEstimator(input_shape[0], alpha, n_age_bins, weights='imagenet')()
    #model = vgg16.MobileNetDeepEstimator(IMG_SIZE, classes=2)()
    #model, history = MobileNetDeepEstimator(input_shape[0], weights='imagenet')()


    opt = SGD(lr=0.0001, momentum=0.9)

    model.compile(
        optimizer=opt,
        loss=["binary_crossentropy"],
        metrics=['accuracy'],
    )

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    mk_dir("models")
    with open(os.path.join("models/transfer_learning", "MobileNet.json"), "w") as f:
        f.write(model.to_json())

    mk_dir("checkpoints")

    run_id = "MobileNet - " + str(batch_size) + " " + '' \
        .join(random
              .SystemRandom()
              .choice(string.ascii_uppercase) for _ in
              range(10)
              )
    print(run_id)

    reduce_lr = ReduceLROnPlateau(
        verbose=1, epsilon=0.001, patience=4)

    callbacks = [
        LearningRateScheduler(schedule=Schedule(nb_epochs)),
        reduce_lr,
        ModelCheckpoint(
            os.path.join('checkpoints/transfer_learning', 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="auto"
        ),
        TensorBoard(log_dir='logs/' + run_id)
    ]

    logging.debug("Running training...")

    history = model.fit_generator(
                                image_generator.flow(mode='train'),
                                steps_per_epoch=int(len(train_keys) / batch_size),
                                epochs=nb_epochs,
                                callbacks=callbacks,
                                validation_data=image_generator.flow('val'),
                                validation_steps=int(len(validation_keys) / batch_size)
    )


    logging.debug("Saving weights...")
    model.save(os.path.join("models/transfer_learning", "mobileNet_model.h5"))
    model.save_weights(os.path.join("models/transfer_learning", FINAL_WEIGHTS_PATH), overwrite=True)
    pd.DataFrame(history.history).to_hdf(os.path.join("models/transfer_learning", "history.h5"), "history")
   
    logging.debug("plot the results...")
    logging.getLogger('matplotlib.font_manager').disabled = True
    hist_path='models/transfer_learning/history.h5'
    df = pd.read_hdf(hist_path, "history")
    input_dir = os.path.dirname(hist_path)
    plt.figure(figsize=(8,8))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    #plt.plot(df["loss"], label="train_loss")
    #plt.plot(df["age_loss"], label="loss (age)")
    #plt.plot(df["val_loss"], label="test_loss")
    #plt.plot(df["val_age_loss"], label="val_loss (age)")
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.title('loss_function')
    plt.savefig(os.path.join(input_dir, "loss.png"))
    plt.show()
    #plt.cla()

    plt.figure(figsize=(8,8))
    plt.plot(history.history['accuracy'], label = 'train')
    plt.plot(history.history['val_accuracy'], label = 'valid')
    #plt.plot(df["accuracy"], label="train_acc")
    #plt.plot(df["age_acc"], label="accuracy (age)")
    #plt.plot(df["val_accuracy"], label="test_acc")
    #plt.plot(df["val_age_acc"], label="val_accuracy (age)")
    plt.legend()
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.title('Accuracy')
    plt.savefig(os.path.join(input_dir, "accuracy.png"))
    plt.show()
    
    '''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show() '''


if __name__ == '__main__':
    main()
