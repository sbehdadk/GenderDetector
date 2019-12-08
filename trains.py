import argparse
import matplotlib.pyplot as plt
import logging
import os
import random
import string
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, EarlyStopping
#from keras.optimizers import RMSprop
from keras.optimizers import Adam
#from keras.optimizers import SGD

from config import FINAL_WEIGHTS_PATH, IMG_SIZE
from data_generator import ImageGenerator
from data_loader import DataManager, split_wiki_data
#from models.mobileNet.mobile_net import MobileNetDeepEstimator
#import models.mobileNet.cnn_all_Model as CNN 
import models.inceptionV4.inception_v4 as inception
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
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=70,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    #input_path = args.input
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split

    input_path = '/home/Sina/GenderDetector/datasets/wiki_crop/wiki.mat'
    #batch_size = 120
    patience = 4
    nb_epochs = 10
    #validation_split = .2

    logging.debug("Loading data...")

    dataset_name = 'wiki'
    data_loader = DataManager(dataset_name, dataset_path=input_path)
    ground_truth_data = data_loader.get_data()
    train_keys, val_keys = split_wiki_data(ground_truth_data, validation_split)

    print("Samples: Training - {}, Validation - {}".format(len(train_keys), len(val_keys)))
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    images_path = '/home/Sina/GenderDetector/datasets/wiki_crop'

    image_generator = ImageGenerator(ground_truth_data, batch_size,
                                     input_shape[:2],
                                     train_keys, val_keys,
                                     path_prefix=images_path,
                                     vertical_flip_probability=0
                                     )
    
    n_age_bins = 21
    alpha = 1
    num_classes = 2
    
    model = inception.create_inception_v4()
    #model = MobileNetDeepEstimator(input_shape[0], alpha, n_age_bins, weights='imagenet')()
    #model = CNN.big_XCEPTION(input_shape, num_classes)
    
    opt = Adam(lr=0.001)

    model.compile(
        optimizer=opt,
        loss=["categorical_crossentropy"],
        #metrics=['gender': 'accuracy'],
        metrics=['accuracy'],

    )        

    logging.debug("Model summary...")
    model.count_params()
    model.summary()


    logging.debug("Saving model...")
    mk_dir("models")
    with open(os.path.join("models", "MobileNet.json"), "w") as f:
        f.write(model.to_json())

    mk_dir("checkpoints/mobileNet")

    run_id = "MobileNet - " + str(batch_size) + " " + '' \
        .join(random
              .SystemRandom()
              .choice(string.ascii_uppercase) for _ in
              range(10)
              )
    print(run_id)

    #mode_callback
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau(
        verbose=1, epsilon=0.001, patience=int(patience/2))
    
    model_checkpoint = ModelCheckpoint(
            os.path.join('checkpoints/big_XCEPTION', 'wiki_128_10Epoch_Adam_freezed_Weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
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

    history = model.fit_generator(image_generator.flow(mode='train'),
                                  steps_per_epoch=int(len(train_keys) / batch_size),
                                  epochs=nb_epochs, verbose=1,
                                  callbacks=callbacks,
                                  validation_data=image_generator.flow('val'),
                                  validation_steps=int(len(val_keys) / batch_size))


    logging.debug("Saving weights...")
    model.save(os.path.join("models", "big_XCEPTION_model.h5"))
    model.save_weights(os.path.join("models", FINAL_WEIGHTS_PATH), overwrite=True)
    pd.DataFrame(history.history).to_hdf(os.path.join("models", "history.h5"), "history")
    
    logging.debug("plot the results...")
    
    df = pd.read_hdf(input_path, "history")
    input_dir = os.path.dirname(input_path)
    plt.plot(df["gender_loss"], label="loss (gender)")
    #plt.plot(df["age_loss"], label="loss (age)")
    plt.plot(df["val_gender_loss"], label="val_loss (gender)")
    #plt.plot(df["val_age_loss"], label="val_loss (age)")
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(input_dir, "loss.png"))
    plt.cla()

    plt.plot(df["gender_acc"], label="accuracy (gender)")
    #plt.plot(df["age_acc"], label="accuracy (age)")
    plt.plot(df["val_gender_acc"], label="val_accuracy (gender)")
    #plt.plot(df["val_age_acc"], label="val_accuracy (age)")
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(os.path.join(input_dir, "accuracy.png"))
    
if __name__ == '__main__':
    main()
