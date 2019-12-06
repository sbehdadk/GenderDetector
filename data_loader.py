from scipy.io import loadmat
import numpy as np
from random import shuffle
from utils import calc_age


class DataManager(object):
    def __init__(self, dataset_name='imdb', dataset_path=None, image_size=(48, 48)):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        if self.dataset_path is not None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'imdb':
            self.dataset_path = '/home/feras/GenderDetector/dataset/imdb_crop/imdb.mat'
        else:
            raise Exception('Invalid dataset')

    def get_data(self):
        if self.dataset_name == 'imdb':
            ground_truth_data = self._load_imdb()
        return ground_truth_data
#cleaning uo noisy labels
    def _load_imdb(self):
        face_score_treshold = 3
        dataset = loadmat(self.dataset_path)
        image_names_array = dataset['imdb']['full_path'][0, 0][0]
        gender_classes = dataset['imdb']['gender'][0, 0][0]

        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        #dob = dataset['imdb']['dob'][0, 0][0]
        #photo_taken = dataset['imdb']['photo_taken'][0, 0][0]
# filtering nun numeric values
        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        #age_classes = np.array([calc_age(photo_taken[i], dob[i]) for i in range(len(dob))])

        # 0 <= age_classes <= 100
        #valid_age_range = np.isin(age_classes, [x for x in range(101)])

        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        #mask = np.logical_and(mask, valid_age_range)

        image_names_array = image_names_array[mask]

        gender_classes = gender_classes[mask].tolist()
        #age_classes = age_classes[mask].tolist()

        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
        #return dict(zip(image_names, zip(gender_classes, age_classes)))
        return  dict(zip(image_names,gender_classes)) #convert two lists into a dictionary

def get_labels(dataset_name):
    if dataset_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    else:
        raise Exception('Invalid dataset name')

#scaling the images and applying transformations to them
def split_imdb_data(ground_truth_data, validation_split=.2, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle:
        shuffle(ground_truth_keys)
    training_split = 1 - validation_split
    num_train = int(training_split * len(ground_truth_keys))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys  #training 80% validation 20%



#validation_split = .2
#input_path = '/home/feras/age-gender-estimator-keras-master/dataset/imdb_crop/imdb.mat'
#dataset_name = 'imdb'
#data_loader = DataManager(dataset_name, dataset_path=input_path)
#ground_truth_data = data_loader.get_data()
#train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)
#print('Number of training samples:', len(train_keys))
#print('Number of validation samples:', len(val_keys))

