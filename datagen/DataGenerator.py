import numpy as np
import cv2
from keras.utils import Sequence
import keras


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    ALEXNET = "AlexNet"
    DENSENET = "DenseNet"
    SQUEEZENET = "SqueezeNet"
    TWOSTREAMNET = "TwoStreamNet"
    NNET = "NNet"

    def __init__(self, list_IDs, m_input_data, labels, model, use_all_joints,
                 to_fit=True, batch_size=32, dim=(150, 26),
                 n_channels=1, n_classes=5, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.m_input_data = m_input_data
        self.labels = labels
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.model = model
        self.use_all_joints = use_all_joints

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        T = np.empty((self.batch_size, *self.dim, self.n_channels))
        if self.model == self.NNET:
            X = np.empty((self.batch_size, *self.dim))
            T = np.empty((self.batch_size, *self.dim))
        #V = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = self._load_image(self.m_input_data[ID])
            if not self.use_all_joints:
                img = img[:, 0:14]
            elif self.model in [self.NNET]:
                X[i,] = img.flatten()
            elif self.model in [self.ALEXNET, self.DENSENET, self.SQUEEZENET]:
                img = DataGenerator.resize_img(img, self.dim)
                X[i,] = img
            else:
                X[i,] = img
            if self.model == self.TWOSTREAMNET:
                shifted = np.roll(img, 1, 0)
                shifted[-1] = img[-1]
                T[i,] = shifted - img
        if self.model == self.TWOSTREAMNET:
            return [X, T]
        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self.labels[ID]

        return keras.utils.to_categorical(y, num_classes=self.n_classes)

    def _load_grayscale_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        return img

    def _load_image(self, image_path):
        img = cv2.imread(image_path)
        img = img / 255.0
        return img

    def sumeveryxrows(self, myarray, x):
        return [np.sum(myarray[n: n + x]) for n in range(0, len(myarray), x)]

    @staticmethod
    def resize_img(img, dim):
        rimg = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_CUBIC)
        return rimg

    @staticmethod
    def image_to_net_input(img, model, dim):
        img = img[:, 0:dim[1]]
        if model in [DataGenerator.NNET]:
            return img.flatten()
        elif model in [DataGenerator.ALEXNET, DataGenerator.DENSENET, DataGenerator.SQUEEZENET]:
            img = DataGenerator.resize_img(img, dim)
            return img
        if model == DataGenerator.TWOSTREAMNET:
            shifted = np.roll(img, 1, 0)
            shifted[-1] = img[-1]
            return [img, shifted - img]
        return img