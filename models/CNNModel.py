import keras.optimizers
from keras.layers import BatchNormalization
from keras.layers import Convolution2D, Flatten, Dense, Dropout
from keras.models import Sequential


class CNNModel:
    """
    Simple CNN-Model with 2 convoltions and a top layer to classify
    Convolution layers are normalized
    """
    img_rows, img_cols, n_channels = 60, 25, 3

    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_inputshape(self):
        return self.img_rows, self.img_cols, self.n_channels

    def reshape_input_data(self, x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, self.n_channels)
        x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, self.n_channels)
        return x_train, x_test

    def load_model(self, classes=10):
        # TODO build your own model here
        model = Sequential()
        model.add(Dropout(.3, input_shape=self.load_inputshape()))
        model.add(Convolution2D(512, kernel_size=(3,3), strides=(1,1), activation="relu", input_shape=self.load_inputshape()))
        model.add(BatchNormalization())
        model.add(Convolution2D(256, kernel_size=(3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(.1))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(classes, activation="softmax"))
        opt = keras.optimizers.Adam(lr=0.00001)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model




