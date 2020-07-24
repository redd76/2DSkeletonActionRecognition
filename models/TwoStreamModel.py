"""
Li 2019, Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation
Chao Li, Qiaoyong Zhong, Di Xie, Shiliang Pu
Hikvision Research Institute

The code for creating the model was taken from :
https://github.com/fandulu/Keras-for-Co-occurrence-Feature-Learning-from-Skeleton-Data-for-Action-Recognition
at 15.07.2020 (no releases)
"""

from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Permute, concatenate
from keras.models import Model
import keras.optimizers

class TwoStreamModel:
    """
    CNN with two branches
    Input1 is the motion image
    Input 2 is the motion image(t) - motion image (t-1)
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

    def load_model(self, classes=5):
        frame_l, joint_n, joint_d = self.img_rows, self.img_cols, self.n_channels
        input_joints = Input(name='joints', shape=(frame_l, joint_n, joint_d))
        input_joints_diff = Input(name='joints_diff', shape=(frame_l, joint_n, joint_d))

        ##########branch 1##############
        x = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_joints)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=64, kernel_size=(3, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        #x = Permute((1, 3, 2))(x)

        #x = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(x)
        #x = BatchNormalization()(x)
        #x = LeakyReLU()(x)
        x = Dropout(.1)(x)
        ##########branch 1##############

        ##########branch 2##############Temporal difference
        x_d = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_joints_diff)
        x_d = BatchNormalization()(x_d)
        x_d = LeakyReLU()(x_d)

        x_d = Conv2D(filters=64, kernel_size=(3, 1), padding='same')(x_d)
        x_d = BatchNormalization()(x_d)
        x_d = LeakyReLU()(x_d)

        #x_d = Permute((1, 3, 2))(x_d)

        #x_d = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(x_d)
        #x_d = BatchNormalization()(x_d)
        #x_d = LeakyReLU()(x_d)
        x_d = Dropout(.1)(x_d)
        ##########branch 2##############

        x = concatenate([x, x_d], axis=-1)

        x = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(0.1)(x)

        x = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(classes, activation="softmax")(x)

        model = Model([input_joints, input_joints_diff], x)
        # model = Model([input_joints], x)
        lr = 0.001
        adam = keras.optimizers.Adam(lr)
        model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
