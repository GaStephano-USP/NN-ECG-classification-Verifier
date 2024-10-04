from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Softmax, Add, Flatten, Activation
from keras.layers import BatchNormalization, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import Adam, RMSprop, SGD
from keras.activations import relu
from keras.layers import LeakyReLU, PReLU
from keras.regularizers import l1, l2
import os

def get_resNet_model(input_shape):
    kernel_size = 5

    inp = Input(shape=input_shape)

    y = Conv1D(filters=32, kernel_size=kernel_size, strides=1)(inp)

    x = Conv1D(filters=32, kernel_size=kernel_size,
               strides=1, padding='same')(y)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=kernel_size,
               strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, y])
    x = Activation("relu")(x)
    y = MaxPooling1D(pool_size=5, strides=2)(x)

    x = Conv1D(filters=32, kernel_size=kernel_size,
               strides=1, padding='same')(y)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=kernel_size,
               strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, y])
    x = Activation("relu")(x)
    y = MaxPooling1D(pool_size=5, strides=2)(x)

    x = Conv1D(filters=64, kernel_size=kernel_size,
               strides=1, padding='same')(y)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters=64, kernel_size=kernel_size,
               strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    y = Conv1D(filters=64, kernel_size=kernel_size,
               strides=1, padding='same')(y)
    x = Add()([x, y])
    x = Activation("relu")(x)
    y = MaxPooling1D(pool_size=5, strides=2)(x)

    x = Conv1D(filters=64, kernel_size=kernel_size,
               strides=1, padding='same')(y)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters=64, kernel_size=kernel_size,
               strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, y])
    x = Activation("relu")(x)
    y = MaxPooling1D(pool_size=5, strides=2)(x)

    x = Conv1D(filters=128, kernel_size=kernel_size,
               strides=1, padding='same')(y)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters=128, kernel_size=kernel_size,
               strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    y = Conv1D(filters=128, kernel_size=kernel_size,
               strides=1, padding='same')(y)
    x = Add()([x, y])
    x = Activation("relu")(x)
    y = MaxPooling1D(pool_size=5, strides=2)(x)

    x = Flatten()(y)

    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = Dense(5)(x)
    x = Softmax()(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(beta_1=0.9, beta_2=0.999),
                  metrics=['categorical_accuracy'])

    return model


class BaseModel(object):

    def __init__(self, input_shape, batch_size=32):
        self.model = get_resNet_model(input_shape)
        self.batch_size = batch_size

    def fit(self, x_train, y_train, x_test=None, y_test=None):
        def lr_schedule(epoch, lr):
            epoch = epoch + 1  # indexed from 0
            if epoch == 8 or epoch == 13 or epoch == 18:
                return lr/10
            return lr
        valid_data = (x_test, y_test) if x_test is not None else None
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=20,
                       verbose=0,
                       validation_data=valid_data,
                       callbacks=[LearningRateScheduler(lr_schedule, verbose=0),
                                  ])
    
    #Added save routine for the process of translation
    def save_model(self, output_path):
        self.model.save(output_path)

    def predict(self, x_test):
        return self.model.predict(x_test, batch_size=self.batch_size)
