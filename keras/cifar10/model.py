# -- coding: utf8 --

import keras
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras_contrib.applications import DenseNet


def get_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def get_densenet_model():
    keras.backend.set_image_data_format('channels_last')
    img_rows, img_cols = 32, 32
    img_channels = 3

    # Parameters for the DenseNet model builder
    img_dim = (img_channels, img_rows, img_cols) if keras.backend.image_data_format() == 'channels_first' else (
        img_rows, img_cols, img_channels)
    depth = 40
    nb_dense_block = 3
    growth_rate = 12
    nb_filter = 16
    dropout_rate = 0.0  # 0.0 for data augmentation

    # Create the model (without loading weights)
    model = DenseNet(depth=depth, nb_dense_block=nb_dense_block,
                     growth_rate=growth_rate, nb_filter=nb_filter,
                     dropout_rate=dropout_rate,
                     input_shape=img_dim,
                     weights=None)

    optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model
