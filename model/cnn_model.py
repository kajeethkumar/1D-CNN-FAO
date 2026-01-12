from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    BatchNormalization, Dropout,
    Flatten, Dense
)

def build_cnn(input_shape=(186, 1), num_classes=5):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, 6, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(3, strides=2, padding="same")(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, 3, activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, strides=2, padding="same")(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, 3, activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, strides=2, padding="same")(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)
