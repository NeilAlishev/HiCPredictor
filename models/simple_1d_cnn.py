import numpy as np
import random
import math
import scipy
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# Main class
class Model:
    def __init__(self):
        model_m = Sequential()

        model_m.add(layers.Conv1D(25, 50, activation='relu', input_shape=(50000, 4)))
        model_m.add(layers.Conv1D(25, 50, activation='relu'))
        model_m.add(layers.MaxPooling1D(5, strides = 2))

        model_m.add(layers.Conv1D(50, 25, activation='relu'))
        model_m.add(layers.MaxPooling1D(5, strides = 2))

        model_m.add(layers.Conv1D(50, 25, activation='relu'))
        model_m.add(layers.MaxPooling1D(20, strides = 4))

        model_m.add(layers.Conv1D(70, 20, activation='relu'))
        model_m.add(layers.MaxPooling1D(25, strides = 4))

        # dilated layers
        model_m.add(layers.Conv1D(100, 15, activation='relu', dilation_rate = 2))
        model_m.add(layers.Conv1D(100, 15, activation='relu', dilation_rate = 2))
        model_m.add(layers.MaxPooling1D(25, strides = 4))

        model_m.add(layers.Flatten())
        model_m.add(layers.Dense(2500, activation='linear'))

        # make model
        self.model = model_m


    def get_model(self):
        return self.model