import numpy as np
import tensorflow as tf
from random import randrange


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, params):
        'Initialization'
        self.sequence_length_min = params['sequence_length_min']
        self.sequence_length_max = params['sequence_length_max']
        self.t1_min = params['t1_min']
        self.t1_max = params['t1_max']
        self.t2_min = params['t2_min']
        self.t2_max = params['t2_max']
        self.batch_size = params['batch_size']
        self.epochs_num = params['epochs_num']
        self.steps_per_epoch = params['steps_per_epoch']

        self.vocab_encoded = tf.keras.utils.to_categorical(np.arange(8))
        self.vocab = {
            "a": self.vocab_encoded[0],
            "b": self.vocab_encoded[1],
            "c": self.vocab_encoded[2],
            "d": self.vocab_encoded[3],
            "s": self.vocab_encoded[4],
            "e": self.vocab_encoded[5],
            "x": self.vocab_encoded[6],
            "y": self.vocab_encoded[7],
        }
        self.classes_encoded = tf.keras.utils.to_categorical(np.arange(4))
        self.classes = {
            "XX": self.classes_encoded[0],
            "XY": self.classes_encoded[1],
            "YX": self.classes_encoded[2],
            "YY": self.classes_encoded[3],
        }

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        x = np.empty((self.batch_size, self.sequence_length_max, self.vocab_encoded[0].shape[0]))
        y = np.empty((self.batch_size, self.classes_encoded[0].shape[0]))
        for i in range(self.batch_size):
            x[i], y[i] = self.__sample_generation()
        return x, y

    def __sample_generation(self):
        x = np.empty((self.sequence_length_max, self.vocab_encoded[0].shape[0]))
        sequence_length = randrange(self.sequence_length_min, self.sequence_length_max)
        x[0] = self.vocab["s"]
        for i in range(1, sequence_length):
            x[i] = self.vocab_encoded[randrange(4)]  # random symbol from a,b,c,d
        for i in range(sequence_length, self.sequence_length_max):
            x[i] = self.vocab["e"]

        class_index = randrange(4)
        x[randrange(self.t1_min, self.t1_max)] = self.vocab["x"] if class_index < 2 else self.vocab["y"]
        x[randrange(self.t2_min, self.t2_max)] = self.vocab["x"] if class_index % 2 == 0 else self.vocab["y"]
        y = self.classes_encoded[class_index]
        return x, y
