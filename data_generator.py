import numpy as np
import tensorflow as tf
from random import randrange


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, params):
        'Initialization'
        self.sequence_length = params['sequence_length']
        self.t1 = params['t1']
        self.t2 = params['t2']
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
        x = np.empty((self.batch_size, self.sequence_length, self.vocab_encoded[0].shape[0]))
        y = np.empty((self.batch_size, self.classes_encoded[0].shape[0]))
        for i in range(self.batch_size):
            x[i], y[i] = self.__sample_generation()
        return x, y

    def __sample_generation(self):
        x = np.empty((self.sequence_length, self.vocab_encoded[0].shape[0]))
        for i in range(self.sequence_length):
            x[i] = self.vocab_encoded[randrange(4)]  # random symbol from a,b,c,d
        x[0] = self.vocab["s"]
        x[self.sequence_length - 1] = self.vocab["e"]
        class_index = randrange(4)
        x[self.t1] = self.vocab["x"] if class_index < 2 else self.vocab["y"]
        x[self.t2] = self.vocab["x"] if class_index % 2 == 0 else self.vocab["y"]
        y = self.classes_encoded[class_index]
        return x, y
