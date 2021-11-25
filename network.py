#!/usr/bin/env python3
# ==============================================================================
# CSE 824
# Created on: November 25th, 2021
# Authors: Chris Nosowsky (nosowsky@msu.edu),
#          Hamzeh Alzweri (alzwerih@msu.edu)
#
# Builds our neural network
# ==============================================================================

import tensorflow as tf
import numpy as np
from keras import models, layers, optimizers


class Network:
    NUM_FEATURES = 10    # Will change later

    def __init__(self, save_model=False):
        self.save_model = save_model

    def load_data(self):
        train_data = np.array([])
        train_labels = np.array([])
        test_data = np.array([])
        test_labels = np.array([])
        return train_data, train_labels, test_data, test_labels

    def setup_model(self):
        train_data, train_labels, test_data, test_labels = self.load_data()
        model = tf.keras.Sequential(tf.keras.layers.Dense(128, activation=tf.nn.relu,
                                                          input_shape=(
                                                              self.NUM_FEATURES,)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(20, activation=tf.nn.softmax))
        # optimizer = tf.keras.optimizers.Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

