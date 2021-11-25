#!/usr/bin/env python3
# ==============================================================================
# CSE 824
# Created on: November 25th, 2021
# Authors: Chris Nosowsky (nosowsky@msu.edu),
#          Hamzeh Alzweri (alzwerih@msu.edu)
#
# Builds our neural network
# ==============================================================================

import logging
import datetime
import numpy as np
import tensorflow as tf
from display import Display
from keras import models, layers, optimizers


class Network:
    NUM_FEATURES = 10    # Will change later

    def __init__(self, save_model=False, save_plot=False):
        self.proc_time = datetime.datetime.now().strftime('%b_%d_%Y_%H_%M')
        self.save_model = save_model
        self.save_plot = save_plot

    def load_data(self):
        train_data = np.array([])
        train_labels = np.array([])
        test_data = np.array([])
        test_labels = np.array([])
        return train_data, train_labels, test_data, test_labels

    def setup_model(self):
        train_data, train_labels, test_data, test_labels = self.load_data()

        train_labels = np.asarray(train_labels).astype('float32')
        test_labels = np.asarray(test_labels).astype('float32')

        model = tf.keras.Sequential(tf.keras.layers.Dense(128, activation=tf.nn.relu,
                                                          input_shape=(
                                                              self.NUM_FEATURES,)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(20, activation=tf.nn.softmax))
        # optimizer = tf.keras.optimizers.Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        try:
            history = model.fit(train_data,
                                train_labels,
                                validation_split=0.2,
                                epochs=10,  # Keep low for now
                                batch_size=256)
            acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1]
            loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            print('\n\nAccuracy: ' + str(acc))
            print('Validation Accuracy: ' + str(val_acc))
            print('Loss: ' + str(loss))
            print('Validation Loss: ' + str(val_loss) + '\n\n')

            if self.save_model:
                print("Saving trained model..")
                model_file_name = datetime.datetime.now().strftime('./models/model_'
                                                                   + self.proc_time + '.h5')
                model.save(model_file_name)
                print("Saved successful!")

            # Uncomment below to graph model performance
            # d = Display(proc_time=self.proc_time, history=history, save=False)
            # d.display_results()
        except ValueError as e:
            logging.exception(e)


