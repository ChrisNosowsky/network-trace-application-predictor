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
from sklearn.model_selection import train_test_split
import pandas as pd
from openpyxl import *
from sklearn import svm, preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression


class Network:

    def __init__(self, save_model=False, save_plot=False):
        self.proc_time = datetime.datetime.now().strftime('%b_%d_%Y_%H_%M')
        self.save_model = save_model
        self.save_plot = save_plot

    def load_data_pandas(self, dataPath):
        print("Loading data..")
        data = pd.read_excel(dataPath)
        # Y = data['label']
        data = data.loc[data['Grant received'] != 0]    # MAC UL TX ONLY
        Y = data['label']
        # print(X)
        X = data.drop(['label'], axis=1, inplace=False)
        # Adjust the random_state if you are getting a Dimensions must be equal error
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
        # print(X_train)
        print("loaded " + str(len(X_train)) + " training examples and " + str(len(X_test)) + " test examples")
        return X_train, X_test, Y_train, Y_test

    def load_data_numpy(self, x_train, x_test, y_train, y_test):
        train_data = x_train.to_numpy()
        train_labels = y_train.to_numpy()
        test_data = x_test.to_numpy()
        test_labels = y_test.to_numpy()
        return train_data, train_labels, test_data, test_labels

    def SVM_model(self, path):
        X_train, X_test, Y_train, Y_test = self.load_data_pandas(path)

        clf = svm.SVC(kernel='poly', C=1, probability=True)

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_scaled = scaler.transform(X_train)

        clf.fit(X_scaled, Y_train)

        X_test_scaled = scaler.transform(X_test)

        y_pred = clf.predict(X_test_scaled)
        print(accuracy_score(Y_test, y_pred))
        # accuracy, best I got is 0.336 PogChamp
        print((y_pred == Y_test).sum() / len(Y_test))

    def linear_regression(self, path):
        X_train, X_test, Y_train, Y_test = self.load_data_pandas(path)
        train_data, train_labels, test_data, test_labels = self.load_data_numpy(X_train, X_test, Y_train, Y_test)
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_labels)
        test_labels = label_encoder.fit_transform(test_labels)
        one = OneHotEncoder()
        train_labels = train_labels.reshape(-1, 1)
        test_labels = test_labels.reshape(-1, 1)
        train_labels = one.fit_transform(train_labels).toarray()
        test_labels = one.fit_transform(test_labels).toarray()
        print(test_labels[0])
        sc = StandardScaler()
        train_data = sc.fit_transform(train_data)
        test_data = sc.fit_transform(test_data)
        model = LinearRegression()
        model.fit(train_data, train_labels)
        print(model.score(test_data, test_labels))

    def keras_model(self, path):
        X_train, X_test, Y_train, Y_test = self.load_data_pandas(path)
        train_data, train_labels, test_data, test_labels = self.load_data_numpy(X_train, X_test, Y_train, Y_test)

        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_labels)
        test_labels = label_encoder.fit_transform(test_labels)
        one = OneHotEncoder()
        train_labels = train_labels.reshape(-1, 1)
        test_labels = test_labels.reshape(-1, 1)
        train_labels = one.fit_transform(train_labels).toarray()
        test_labels = one.fit_transform(test_labels).toarray()
        print(test_labels[0])
        sc = StandardScaler()
        train_data = sc.fit_transform(train_data)
        test_data = sc.fit_transform(test_data)
        print(train_data[0])

        num_features = len(train_data[0])
        num_classes = len(train_labels[0])
        # model = tf.keras.Sequential(tf.keras.layers.Dense(64, activation=tf.nn.relu,
        #                                                   input_shape=(
        #                                                       num_features,)))
        # model.add(tf.keras.layers.Dense(32, activation='relu'))
        # # model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
        # # optimizer = tf.keras.optimizers.Adam(lr=0.01)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, input_shape=(num_features,), activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

        try:
            history = model.fit(train_data,
                                train_labels,
                                validation_split=0.2,
                                epochs=500,  # Keep low for now
                                batch_size=512)
            acc = history.history['accuracy'][-1]
            # val_acc = history.history['val_accuracy'][-1]
            # loss = history.history['loss'][-1]
            # val_loss = history.history['val_loss'][-1]
            print('\n\nTrain Accuracy: ' + str(acc))
            # print('Validation Accuracy: ' + str(val_acc))
            # print('Loss: ' + str(loss))
            # print('Validation Loss: ' + str(val_loss) + '\n\n')
            score, acc = model.evaluate(test_data, test_labels,
                                        batch_size=512)
            print('Test loss:', score)
            print('Test accuracy:', acc)
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


