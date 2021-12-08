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
from display import Display
import tensorflow as tf
from tensorflow.keras import Sequential
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm, preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from keras.regularizers import L1L2
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class Network:

    def __init__(self, save_model=False, save_plot=False):
        self.proc_time = datetime.datetime.now().strftime('%b_%d_%Y_%H_%M')
        self.save_model = save_model
        self.save_plot = save_plot
        self.num_features = 0
        self.num_classes = 0

    @staticmethod
    def load_data_pandas(dataPath):
        print("Loading data..")
        data = pd.read_excel(dataPath)
        if 'Grant received' in data.columns:
            data = data.loc[data['Grant received'] != 0]    # MAC UL TX ONLY
        Y = data['label']
        # print(X)
        X = data.drop(['label'], axis=1, inplace=False)
        # X = data.drop(data.columns[[1, 69]], axis=1, inplace=True)
        # Adjust the random_state if you are getting a Dimensions must be equal error (I use 7 first few, 3 for RLC_UL)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
        # print(X_train)
        print("loaded " + str(len(X_train)) + " training examples and " + str(len(X_test)) + " test examples")
        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def load_data_numpy(x_train, x_test, y_train, y_test):
        train_data = x_train.to_numpy()
        train_labels = y_train.to_numpy()
        test_data = x_test.to_numpy()
        test_labels = y_test.to_numpy()
        return train_data, train_labels, test_data, test_labels

    def SVM_model(self, path):
        X_train, X_test, Y_train, Y_test = self.load_data_pandas(path)
        train_data, train_labels, test_data, test_labels = self.load_data_numpy(X_train, X_test, Y_train, Y_test)
        train_labels, test_labels = self.encode_data(train_labels, test_labels)
        train_labels, test_labels = self.one_hot_encode(train_labels, test_labels)
        train_data, test_data = self.normalize_data(train_data, test_data)
        num_features = len(train_data[0])
        num_classes = len(train_labels[0])
        model = Sequential(
            [
                tf.keras.Input(shape=(num_features,)),
                RandomFourierFeatures(
                    output_dim=4096, scale=10.0, kernel_initializer="gaussian"
                ),
                tf.keras.layers.Dense(num_classes),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.hinge,
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
        )

        history = model.fit(train_data, train_labels, epochs=500, batch_size=512, validation_split=0.2)
        score, acc = model.evaluate(test_data, test_labels)
        print('SVM  loss: {}%'.format(round(score * 100, 2)))
        print('SVM  accuracy: {}%'.format(round(acc * 100, 2)))
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(loss) + 1)


        plt.plot(epochs, loss, 'b-', label='Training loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation loss')
        plt.title('SVM Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        #plt.show()
        plt.savefig("SVM  loss")
        plt.clf()

        acc = history.history['acc']
        val_acc = history.history['val_acc']

        plt.plot(epochs, acc, 'y-', label='Training acc')
        plt.plot(epochs, val_acc, 'g-', label='Validation acc')
        plt.title('SVM training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("SVM accuracy")
        #plt.show()
        return

    def SVM_model2(self, path):
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

    @staticmethod
    def perform_grid_search_logistic_regression(train_data, train_labels, test_data, test_labels):
        model = LogisticRegression()

        solvers = ['newton-cg', 'lbfgs', 'liblinear']
        penalty = ['l2']
        c_values = [100, 10, 1.0, 0.1, 0.01]
        parameters = dict(solver=solvers, penalty=penalty, C=c_values)
        grid = GridSearchCV(model, param_grid=parameters, verbose=True, n_jobs=-1, scoring='accuracy', error_score=0)
        train_labels = train_labels.reshape(-1, 1)
        grid_results = grid.fit(train_data, train_labels.ravel())
        print("Best: %f using %s" % (grid_results.best_score_, grid_results.best_params_))
        """
        LogisticRegression with best Hyper-parameters: 
            Best: 0.306495 using {'C': 1.0, 'penalty': 'l2', 'solver': 'newton-cg'}
        """
        print("score = %3.2f" % (grid.score(test_data, test_labels)))
        print(grid_results.predict(test_data))
        print(grid_results.best_estimator_.predict(test_data))

    @staticmethod
    def perform_grid_search_xgboost(train_data, train_labels, test_data, test_labels):
        train_labels = train_labels.reshape(-1, 1)

        learning_rate = [0.01, 0.1, 0.3]
        max_depth = [4, 5, 6]
        n_estimators = [500, 750, 1000]
        gamma = [0, 0.5, 1]
        colsample_bytree = [0, 0.5, 0.8]

        hyper_parameters = dict(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators,
                                gamma=gamma,
                                colsample_bytree=colsample_bytree)

        model = XGBClassifier(min_child_weight=1,
                              objective='binary:logistic',
                              seed=27,
                              use_label_encoder=False)

        grid = GridSearchCV(model, param_grid=hyper_parameters,
                            verbose=True, n_jobs=-1, scoring='accuracy', error_score=0)
        grid_results = grid.fit(train_data, train_labels.ravel())
        """
        LogisticRegression with best Hyper-parameters: 
            Best: 0.172997 using {'learning_rate': 0.01, 'gamma': 1, 'max_depth': 4, 'n_estimators': 500, 
            'colsample_bytree': 0.8}
        """
        print("Best: %f using %s" % (grid_results.best_score_, grid_results.best_params_))

    @staticmethod
    def normalize_data(train_data, test_data):
        # label_encoder = LabelEncoder()
        # train_data = train_data.reshape(-1, 1)
        # test_data = test_data.reshape(-1, 1)
        # train_data = label_encoder.fit_transform(train_data)
        # test_data = label_encoder.fit_transform(test_data)
        sc = StandardScaler()
        train_data = sc.fit_transform(train_data)
        test_data = sc.fit_transform(test_data)
        return train_data, test_data

    @staticmethod
    def encode_data(train_labels, test_labels):
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_labels)
        test_labels = label_encoder.fit_transform(test_labels)
        return train_labels, test_labels

    @staticmethod
    def one_hot_encode(train_labels, test_labels):
        one = OneHotEncoder()
        train_labels = train_labels.reshape(-1, 1)
        test_labels = test_labels.reshape(-1, 1)
        train_labels = one.fit_transform(train_labels).toarray()
        test_labels = one.fit_transform(test_labels).toarray()
        return train_labels, test_labels

    def logistic_regression(self, path):
        X_train, X_test, Y_train, Y_test = self.load_data_pandas(path)
        train_data, train_labels, test_data, test_labels = self.load_data_numpy(X_train, X_test, Y_train, Y_test)

        train_labels, test_labels = self.encode_data(train_labels, test_labels)
        train_labels, test_labels = self.one_hot_encode(train_labels, test_labels)
        train_data, test_data = self.normalize_data(train_data, test_data)
        num_features = len(train_data[0])
        num_classes = len(train_labels[0])
        model = Sequential()
        model.add(tf.keras.layers.Dense(num_classes,
                        activation='softmax',
                        kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                        input_shape=(num_features,)) ) # input dimension = number of features your data has
        model.compile(optimizer='sgd',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(train_data, train_labels, epochs=500, validation_split=0.2, batch_size=512)
        score, acc = model.evaluate(test_data, test_labels)
        print('Logistic Regression loss: {}%'.format(round(score * 100, 2)))
        print('Logistic Regression accuracy: {}%'.format(round(acc * 100, 2)))

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(loss) + 1)


        plt.plot(epochs, loss, 'b-', label='Training loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation loss')
        plt.title('Logistic regression training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        #plt.show()
        plt.savefig("logistic regression loss")
        plt.clf()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        plt.plot(epochs, acc, 'y-', label='Training acc')
        plt.plot(epochs, val_acc, 'g-', label='Validation acc')
        plt.title('Logistic Regression training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("logistic regression accuracy")
        plt.show()
        return history

    def logistic_regression2(self, path):
        X_train, X_test, Y_train, Y_test = self.load_data_pandas(path)
        train_data, train_labels, test_data, test_labels = self.load_data_numpy(X_train, X_test, Y_train, Y_test)

        train_labels, test_labels = self.encode_data(train_labels, test_labels)
        # train_labels, test_labels = self.one_hot_encode(train_labels, test_labels)
        train_data, test_data = self.normalize_data(train_data, test_data)

        # To do grid search, just uncomment below
        # self.perform_grid_search_logistic_regression(train_data, train_labels, test_data, test_labels)

        model = LogisticRegression(solver='newton-cg', penalty='l2', C=1.0)
        model.fit(train_data, train_labels)
        importance = model.coef_[0]
        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
        print(model.score(test_data, test_labels))

    def logistic_regression3(self, path):
        X_train, X_test, Y_train, Y_test = self.load_data_pandas(path)
        train_data, train_labels, test_data, test_labels = self.load_data_numpy(X_train, X_test, Y_train, Y_test)

        train_labels, test_labels = self.encode_data(train_labels, test_labels)
        # train_labels, test_labels = self.one_hot_encode(train_labels, test_labels)
        train_data, test_data = self.normalize_data(train_data, test_data)

        # To do grid search, just uncomment below
        # self.perform_grid_search_logistic_regression(train_data, train_labels, test_data, test_labels)

        model = DecisionTreeClassifier()
        model.fit(train_data, train_labels)
        importance = model.feature_importances_
        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
        print(model.score(test_data, test_labels))


    def xg_boost(self, path):
        X_train, X_test, Y_train, Y_test = self.load_data_pandas(path)
        train_data, train_labels, test_data, test_labels = self.load_data_numpy(X_train, X_test, Y_train, Y_test)

        train_labels, test_labels = self.encode_data(train_labels, test_labels)
        # train_labels, test_labels = self.one_hot_encode(train_labels, test_labels)
        train_data, test_data = self.normalize_data(train_data, test_data)

        # To do grid search, just uncomment below and remove all params from above model
        # self.perform_grid_search_xgboost(train_data, train_labels, test_data, test_labels)
        # RRC BEST BELOW
        model = XGBClassifier(learning_rate=0.1,
                              n_estimators=500,
                              max_depth=5,
                              min_child_weight=1,
                              gamma=0.5,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              objective='binary:logistic',
                              nthread=4,
                              scale_pos_weight=1,
                              seed=27)

        # # MAC BEST BELOW
        # # model = XGBClassifier(learning_rate=0.01,
        # #                       n_estimators=500,
        # #                       max_depth=4,
        # #                       min_child_weight=1,
        # #                       gamma=1,
        # #                       subsample=0.8,
        # #                       colsample_bytree=0.8,
        # #                       objective='binary:logistic',
        # #                       nthread=4,
        # #                       scale_pos_weight=1,
        # #                       seed=27)
        evalset = [(train_data, train_labels), (test_data, test_labels)]
        model.fit(train_data, train_labels, eval_metric = "mlogloss", eval_set=evalset)
        score = model.score(test_data, test_labels)
        print('XGBoost Score: {}%'.format(round(score * 100, 2)))
        importance = model.feature_importances_
        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # # retrieve performance metrics
        # results = model.evals_result()
        # # plot learning curves
        # plt.clf()
        # plt.title('XGBoost training and validation loss')
        # plt.plot(results['validation_0']['mlogloss'], label='train')
        # plt.plot(results['validation_1']['mlogloss'], label='test')
        # # show the legend
        # plt.legend()
        # plt.savefig("XGBoost loss")
        # #plt.show()
        # # show the plot

    def build_model(self, batch_size, nb_epoch):
        model = Sequential()
        model.add(tf.keras.layers.Dense(128, input_shape=(self.num_features,), activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def keras_model(self, path):
        X_train, X_test, Y_train, Y_test = self.load_data_pandas(path)
        train_data, train_labels, test_data, test_labels = self.load_data_numpy(X_train, X_test, Y_train, Y_test)

        train_labels, test_labels = self.encode_data(train_labels, test_labels)
        train_labels, test_labels = self.one_hot_encode(train_labels, test_labels)
        train_data, test_data = self.normalize_data(train_data, test_data)

        self.num_features = len(train_data[0])
        self.num_classes = len(train_labels[0])
        # model = tf.keras.Sequential(tf.keras.layers.Dense(64, activation=tf.nn.relu,
        #                                                   input_shape=(
        #                                                       self.num_features,)))
        # model.add(tf.keras.layers.Dense(32, activation='relu'))
        # # model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
        model = Sequential()
        model.add(tf.keras.layers.Dense(256, input_shape=(self.num_features,), activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

        try:
            # batch_size = [10, 20, 40, 60, 80, 100]
            # epochs = [10, 20]
            # param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
            # k_model = KerasClassifier(build_fn=self.build_model, verbose=0)
            # clf = GridSearchCV(k_model, param_grid=param_grid,
            #                     verbose=True, n_jobs=-1, scoring='accuracy', error_score=0)
            # clf.fit(train_data, train_labels)

            history = model.fit(train_data,
                                train_labels,
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
            print('Fully connected NN Test loss:', score)
            print('Fully connected NN Test accuracy:', acc)
            # if self.save_model:
            #     print("Saving trained model..")
            #     model_file_name = datetime.datetime.now().strftime('./models/model_'
            #                                                        + self.proc_time + '.h5')
            #     model.save(model_file_name)
            #     print("Saved successful!")
            #
            # # Uncomment below to graph model performance
            # d = Display(proc_time=self.proc_time, history=history, save=True)
            # d.display_results()
            # return history
        except ValueError as e:
            logging.exception(e)


