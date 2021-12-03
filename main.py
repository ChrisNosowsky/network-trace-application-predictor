#!/usr/bin/env python3
# ==============================================================================
# CSE 824
# Created on: November 25th, 2021
# Authors: Chris Nosowsky (nosowsky@msu.edu),
#          Hamzeh Alzweri (alzwerih@msu.edu)
#
# Main script to run our application
# ==============================================================================
from sklearn.model_selection import train_test_split
import pandas as pd
from openpyxl import *
from sklearn import svm, preprocessing

def load_data(dataPath):
    print("Loading data..")
    data = pd.read_excel(dataPath)
    Y = data['label']
    X = data.drop('label', axis=1, inplace=False)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=8)
    print("loaded " + str(len(X_train)) + " training examples and " + str(len(X_test)) + " test examples")
    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
  print("Running Cellular Network Trace Application...")
    X_train, X_test, Y_train, Y_test = load_data("./data/Master_LTE_RRC_OTA_Packet_updated.xlsx")
    clf = svm.SVC(kernel='poly')
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)

    clf.fit(X_scaled, Y_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    # accuracy, best I got is 0.336 PogChamp
    print((y_pred == Y_test).sum()/len(Y_test))