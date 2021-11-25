#!/usr/bin/env python3
# ==============================================================================
# CSE 824
# Created on: November 25th, 2021
# Authors: Chris Nosowsky (nosowsky@msu.edu),
#          Hamzeh Alzweri (alzwerih@msu.edu)
#
# Predict class to handle predictions based on input data from cellular traces
# ==============================================================================

import numpy as np


class Predict:
    def __init__(self):
        pass

    def predict_results(self, x, model):
        print("Predicting Results...")
        prediction = model.predict(np.array(x))
