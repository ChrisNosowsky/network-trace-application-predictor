#!/usr/bin/env python3
# ==============================================================================
# CSE 824
# Created on: November 25th, 2021
# Authors: Chris Nosowsky (nosowsky@msu.edu),
#          Hamzeh Alzweri (alzwerih@msu.edu)
#
# Displays our accuracy and loss metrics
# ==============================================================================

import matplotlib.pyplot as plt
import datetime


class Display:
    def __init__(self, proc_time, history, save=False):
        self.proc_time = proc_time
        self.history = history
        self.save = save

    def display_results(self):
        """
        Displays prediction results in a chart

            Parameters:
                    history (Any): History from the model
        """

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(loss) + 1)
        plt.clf()
        plt.plot(epochs, loss, 'b-', label='Training loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation loss')
        plt.title('Fully connected Neural Network training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if self.save:
            fig_file_name = datetime.datetime.now().strftime('figs/loss_' + self.proc_time + '.png')
            plt.savefig(fig_file_name)
        #plt.show()

        plt.clf()

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        plt.plot(epochs, acc, 'b-', label='Training acc')
        plt.plot(epochs, val_acc, 'r-', label='Validation acc')
        plt.title('Fully connected Neural Network training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        if self.save:
            fig_file_name = datetime.datetime.now().strftime('./figs/accuracy_' + self.proc_time + '.png')
            plt.savefig(fig_file_name)
        #plt.show()
