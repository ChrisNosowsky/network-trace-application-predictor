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

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(loss) + 1)

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if self.save:
            fig_file_name = datetime.datetime.now().strftime('figs/loss_' + self.proc_time + '.png')
            plt.savefig(fig_file_name)
        plt.show()

        plt.clf()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        if self.save:
            fig_file_name = datetime.datetime.now().strftime('./figs/accuracy_' + self.proc_time + '.png')
            plt.savefig(fig_file_name)
        plt.show()
