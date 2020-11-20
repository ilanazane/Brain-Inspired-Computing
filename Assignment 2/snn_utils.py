#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:25:23 2020

@author: jiewang
"""

import numpy as np
from hyperparams import *
import time

# Functions

def time_to_rate(sTrains):
    """
    Calculate spike train firing rate
    """

    return np.sum(sTrains, axis = -1) / (sTrains.shape[-1])

def time_to_bool(sTrains, threshold = rateT):
    """
    Calculate firing rate and convert to boolean variables
    """
    rates = time_to_rate(sTrains)
    results = np.zeros_like(rates)

    results[rates > threshold] = 1


    return results

# Boolean functions
def AND(features):
    # all 1 return 1
    if np.sum(features) == len(features):
        return 1
    else:
        return 0
def OR(features):
    # all 0 then 0
    if np.sum(features) == 0:
        return 0
    else:
        return 1


# Data loader
class Bool2SpikeLoader(object):
    """
    Spike train generator and data loader for boolean functions
    """
    def __init__(self, spiketrain_0, spiketrain_1, intervalLength, batch_size = 1):
        # average spike interval for value 0
        self.spiketrain_0 = spiketrain_0
        # average spike interval for value 1
        self.spiketrain_1 = spiketrain_1
        # time span of each spike train
        self.trainLength = intervalLength
        # the generated data
        self.data = None
        # current batch position
        self.currentBatch = 0
        # batch size
        self.batchSize = batch_size
    def generate_zero_train(self):
        """
        Generate spike train for value 0
        """
        
        return self.spiketrain_0
    def generate_one_train(self):
        """
        Generate spike train for value 1
        """

        return self.spiketrain_1

    def generate_data(self, n_features, true_fun, encoding = "firing rate"):
        """
        Generate training data based on a true boolean function
        """
        dataSize = 100
        #dataSize = np.power(2, n_features)
        # generate data samples with binary encoding of a number
        self.data = np.zeros((dataSize, n_features + 1))
        for i in range(dataSize):
            row = np.zeros(n_features + 1)
            for j in range(n_features):
                row[j] = np.random.choice(2,1)

            row[n_features] = true_fun(row[:-1])
            self.data[i,:] = row

        return self.data
    # def get_n_batch(self, batch_size = -1):
    #     """
    #     Set up batch size and calculate the number of batches
    #     """
    #     # batch size choices: 0 - entire dataset; <0 - default batch size; >0 specified batch size
    #     if batch_size == 0:
    #         self.batchSize = len(self.data)
    #     elif batch_size > 0:
    #         self.batchSize = batch_size
    #     self.currentBatch = 0
    #     return int(np.ceil(len(self.data) / self.batchSize))
    def get_batch(self, encoding = "firing_rate"):
        """
        Get batch data
        """
        # Restart iteration if reaching the end of the dataset
        
        if self.currentBatch >= len(self.data):
            self.currentBatch = 0
        self.currentBatch = self.currentBatch + self.batchSize
        # raw data with boolean variable
        batchData = self.data[self.currentBatch - self.batchSize: min(self.currentBatch, len(self.data))]



        # return spike trains with firing rate encoding
        if encoding == "firing_rate":
            # each batch data is 3D: number of sample, number of features, and length of spike train of each feature
            batchSpikeData = np.zeros((batchData.shape[0], batchData.shape[1], self.trainLength))
            for i in range(batchData.shape[0]):
                for j in range(batchData.shape[1]):
                    if batchData[i,j] == 1:
                        batchSpikeData[i,j,:] = self.generate_one_train()
                    else:
                        batchSpikeData[i,j,:] = self.generate_zero_train()

            return batchData,batchSpikeData

        else:
            return None