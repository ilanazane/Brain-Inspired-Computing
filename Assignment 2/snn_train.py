#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:37:31 2020

@author: jiewang
"""
from snn_net import SNN_MLP
from snn_optim import *
from snn_utils import *

aor2Loader = Bool2SpikeLoader(spiketrain_0, spiketrain_1, intervalLength, batch_size = 4)
aor2Loader.generate_data(n_features = 2, true_fun = AND)
# setup SNN with Hebbian optimizer
NetA = SNN_MLP(struct = [2,1], optimizer_cls = HebbianOptim, params = {\
            "learn_rate": 0.01, "normalize": True, "mean_pre": 0, "mean_post": 0.1})
# training
NetA.fit(data_loader = aor2Loader, params = {"epoch": 200, "epsilon": 1e-3})
# testing
#xor2Loader.get_n_batch(0)
target,testData = aor2Loader.get_batch(encoding = "firing_rate")
predictions = NetA.predict(testData[:,:-1,:])  # no label
predictions = time_to_rate(predictions)
print("Test output firing rate: " + str(predictions))
print("Target boolean output: " + str(target[:,-1]))