from PIL import Image
import os
import sys
import cv2
import numpy as np
from numpy import*
import matplotlib.pyplot as plt
import nengo
from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform

#Test 1 nengo --> take 1 image from resized images and pass it into neuron as 2D array

img = r"C:\Users\User\Desktop\resizedDir2\small_i05june05_static_street_boston_p1010907.jpeg"

temp=asarray(Image.open(img))
x=temp.shape[0]
y=temp.shape[1]*temp.shape[2]

temp.resize((x,y)) # a 2D array
#print(temp)
#Flatten image for nengo
temp = temp.flatten();


#nengo 
model = nengo.Network(label = "Test input")
with model:
    neuron = nengo.Ensemble(3072,
        dimensions=1,  # Represent a scalar
        # Set intercept to 0.5
        #intercepts=Uniform(-0.5, -0.5),
        # Set the maximum firing rate of the neuron to 100hz
        #max_rates=Uniform(100, 100),
        # Set the neuron's firing rate to increase for positive input
        #encoders=[[1]],
        )
    #input the 2d array
    input_ = nengo.Node(temp)
    nengo.Connection(input_, neuron.neurons)
    # The original input
    input_probe = nengo.Probe(input_)
    # The raw spikes from the neuron
    spikes = nengo.Probe(neuron.neurons)
    # Subthreshold soma voltage of the neuron
    voltage = nengo.Probe(neuron.neurons, "voltage")
    # Spikes filtered by a 10ms post-synaptic filter
    filtered = nengo.Probe(neuron, synapse=0.01)

#run neuron simmulator
with nengo.Simulator(model) as sim:  # Create the simulator
    sim.run(1)  # Run it for 1 second


# Plot the decoded output of the ensemble
plt.figure()
plt.plot(sim.trange(), sim.data[filtered])
plt.plot(sim.trange(), sim.data[input_probe])
plt.xlim(0, 1)

# Plot the spiking output of the ensemble
plt.figure(figsize=(10, 8))
plt.subplot(221)
rasterplot(sim.trange(), sim.data[spikes])
plt.ylabel("Neuron")
plt.xlim(0, 1)

# Plot the soma voltages of the neurons
plt.subplot(222)
plt.plot(sim.trange(), sim.data[voltage][:, 0], "r")
plt.xlim(0, 1)

plt.show();


#Tutorial:https://www.nengo.ai/nengo/examples/basic/single-neuron.html