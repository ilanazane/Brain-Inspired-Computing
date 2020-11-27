import numpy as np
from snn_neuron import LIFFCLayer
from snn_optim import HebbianOptim
from hyperparams import *
from snn_utils import *
import matplotlib.pyplot as plot

class SNN_MLP(object):
    def __init__(self, struct, optimizer_cls = HebbianOptim, params = {\
            "learn_rate": 1.0, "normalize": True,\
            "mean_pre": 0.1, "mean_post": 0.1}):
        assert len(struct) >= 2
        super(SNN_MLP, self).__init__()
        self.layers = []
        self.optimizers = []
        # multiple LIF FC layers
        for i in range(len(struct) - 1):
            layer = LIFFCLayer(struct[i], struct[i+1])
            self.layers.append(layer)
            self.optimizers.append(optimizer_cls(layer.synapses, \
                lr = params["learn_rate"], normalize = params["normalize"],\
                mean_pre = params["mean_pre"], mean_post = params["mean_post"]))
#         self.finalLayer = LIFFCLayer(struct[-2], struct[-1])
#         self.optimizer.append(optimizer_cls(struct[-2] + 1, struct[-1]))
    def forward(self, input_trains):
        """
        forward pass of input spike trains
        """
#         print(input_trains)
#         print("input trains: " + str(input_trains.shape))
#         input()
        T = input_trains.shape[-1]
        outputTrains = np.zeros((self.layers[-1].synapses.shape[-1],T))

        for t in range(T):
            vIn = input_trains[:,t]
            vOut = vIn
            for i in range(len(self.layers)):
                layer = self.layers[i]
#                 print(vOut)
#                 input()
                vOut, _ = layer.forward_step(vOut,check=0)
#                 print(vOut)
#                 input()
            outputTrains[:,t] = vOut
        return outputTrains
    def optimize_supervised(self, features, target_train):
        """
        learning with target_train
        """
        L = len(self.layers)  ## 1个layer
        trained = np.zeros_like(features)
        T = features.shape[-1]

        if len(np.where(features[0]>0)[0]) != len(np.where(features[1]>0)[0]):
            check = 0
        else:
            check = 1


        for t in range(T):
            # forward pass
            vIn = features[:,t]
            targets = target_train[:,t]

            
            for i in range(L):
                # input and output spike
                vOut, _ = self.layers[i].forward_step(vIn,check)

                # weight updates
                optimizer = self.optimizers[i]
                optimizer.record({"pre": vIn, "post": vOut})

                optimizer.record({"post": targets}, replace = True) ### actual target becomes post activity.

                # optimize
                optimizer.step()
                self.layers[L-1-i].synapses = optimizer.W

    
 
 
       
    def fit(self, data_loader, params):
        """
        training the network with data
        """
        # maximum number of epochs
        epochLimit = params["epoch"]
        # error threshold for early stopping
        epsilon = params["epsilon"]
        # training
        error = np.float("inf")
        for epoch in range(epochLimit):
            rawData,encodedData = data_loader.get_batch(encoding = "firing_rate")
            for i in range(encodedData.shape[0]):
                features = encodedData[i,:-1,:]
                targets = encodedData[i,-1:,:]
                # outplot = np.where(features[i,:]>0)[0]
                # outplot = outplot.astype(float)
                # for q in range(len(outplot)):
                #     outplot[q] = (outplot[q]+1)/1000
                # #outplot = outplot.astype(float)
                # plot.eventplot(outplot,orientation = "horizontal",linelengths = 0.1, lineoffsets = 0.5)
                # plot.ylim((0,1.0))
                # plot.xlim((0, 1.0))
                # if len(outplot)<20:
                #     plot.title("spike train for input '0' over 1000 ms")
                # else:
                #     plot.title("spike train for input '1' over 1000 ms")
                # plot.show()
                # input("enter")

                self.optimize_supervised(features, targets)  ### weight update 
            
    
    
            #data_loader.get_n_batch(0)
            #rawData = data_loader.get_batch(encoding = "raw") #column1: x, column2: y, column3: label
    
            #encodedData = data_loader.get_batch(encoding = "firing_rate")
            preds = np.zeros((rawData.shape[0],1)) #(4,1)
            rates = np.zeros((rawData.shape[0],1)) #(4,1)
    
            for i in range(rawData.shape[0]):
    
                outSpikes = self.forward(encodedData[i,:-1,:])

                rates[i] = time_to_rate(outSpikes)
                preds[i] = time_to_bool(outSpikes)
        
            error = np.sqrt(np.sum((preds - rawData[:,-1]) ** 2))
            if epoch % 10 == 0:
                print("epoch: " + str(epoch))
                print("\tWeights: " + str([self.layers[i].synapses for i in range(len(self.layers))]))
                print("\traw inputs: " + str(rawData[:,:-1]))
                print("\tTarget boolean value: " + str(rawData[:,-1]))
                print("\tTarget firing rate: " + str([time_to_rate(encodedData[i,-1,:]) for i in range(encodedData.shape[0])]))
                print("\tPredicted firing rate: " + str(rates.reshape(-1)))
                print("\tPredicted boolean value: " + str(preds.reshape(-1)))
                #print("\tError: "+str(error))


            # if error < epsilon:
            #     print("Early stop at epoch: " + str(epoch))
            #     break
    def predict(self, features):
        outputs = np.zeros((features.shape[0], self.layers[-1].synapses.shape[-1], features.shape[-1]))
        for i in range(features.shape[0]):
            outputs[i,:,:] = self.forward(features[i,:,:])
            print("\tWeights: " + str([self.layers[i].synapses for i in range(len(self.layers))]))
        return outputs


