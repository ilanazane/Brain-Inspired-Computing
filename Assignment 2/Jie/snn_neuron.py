import numpy as np
from hyperparams import *
class LIFFCLayer(object):
    def __init__(self, n_input, n_neuron, \
                delta_t = 1.0, resistance = 10.0, conductance = 2.0,refact_time = 2.0,\
                v_threshold = 0.75, v_resting = 0.0, spike_magnitude = 1.0, input_decay = 0.7):
        """
        @input
        - n_input: number of input signals for each neuron
        - n_neuron: number of neurons in this layer
        - delta_t: time step controls the interaction speed (fixed unless simulation requires)
        - resistance, conductance: LIF parameters
        - refact_time: neuron keep resting for multiple steps after spike
        - v_threshold: firing threshold potential
        - v_resting: resting potential
        - spike_magnitude: the magnitude of a spike
        - input_decay: mapped input current will leave a trace over time
        """
        super(LIFFCLayer, self).__init__()
        # weights, or spike input efficacies; normalized so that sum of square of single neuron weights is 1
        W = np.random.uniform(0.0, 1.0, (n_input, n_neuron)) / n_input

 
        scale = np.sqrt(np.sum(W ** 2, axis = 0))
        self.synapses = W / scale
        self.synapses = W

        # Simulation config (may not all be needed!!)
        # time step size
        self.dt = delta_t
        # Current neuron potential (mV)
        self.Vm = np.zeros(n_neuron)
        # Refectory period
        self.tau_ref  = refact_time
        # Resistance
        self.Rm = resistance
        # Conductance
        self.Cm = conductance
        # refactory period, originally 2
        self.tRef = refact_time
        self.refCtDwn = np.zeros(n_neuron)
        # spike threshold, originally 0.75/1
        self.Vth = v_threshold
        # resting potential
        self.Vr = v_resting
        # spike magnitude
        self.Vspike = spike_magnitude
        # previous input
        self.inVec = np.zeros(n_neuron)
        # input decay
        self.inDecay = input_decay
 



    def forward(self, in_train):
        """
        @input
        - intrain: several time steps of input signals from all input neurons of previous layer
        @output
        - outTrain: spike subtrain of all embedding neurons in this layer
        """
        outTrain = np.zeros((self.synapses.shape[-1], in_train.shape[-1]))
        vHistory = np.zeros_like(outTrain)
        # forward through time
        for i in range(in_train.shape[-1]):
            spikes, potentials = self.forward_step(in_train[:,i])
            vHistory[:,i] = potentials
            outTrain[:,i] = spikes
        return outTrain, vHistory
    
    def forward_step(self, features,check):
        """
        forward step of a single time step
        
        """

        inputVec = np.matmul(features,self.synapses)


        # else:
        #     inputVec = features[0]*self.synapses[0]+features[1]*self.synapses[1]

        self.inVec = inputVec
        # leaky integrate and fire
        return self.do_LIF(inputVec)

    def do_LIF(self, I):
        """
        One step of LIF update
        """         
        
 #        print("neuron potential: " + str(self.Vm))
#        neurons that just fired
        firedNeurons = self.Vm > self.Vth
        # newly fired neurons will go back to resting potental and start refactory period
        self.Vm[firedNeurons] = self.Vr
        self.refCtDwn[firedNeurons] = self.tRef
        # neurons that recently fired and still within the count down (refactory period)
        alreadyFiredNeurons = self.refCtDwn > 0
        # difference in neuron potential (LIF)
        diff = (I - (self.Vm - self.Vr)/self.Rm)/self.Cm
        # update neurons that is not in refactory period
        self.Vm = self.Vm + diff * (1-alreadyFiredNeurons) * self.dt
        self.refCtDwn[alreadyFiredNeurons] -= 1


        # return the newly fired neurons
        return firedNeurons * self.Vspike, self.Vm


