from logic import *

LG = LogicGate()
_,zero_count= LG.model(a=0.02,b=0.2,c=-65,d=8,start_current=10)
_,one_count = LG.model(a=0.02,b=0.2,c=-65,d=8,start_current=30)


zero_count = np.where(zero_count>0)[0]//2
one_count = np.where(one_count>0)[0]//2


spiketrain_0 = np.zeros((1000,))
spiketrain_1 = np.zeros((1000,))

for i in zero_count:
    spiketrain_0[i] = 1

for i in one_count:        
    spiketrain_1[i] = 1

intervalLength = 1000

rateT = 0.02 # firing rate threshold, considered as 1 if firing rate > 0.1, 0 otherwise
learnRate = 0.01
