import numpy as np


class HebbianOptim():
    """
    Standard Hebb Rule:
    - decay = 0, mean_pre = 0, mean_post = 0
    Hebb with decay:
    - 0 < decay < 1, mean_pre = mean_post = 0
    Hebb with post synaptic threshold: (default)
    - decay = 0, mean_pre = 0, 0 < mean_post < 1
    Covariance rule:
    - decay = 0, 0 < mean_pre < 1, 0 < mean_post < 1
    """
    def __init__(self, weights, lr = 0.1, normalize = False, decay = 0.0, mean_pre = 0., mean_post = 0.):
        super(HebbianOptim, self).__init__()
        # learning rate of each weight
        self.W = weights
        self.normalize = normalize
        self.preSynTrains = []
        self.postSynTrains = []
        self.A = lr / weights.shape[0]
        # decay factor
        self.decay = np.mean(self.A) * decay
        # average pre-synaptic firing rate
        self.preHist = []
        self.meanPre = mean_pre * np.ones(weights.shape[0])
        # average post-synaptic firing rate
        self.meanPost = mean_post * np.ones(weights.shape[1])
    def __call__(self, diff):
        w = self.W + diff
        if self.normalize:
            scale = np.sqrt(np.sum(w ** 2, axis = 0))
            self.W = w / scale
        else:
            self.W = w
        return self.W
    def record(self, updates, replace = False):
#         print("Optimizer record: " + str(updates))
#         input()

        if replace:
            if "pre" in updates:
                self.preSynTrains[-1] = updates["pre"]
            if "post" in updates:
                self.postSynTrains[-1] = updates["post"]
        else:
            if "pre" in updates:
                self.preSynTrains.append(updates["pre"])
            if "post" in updates:
                self.postSynTrains.append(updates["post"])
        
    def dw(self, pres, posts):
        """
        Hebbian learning rule
        """
        dvPre = pres
        dvPost = posts - self.meanPost

        return np.outer(dvPre, dvPost) * self.A 
    def step(self):
        # pre synaptic firing rate
        vPre = self.preSynTrains[-1]
        # post synaptic firing rate
        vPost = self.postSynTrains[-1]
        # if len(np.where(vPre>0)[0])>0:
        #     print(vPre)
        #     print(vPost)
        #     input("Enter")

        # calculate udpate
        diff = self.dw(pres = vPre, posts = vPost)
        # if len(np.where(vPre>0)[0])>0:
        #     print('Diff: ', diff)
        #     input("Enter")
        self.W = self.__call__(diff)

        #self.meanPre = 0.05 * vPre + 0.95 * self.meanPre
        self.meanPost = 0.05 * vPost + 0.95 * self.meanPost


