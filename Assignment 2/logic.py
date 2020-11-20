#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:51:39 2020

@author: jiewang
"""


import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import animation

class LogicGate():
    def __init__(self):
        self.start_time = 300
        self.time_step = 0.5
    
        self.y=np.arange(0,1000.1,self.time_step)
        
    #function that injects current at certain time
    def start(self,start_time,start_current):
        x=np.zeros((len(self.y)))                #create empty array to plot current
    
        for i in range (0,len(self.y)):
            if self.y[i] > start_time:            #when we reach start time, start plotting input current
                x[i]=start_current          #each second afterwards will hold the constant value of the input current
        return x
    
    
    # # Calculate Differential Equations
    
    # In[46]:
    
    
    #these equations are not linear so they need to be approximated using Euler approximation
    #we rewrite the derivatives to show the change in voltage over change in time between two different timesteps
    def equations(self,current,u,v,a,b):
        v = v+self.time_step*(0.04*v**2+5*v+140-u+current)
        u = u+self.time_step*(a*(b*v-u))
    
        return u,v
    
    
    # # Create the Izhikevich Model
    
    # In[47]:
    

    def model(self,a,b,c,d,start_current):
        v = -65*np.ones((len(self.y)))         #creates the baseline array that holds the resting membrane potential
        u = np.zeros((len(self.y)))            #initialize values for membrane recovery rate
        u[0] = b*-65                      #initial value for u
        spikeTrain = np.zeros_like(v)
        
        current = self.start(self.start_time,start_current)
        for i in range(0,len(self.y)-1):
            u[i+1],v[i+1]  = self.equations(current[i],u[i],v[i],a,b)
             #after the spike reaches apex of 30mV, membrane voltage and recovery are reset
            if v[i+1] > 30:
                v[i+1] = c
                u[i+1] = u[i+1]+d
                spikeTrain[i+1] = 1
                
    
        #self.plotting(current,v,a,b,c,d,self.y,start_current)
        return v,spikeTrain
    # # Plotting Current as a Function of Time and Membrane Potential
    
    # In[48]:
    
    
    def plotting(current,v,a,b,c,d,y,start_current):
        #PLOT OUTPUT CURRENT
        fig, ax1 = plt.subplots(figsize=(12,7))
        ax1.plot(y, v, label = 'Output')
        ax1.set_xlabel('Time in ms')
        ax1.set_ylabel('Output in mV', color='k')
        ax1.tick_params('y', colors='k')
        ax1.set_ylim(-95,40)
        ax1.set_title('Izhikevich Model with Input Current of %s A ' %(start_current))
        plt.show()
    

    
if __name__ == '__main__':
    xx = 1
    yy = 1
    W_x = np.random.uniform(0.0, 1.0, (1, 2)) / 2 #(1,2)
    W_y = np.random.uniform(0.0, 1.0, (1, 2)) / 2 #(1,2)
 
    
    if xx == yy== 1:
       x_current = y_current = 10

    if xx==yy==0:
       x_current = y_current = 5
    
    if xx != yy:

        if xx == 0:
            x_current = 5
            y_current = 10
        else:
            x_current = 10
            y_current = 5


    LG = LogicGate()
    x_activity,x_spiketrain= LG.model(a=0.02,b=0.2,c=-65,d=8,start_current=x_current)
    y_activity,y_spiketrain = LG.model(a=0.02,b=0.2,c=-65,d=8,start_current=y_current)
    

    



    
    