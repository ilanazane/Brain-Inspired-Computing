#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import animation


# # Create Input Space and Starting Parameters



start_time=300                         #starting point in ms
start_current=10                       #input current in mv
time_step=0.5                          #time increments


y=np.arange(0,1000.1,time_step)

#function that injects current at certain time
def start(start_time,start_current):
    x=np.zeros((len(y)))                #create empty array to plot current

    for i in range (0,len(y)):
        if y[i] >start_time:            #when we reach start time, start plotting input current
            x[i]=start_current          #each second afterwards will hold the constant value of the input current
    return x


# # Calculate Differential Equations



#these equations are not linear so they need to be approximated using Euler approximation
#we rewrite the derivatives to show the change in voltage over change in time between two different timesteps
def equations(current,u,v,a,b):
    v = v+time_step*(0.04*v**2+5*v+140-u+current)
    u = u+time_step*(a*(b*v-u))

    return u,v


# # Create the Izhikevich Model



def model(a,b,c,d):
    v = -65*np.ones((len(y)))         #creates the baseline array that holds the resting membrane potential
    u = np.zeros((len(y)))            #initialize values for membrane recovery rate
    u[0] = b*-65                      #initial value for u

    current = start(start_time,start_current)
    for i in range(0,len(y)-1):
        u[i+1],v[i+1]  = equations(current[i],u[i],v[i],a,b)
         #after the spike reaches apex of 30mV, membrane voltage and recovery are reset
        if v[i+1] > 30:
            v[i+1] = c
            u[i+1] = u[i+1]+d

    plotting(current,v,a,b,c,d,y)


# # Plotting Current as a Function of Time and Membrane Potential




def plotting(current,v,a,b,c,d,y):
    #PLOT OUTPUT CURRENT
    fig, ax1 = plt.subplots(figsize=(12,7))
    ax1.plot(y, v, 'b',label = 'Output')
    ax1.set_xlabel('Time in ms')
    ax1.set_ylabel('Output (mV)', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_ylim(-95,40)
    ax1.set_title('Izhikevich Model with Input Current of %s mV ' %(start_current),fontsize=20)


    ax2 = ax1.twinx()
    ax2.plot(y, current, 'r', label = 'Input Current')
    ax2.set_ylim(0,start_current*20)
    ax2.set_ylabel('Input (mV)', color='r')
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    ax1.legend(loc=1)
    ax2.legend(loc=3)



    plt.show()




model(0.02,0.2,-65,8)
