import numpy as np
import matplotlib.pyplot as plt

#The user will define the amnt of time of neuron will run
time = float(input("Time:"));
time = abs(time);
tRef = 0; #initial refractory time
# Let user define LIF properties 
Cm = float(input("Capacitance:")); #Define capacitance
I = float(input("What is your current:")); #Define input current 
Rm = float(input("Resistance value:")); #Define resistance 
tau_m = Cm*Rm; #This is our time constant 
tau_ref = float(input("How long will the refractory period be:")); #refractory period 
dt = float(input("What is the time step:")); #simulation time step 
Vt = float(input("Spike Threshold:")); #spike threshold 
total_simulation = np.arange(0, time+dt,dt); #is an array of time 
Vm = [0] * len(total_simulation);
dv = float(input("Change in voltage:")); #simmilar to dt but for voltage not time 

#Our original equation is given by tau_m(dv/dt) = -Vm(t) + Rm(I(t)) and we want to find Vm
#Let's use this equation to solve for each Vm at time t
for i in range(len(total_simulation)):
    if time > tRef:
        Vm[i] = Vm[i - 1] + (-Vm[i - 1] + I*Rm)/tau_m*dt;
        if Vm[i] >= dv:
            Vm[i] += dv;
            tRef = time + tau_m;

#Plot points 
np.array(total_simulation);
np.array(Vm);
plt.plot(total_simulation,Vm);
plt.xlabel("Time");
plt.ylabel("Membrane Potential");
plt.title("LIF MODEL");
plt.show();
