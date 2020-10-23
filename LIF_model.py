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
Vm = np.zeros(len(total_simulation));
dv = float(input("Change in voltage:")); #simmilar to dt but for voltage not time 
currArr = np.zeros(len(total_simulation));

#Our original equation is given by tau_m(dv/dt) = -Vm(t) + Rm(I(t)) and we want to find Vm
#Let's use this equation to solve for each Vm at time t
currentS = float(input("Where does your current start:"))
stop = float(input("Where does your current end:"));

for j in range(int(currentS / dt), int(stop / dt)):
    currArr[j] = I

for i in range(len(total_simulation)):
    if total_simulation[i] > tRef:
        Vm[i] = Vm[i - 1] + (-Vm[i - 1] + currArr[i-1]*Rm)/tau_m*dt;
        if Vm[i] >= Vt:
            Vm[i] += dv;
            tRef = total_simulation[i] + tau_ref;




#Plot points 
fig, ax1 = plt.subplots();

ax2 = ax1.twinx();
ax1.plot(total_simulation,Vm,'g--');
ax2.plot(total_simulation,currArr,'r-');

ax1.set_xlabel('Time');
ax1.set_ylabel('Membrane Potential(mV)', color='g');
ax2.set_ylabel("Current", color='r');
ax2.set_ylim([0,I+2]);

plt.title("LIF MODEL");

plt.show()
