#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 00:05:44 2020

@author: jiewang
"""

import numpy as np
#import matplotlib.pyplot as plt
import pylab as plt
from scipy.integrate import odeint

class HHMODEL():
    """Full Hodgkin-Huxley Model implemented in Python"""
    def __init__(self,C,g_Na,g_K,g_L,V_Na,V_K,V_L):
        self.C  =   C
        """membrane capacitance, in uF/cm^, [0.5, 1], 1 is mostly used"""

        self.g_Na = g_Na
        """Sodium (Na) maximum conductances, in mS/cm^2, [0, 500], desired value = 120"""

        self.g_K  =  g_K
        """Postassium (K) maximum conductances, in mS/cm^2, [0, 300], desired value = 36"""

        self.g_L  =  g_L
        """Leak maximum conductances, in mS/cm^2, about 0.3 from H-H paper"""

        self.V_Na =  V_Na
        """ Equilibrium potentials for the sodium, 60"""

        self.V_K  = V_K
        """ Equilibrium potentials for the potassium, -88"""
        self.V_L  = V_L
        """ Leaky Equilibrium potentials, -61"""

        self.t = np.arange(0.0, 100.0, 0.01)
        """ The time to integrate over """

        
    def a_mV(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40)/(1-np.exp(-(V+40)/10))

    def b_mV(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4*np.exp(-(V+65)/18)

    def a_hV(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*np.exp(-(V+65)/20)

    def b_hV(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1/(1+np.exp(-(V+35)/10))

    def a_nV(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55)/(1-np.exp(-(V+55)/10))

    def b_nV(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*np.exp(-(V+65)/80)

    def num_Na(self, V, m, h):
        return self.g_Na*(m**3)*h*(V - self.V_Na)

    def num_K(self, V, n):
        return self.g_K*(n**4)*(V - self.V_K)
    
    #  Leak
    def num_L(self, V):
        return self.g_L*(V - self.V_L)

    def Inj_current(self, t):
        if t < 10 or (t>=30 and t<50) or (t>=80):
            I = 0
        elif t>=10 and t<30:
            I = 10
        elif t>= 50 and t<80:
            I = 20
        

        return I

    @staticmethod
    def diffEquations(solved_Eqs, t,self):
   
        V, m, h, n = solved_Eqs

        dVdt = (self.Inj_current(t)- self.num_Na(V, m, h) - self.num_K(V, n) - self.num_L(V)) / self.C
        dmdt = self.a_mV(V)*(1.0-m) - self.b_mV(V)*m
        dhdt = self.a_hV(V)*(1.0-h) - self.b_hV(V)*h
        dndt = self.a_nV(V)*(1.0-n) - self.b_nV(V)*n
        return dVdt, dmdt, dhdt, dndt

    def HHProcess(self):


        solved_Eqs= odeint(self.diffEquations, [-65, 0, 0, 0], self.t, args=(self,))
        V = solved_Eqs[:,0]
        m = solved_Eqs[:,1]
        h = solved_Eqs[:,2]
        n = solved_Eqs[:,3]

        
        plt.figure()
        ax1 = plt.subplot(2,1,1)
        plt.title('Hodgkin-Huxley Neuron')
        plt.plot(self.t, V, 'k')
        plt.ylabel('V (mV)')

#
        plt.subplot(2,1,2, sharex = ax1)
        i_inj_values = [self.Inj_current(t) for t in self.t]
        plt.plot(self.t, i_inj_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-1, 50)

        plt.tight_layout()
        plt.show()
        
        plt.figure()
        plt.plot(self.t, m, 'r', label='m')
        plt.plot(self.t, h, 'g', label='h')
        plt.plot(self.t, n, 'b', label='n')
        plt.ylabel('Gating Value')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    HH = HHMODEL(1.0,120,36,0.3,60,-88,-61)
    HH.HHProcess()
