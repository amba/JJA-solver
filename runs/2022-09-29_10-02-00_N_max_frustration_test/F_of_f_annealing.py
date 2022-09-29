#!/usr/bin/env python3
import sys
import os
sys.path.append(os.getcwd() + '/..')
from JJAsolver.network import network, jj_cpr_ballistic, jj_free_energy_ballistic
from JJAsolver.data import datafile

import copy
import numpy as np
import matplotlib.pyplot as plt
import time

Nx = 16
Ny = 16

data = datafile(
    params = ['Nx', 'Ny', 'N_max', 'frustration', 'free_energy', 'current'],
    folder = 'N_max_frustration_test'
)

def cpr_x(gamma):
    return np.sin(gamma)# + sin2_term * np.sin(2*gamma) + cos_term*np.cos(gamma)
#    return jj_cpr_ballistic(gamma, tau)

                                            
def cpr_y(gamma):
    return np.sin(gamma)# + sin2_term * np.sin(2*gamma)
   # return jj_cpr_ballistic(gamma, tau)

def f_x(gamma):
    return 1 -np.cos(gamma)# - 0.5 * sin2_term * np.cos(2 * gamma) +\
        #        cos_term * np.sin(gamma)
    #return jj_free_energy_ballistic(gamma, tau)

def f_y(gamma):
    return 1 -np.cos(gamma)# - 0.5 * sin2_term * np.cos(2 * gamma)
    #return jj_free_energy_ballistic(gamma, tau)



n = network(
    Nx, Ny,
    cpr_x=cpr_x,
    cpr_y=cpr_y,
    free_energy_x = f_x,
    free_energy_y = f_y,
)


frustration_vals = np.linspace(0, 0.5, 101)
N_max_vals = (100, 500, 1000, 5000, 10000)

T_start = 0.35
t0 = time.time()
for N_max in N_max_vals:
    for f in frustration_vals:
        n.reset_network()
        n.set_frustration(f)
        for i in range(int(N_max) + 200):
            temp = max(T_start * (N_max - i) / N_max, 0)
            delta =  n.optimization_step(temp = temp, optimize_leads=True)
            print("T = ", temp)
            I = n.get_current()
            print("I = ", I)
            print("f = %g, i = %d, delta = %g" % (f, i, delta))        
            if abs(delta) < 1e-2:
                break
        t1 = time.time() - t0
        F = n.free_energy()
        I = n.get_current()
        data.log(
            {'Nx': Nx,
            'Ny': Ny,
            'N_max': N_max,
            'frustration': f,
            'free_energy': F,
            'current': I,}
            )
    data.new_block()
            
