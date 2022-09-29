#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd() + '/..')
from JJAsolver.network import network, jj_cpr_ballistic, jj_free_energy_ballistic
from JJAsolver.data import datafile, datafolder

import copy
import numpy as np
import matplotlib.pyplot as plt
import time

Nx = 16
Ny = 16

folder = datafolder('L0_of_I')


data_L0 = datafile(folder,
    file = 'data_L0.dat',
    params = ['Nx', 'Ny', 'frustration', 'free_energy', 'L0'],
)

data_L_of_I = datafile(folder,
    file = 'data_L_of_I.dat',
    params = ['Nx', 'Ny', 'frustration', 'I', 'free_energy'],
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


frustration_vals = np.linspace(0, 0.5, 11)
d_phi = 0.001

for f in frustration_vals:
    n.reset_network()
    n.set_frustration(f)
    
    n = n.find_ground_state()
    print("f = ", f)
    print("free energy = ", n.free_energy())
    
    I_vals = [n.get_current(),]
    F_vals = [n.free_energy(),]
    for sign in (+1, -1):
        m = copy.deepcopy(n)
        for i in range(20):
            m.optimize(optimize_leads=False, delta_tol=1e-3)
            I = m.get_current()
            print("I = ", I)
            F = m.free_energy()
            print("F = ", F)
            data_L_of_I.log(
                {'Nx': Nx,
                 'Ny': Ny,
                 'frustration': f,
                 'I': I,
                 'free_energy': F
                 }
            )
            I_vals.append(I)
            F_vals.append(F)
            m.add_phase_gradient(sign * d_phi)

    p = np.polyfit(I_vals, F_vals, 2)
    L0 = p[0]
    F0 = p[2]
    data_L0.log(
        {'Nx': Nx,
         'Ny': Ny,
         'frustration': f,
         'free_energy': F0,
         'L0': L0
        }
    )

    
            
    

