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
tau = 0.01

frustration_vals = np.linspace(-0.5, 0.5, 1001)
N_currents = 10
d_phi = 0.01 / N_currents

delta_tol = 1e-6
N_annealing=2000

folder = datafolder('L0_of_I')

data_L0 = datafile(folder,
                   file = "data_L0.dat",
                   params = ['Nx', 'Ny',
                             'tau',
                             'f',  'L0', 'LF', 'N_vortex'],
)

data_L_of_I = datafile(folder,
    file = "data_L_of_I.dat",
    params = ['Nx', 'Ny',
              'tau',
              'f', 'phi', 'I', 'F'],
)


def cpr_x(gamma):
#    return np.sin(gamma)
   # return np.sin(gamma)# + sin2_term * np.sin(2*gamma) + cos_term*np.cos(gamma)
#    print("cpr_x: tau = ", tau)
    return jj_cpr_ballistic(gamma, tau)

                                            
#def cpr_y(gamma):
    #return np.sin(gamma)# + sin2_term * np.sin(2*gamma)
    #return jj_cpr_ballistic(gamma, tau)

def f_x(gamma):
    #return 1  - np.cos(gamma)
 #   return 1 -np.cos(gamma)# - 0.5 * sin2_term * np.cos(2 * gamma) +\
        #        cos_term * np.sin(gamma)
    return jj_free_energy_ballistic(gamma, tau)

#def f_y(gamma):
 #   return 1 -np.cos(gamma)# - 0.5 * sin2_term * np.cos(2 * gamma)
    #return jj_free_energy_ballistic(gamma, tau)

n = network(
    Nx, Ny,
    cpr_x=cpr_x,
    cpr_y=cpr_x,
    free_energy_x = f_x,
    free_energy_y = f_x,
)


for f in frustration_vals:
    print("f = ", f)
    n.reset_network()
    n.set_frustration(f)
    t0 = time.time()
    n.find_ground_state(N_max=N_annealing, delta_tol=delta_tol)
    N_vortex = n.winding_number() / (2 * np.pi)
    I_vals = []
    F_vals = []
    phi_vals = []
    
    t0 = time.time()
    for sign in (+1, -1):
        m = copy.deepcopy(n)
        for i in range(N_currents):
            m.optimize(fix_contacts=True, maxiter=5000, delta_tol=delta_tol)
            I = m.get_current()
            F = m.free_energy()
            phi = m.phi_matrix[-1,0] - m.phi_matrix[0,0]
            print("I = ", I)
            data_L_of_I.log(
                {'Nx': Nx,
                 'Ny': Ny,
                 'tau': tau,
                 'f': f,
                 'phi': phi,
                 'I': I,
                 'F': F,
                 }
            )
            I_vals.append(I)
            F_vals.append(F)
            phi_vals.append(phi)
            m.add_phase_gradient(sign * d_phi)
    print("time for F(I) curve: ", time.time() - t0)
    data_L_of_I.new_block()
    p = np.polyfit(phi_vals, I_vals, 1)
    L0 = 1/p[0]
    p1 = np.polyfit(I_vals, F_vals, 2)
    LF = 2 * p1[0]
    data_L0.log(
        {'Nx': Nx,
         'Ny': Ny,
         'tau': tau,
         'f': f,
         'L0': L0,
         'LF': LF,
         'N_vortex': N_vortex
         }
    )

    
            
    

