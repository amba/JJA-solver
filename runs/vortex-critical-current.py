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


folder = datafolder('L0_of_I')

data_L_of_I = datafile(folder,
                       file = "data_L_of_I.dat",
                       params = ['Nx', 'Ny', 'frustration',
                                 # 'tau',
                                 #'Iy',
                                 'a1', 'b2',
                                 'I', 'F'],
)

b2 = -0.15
a1 = 0

def cpr_x(gamma):
    return np.sin(gamma) + b2 * np.sin(2*gamma) + a1 *np.cos(gamma)
    #    print("cpr_x: tau = ", tau)
    #return Iy * jj_cpr_ballistic(gamma, tau)

                                            
def cpr_y(gamma):
    #return np.sin(gamma)
    return np.sin(gamma) + b2 * np.sin(2*gamma)
#   return jj_cpr_ballistic(gamma, tau)

def f_x(gamma):
    # return 1 - np.cos(gamma)
    return 1 -np.cos(gamma) - 0.5 * b2 * np.cos(2 * gamma) + a1 * np.sin(gamma)
    #return Iy * jj_free_energy_ballistic(gamma, tau)

def f_y(gamma):
     return 1 -np.cos(gamma) - 0.5 * b2 *  np.cos(2 * gamma)
    #return jj_free_energy_ballistic(gamma, tau)


a1_vals = np.linspace(-0.3,0.3, 31)

frustration = 1 / (Nx * Ny)
delta_tol = 1e-4


for a1 in a1_vals:
    print("a1 = ", a1)
    n = network(
        Nx, Ny,
        cpr_x=cpr_x,
        cpr_y=cpr_y,
        free_energy_x = f_x,
        free_energy_y = f_y,
    )

    n.add_vortex(int((Nx-1) / 2) + 0.5, int((Ny-1) /2) + 0.5)
    n.set_frustration(frustration)
    n.optimize(delta_tol=delta_tol, optimize_leads=True)
    print("free energy of ground state = ", n.free_energy())

    N_currents = 100
    d_phi = (0.25 / N_currents)
    for sign in (+1, -1):
        m = copy.deepcopy(n)
        last_F_val = 0
        for i in range(N_currents):
            m.optimize(optimize_leads=False, maxiter=1000, delta_tol=delta_tol)
            I = m.get_current()
            F = m.free_energy()
            print("sign = %d, i = %d, I = %g, F = %g" % (sign, i, I, F))
            data_L_of_I.log(
                {'Nx': Nx,
                 'Ny': Ny,
                 #     'tau': tau,
                 # 'Iy': Iy,
                 'a1': a1,
                 'b2': b2,
                 'frustration': frustration,
                 'I': I,
                 'F': F,
                 }
            )
            m.add_phase_gradient(sign * d_phi)
            if F < 0.999 * last_F_val:
                break
            last_F_val = F
    data_L_of_I.new_block()
