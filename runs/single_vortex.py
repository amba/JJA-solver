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

Nx = 20
Ny = 20
tau = 0

folder = datafolder('L0_of_I')

data_L0 = datafile(folder,
                   file = "data_L0.dat",
                   params = ['Nx', 'Ny', 'tau', 'frustration', 'free_energy', 'L0'],
)

data_L_of_I = datafile(folder,
                       file = "data_L_of_I_tau=%g.dat" % tau,
                       params = ['Nx', 'Ny', 'tau', 'frustration', 'I', 'free_energy'],
)

def cpr_x(gamma):
   # return np.sin(gamma)# + sin2_term * np.sin(2*gamma) + cos_term*np.cos(gamma)
#    print("cpr_x: tau = ", tau)
    return jj_cpr_ballistic(gamma, tau)

                                            
#def cpr_y(gamma):
    #return np.sin(gamma)# + sin2_term * np.sin(2*gamma)
    #return jj_cpr_ballistic(gamma, tau)

def f_x(gamma):
 #   return 1 -np.cos(gamma)# - 0.5 * sin2_term * np.cos(2 * gamma) +\
        #        cos_term * np.sin(gamma)
    return jj_free_energy_ballistic(gamma, tau)

#def f_y(gamma):
 #   return 1 -np.cos(gamma)# - 0.5 * sin2_term * np.cos(2 * gamma)
    #return jj_free_energy_ballistic(gamma, tau)



tau_vals = np.linspace(0.01, 0.99, 5)

for tau in tau_vals:
    print("tau = ", tau)

    n = network(
        Nx, Ny,
        cpr_x=cpr_x,
        cpr_y=cpr_x,
        free_energy_x = f_x,
        free_energy_y = f_x,
    )
    frustration = 1 / (Nx * Ny)
    n.add_vortex(int((Nx-1) / 2) + 0.5, int((Ny-1) /2) + 0.5)
    n.set_frustration(frustration)
    n.optimize(maxiter=300, delta_tol=1e-2, optimize_leads=True
    )
    print("free energy of ground state = ", n.free_energy())
    n.plot_currents()
    plt.show()
    d_phi = 0.03
    I_vals = [n.get_current(),]
    F_vals = [n.free_energy(),]
    t0 = time.time()
    for sign in (+1, -1):
        m = copy.deepcopy(n)
        for i in range(100):
            m.optimize(optimize_leads=False, maxiter=300, delta_tol=1e-2)
            I = m.get_current()
            print("I = ", I)
            F = m.free_energy()
            print("F = ", F)
            data_L_of_I.log(
                {'Nx': Nx,
                 'Ny': Ny,
                 'tau': tau,
                 'frustration': frustration,
                 'I': I,
                 'free_energy': F
                 }
            )
            I_vals.append(I)
            F_vals.append(F)
            m.add_phase_gradient(sign * d_phi)
    print("time for F(I) curve: ", time.time() - t0)
    data_L_of_I.new_block()
    p = np.polyfit(I_vals, F_vals, 2)
    L0 = p[0]/2
    F0 = p[2]
    data_L0.log(
        {'Nx': Nx,
         'Ny': Ny,
         'tau': tau,
         'frustration': f,
         'free_energy': F0,
         'L0': L0
        }
    )
data_L0.new_block()





