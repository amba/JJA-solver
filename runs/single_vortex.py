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

# data_L0 = datafile(folder,
#                    file = "data_L0.dat",
#                    params = ['Nx', 'Ny', 'frustration', 'tau',
#                              'a1', 'b2',
#                              'F', 'L', 'F0', 'L0'],
# )

data_L_of_I = datafile(folder,
                       file = "data_L_of_I.dat",
                       params = ['Nx', 'Ny', 'frustration', 'tau',
                                 #'Iy',
                                 'a1', 'b2',
                                 'I', 'F', 'I0', 'F0'],
)

b2 = -0.15
a1 = 0
Iy = 1

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



#tau_vals = np.linspace(0.01, 0.999, 31)
#Iy_vals= np.linspace(0.8, 1.2, 11)
a1_vals = np.linspace(-0.3,0.3, 31)

frustration = 1 / (Nx * Ny)
delta_tol = 1e-4


for a1 in a1_vals:
    print("a1 = ", a1)
    #print("Iy = ", Iy)
    print("tau = ", tau)
    n = network(
        Nx, Ny,
        cpr_x=cpr_x,
        cpr_y=cpr_y,
        free_energy_x = f_x,
        free_energy_y = f_y,
    )
    n0 = network(
        Nx, Ny,
        cpr_x=cpr_x,
        cpr_y=cpr_y,
        free_energy_x = f_x,
        free_energy_y = f_y,
    )
    n.add_vortex(int((Nx-1) / 2) + 0.5, int((Ny-1) /2) + 0.5)
    n.set_frustration(frustration)
    n.optimize(delta_tol=delta_tol, optimize_leads=True)
    n0.optimize(delta_tol=delta_tol, optimize_leads=True)
    print("free energy of ground state = ", n.free_energy())
    print("free energy without vortex = ", n0.free_energy())

    # n.plot_currents()
    # plt.show()
    # n0.plot_currents()
    # plt.show()
    N_currents = 100
    d_phi = (0.25 / N_currents)
    I_vals = []
    I0_vals = []
    F0_vals = []
    F_vals = []
    t0 = time.time()
    for sign in (+1, -1):
        m = copy.deepcopy(n)
        m0 = copy.deepcopy(n0)
        for i in range(N_currents):
            m.optimize(optimize_leads=False, maxiter=1000, delta_tol=delta_tol)
            m0.optimize(optimize_leads=False, maxiter=1000, delta_tol=delta_tol)
            I = m.get_current()
            print("I = ", I)

            # m.plot_currents()
            # plt.show()
            F = m.free_energy()
            I0 = m0.get_current()
            F0 = m0.free_energy()
            print("F = ", F)
            data_L_of_I.log(
                {'Nx': Nx,
                 'Ny': Ny,
                 'tau': tau,
                 # 'Iy': Iy,
                 'a1': a1,
                 'b2': b2,
                 'frustration': frustration,
                 'I': I,
                 'F': F,
                 'I0': I0,
                 'F0': F0,
                 }
            )
            I_vals.append(I)
            F_vals.append(F)
            m.add_phase_gradient(sign * d_phi)
            m0.add_phase_gradient(sign * d_phi)
            I0_vals.append(I0)
            F0_vals.append(F0)
    print("time for F(I) curve: ", time.time() - t0)
    data_L_of_I.new_block()
    # p = np.polyfit(I_vals, F_vals, 2)
    # L = p[0]/2
    # F = p[2]
    # p0 = np.polyfit(I0_vals, F0_vals, 2)
    # L0 = p0[0]/2
    # F0 = p0[2]
    # print("L = %g, L0 = %g" % (L, L0))
    # print("L / L0 = %g" % (L/L0))
    # print("α = ", (L / L0 - 1) * Nx * Ny)
    # #exit(1)
    # data_L0.log(
    #     {'Nx': Nx,
    #      'Ny': Ny,
    #      'tau': tau,
    #      #'Iy': Iy,
    #      'a1': a1,
    #      'b2': b2,
    #      'frustration': frustration,
    #      'F': F,
    #      'L': L,
    #      'F0': F0,
    #      'L0': L0,
    #     }
    # )





