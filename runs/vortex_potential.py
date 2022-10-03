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
tau = 0.999

def cpr_x(gamma):
    return jj_cpr_ballistic(gamma, tau)

                                            

def f_x(gamma):
    return jj_free_energy_ballistic(gamma, tau)

n = network(
    Nx, Ny,
    cpr_x=cpr_x,
    cpr_y=cpr_x,
    free_energy_x = f_x,
    free_energy_y = f_x,
)

n.add_vortex(int((Nx - 1)/2) + 0.5, int((Ny - 1)/2) + 0.5, vorticity=1)
print(n.phi_matrix[7,7]) # pi
print(n.phi_matrix[8,7]) # 0
n.plot_currents()
n.add_phase_gradient(1.1)
plt.show()
# i_vals = range(200)
# F_vals = []
# for i in i_vals:
#     F = n.free_energy()
#     print("i = %d, F = %g" % (i, F))
#     F_vals.append(F)
#     delta = n.optimization_step(optimize_leads=True, epsilon=0.1)
#     print("delta = ", delta)
#     n.phi_matrix[7,7] = np.pi
#     n.phi_matrix[8,7] = 0
    
# plt.plot(i_vals, F_vals)
# plt.show()

i_vals = range(200)
F_vals = []
for i in i_vals:
    F = n.free_energy()
    print("i = %d, F = %g" % (i, F))
    F_vals.append(F)
    delta = n.optimization_step(fix_contacts=True, epsilon=0.1)
    print(n.phi_matrix[0,0])
    print("delta = ", delta)
plt.plot(i_vals, F_vals)
plt.show()

    
# x_vals = np.linspace(-3, 3, 200)

# tau_vals = (0.01, 0.5, 0.75, 0.9, 0.95, 0.99)
# for tau in tau_vals:

#     F_vals = []

#     for x in x_vals:
#         print("tau = %g, x = %g" % (tau, x))
#         n.reset_network()
#        # n.add_vortex(int((Nx - 1)/2) + 0.5, int((Ny - 1)/2) + 0.5)
#         n.add_vortex(int((Nx - 1)/2) + 0.5 +  x, int((Ny - 1)/2) + 0.5, vorticity=1)
#         F_vals.append(n.free_energy())

#     F_vals = np.array(F_vals)
#     F_vals -= np.amin(F_vals)
#     plt.plot(x_vals, F_vals, label="tau = %g" % tau)
# plt.grid()
# plt.show()

