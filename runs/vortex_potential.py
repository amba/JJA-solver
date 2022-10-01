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
tau = 0.99

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

n.add_vortex(int((Nx - 1)/2) + 0.5, int((Ny - 1)/2))
F1 = n.free_energy()
n.plot_currents()
plt.show()

n.reset_network()
n.add_vortex(int((Nx - 1)/2) + 0.5, int((Ny - 1)/2) + 0.5)
n.plot_currents()
plt.show()
F0 = n.free_energy()

print("F1 = %g, F0 = %g" % (F1, F0))
print("F1 - F0 = %g" % (F1 - F0))
