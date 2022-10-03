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

def cpr_x(gamma): 
    return np.sin(gamma)

                                            

def f_x(gamma):
    return 1 -np.cos(gamma)




n = network(
    Nx, Ny,
    cpr_x=cpr_x,
    cpr_y=cpr_x,
    free_energy_x = f_x,
    free_energy_y = f_x,
)


n.add_vortex(4.5, 4.5)
n.add_vortex(6.5, 6.5)
n.add_vortex(10, 10)
n.add_vortex(18.5, 18.5)
n.plot_currents()
plt.show()
print("winding number: ", n.winding_number() / (2*np.pi))
