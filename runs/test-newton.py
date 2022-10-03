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

Nx = 2 * 16
Ny = 16


def cpr_x(gamma):
    return np.sin(gamma)

def f_x(gamma):
    return 1  - np.cos(gamma)

def diff_x(gamma):
    return np.cos(gamma)

n = network(
    Nx, Ny,
    cpr_x=cpr_x,
    cpr_y=cpr_x,
    free_energy_x = f_x,
    free_energy_y = f_x,
    diff_x = diff_x,
    diff_y = diff_x,
)



f = 0.05
run_vals = (1000, 2000, 5000)
I_norm_vals = []
F_vals = []

for run in run_vals:
    n.reset_network()
    n.set_frustration(f)
    I = 0
    for i in range(run):
        I = n.optimization_step(epsilon=0.55)
        print("i = %d\n I = %g" % (i, I))
    I_norm_vals.append(I)
    F_vals.append(n.free_energy())

plt.plot(run_vals, F_vals)
plt.show()
plt.plot(I_norm_vals, F_vals)
plt.show()
