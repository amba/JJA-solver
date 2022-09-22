#!/usr/bin/env python3

from JJAsolver.network import network
import numpy as np

Nx = 16
Ny = 16

def cpr(gamma):
    return np.sin(gamma)

def free_energy(gamma):
    return 1 - np.cos(gamma)

n = network(Nx, Ny, cpr_x=cpr, cpr_y=cpr, free_energy_x = free_energy, free_energy_y = free_energy)

n.set_frustration(0.5)

for i in range(1000):
    print(n.optimization_step())

