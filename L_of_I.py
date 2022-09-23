#!/usr/bin/env python3

from JJAsolver.network import network
import numpy as np
import matplotlib.pyplot as plt
Nx = 16
Ny = 16

def cpr(gamma):
    return np.sin(gamma)

def free_energy(gamma):
    return 1 - np.cos(gamma)

n = network(Nx, Ny, cpr_x=cpr, cpr_y=cpr, free_energy_x = free_energy, free_energy_y = free_energy)

n.add_vortex(int(Nx/2) - 0.5, int(Ny/2) - 0.5)
j = 0.1
n.set_current(j * Ny)
n.set_frustration(0.1)

for i in range(1000):
    print("delta = ", n.optimization_step())
    print("F = ", n.free_energy())
    print("I = ", n.get_current())
    if i % 10 == 0:
        plt.clf()
        n.plot_currents()
        plt.show()
