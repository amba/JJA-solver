#!/usr/bin/env python3

from JJAsolver.network import network
import numpy as np
import matplotlib.pyplot as plt
Nx = 16
Ny = 16

tau = 0.9

def cpr_ballistic(gamma):
    return np.sin(gamma) / np.sqrt(1 - tau * np.sin(gamma/2)**2)


def free_energy_ballistic(gamma):
    return 4 / tau * (1 - np.sqrt(1 - tau * np.sin(gamma/2)**2))


def cpr(gamma):
    return np.sin(gamma)

def free_energy(gamma):
    return 1 - np.cos(gamma)

n = network(Nx, Ny, cpr_x=cpr_ballistic, cpr_y=cpr_ballistic, free_energy_x = free_energy_ballistic, free_energy_y = free_energy_ballistic)

n.add_vortex(int(Nx/2) - 0.5, int(Ny/2) - 0.5)
j = 0.3
n.set_current(j * Ny)
# n.set_frustration(0.1)

for i in range(1000):
    print("delta = ", n.optimization_step())
    print("F = ", n.free_energy())
    print("I = ", n.get_current())
    if i % 10 == 0:
        plt.clf()
        n.plot_currents()
        plt.show()
