#!/usr/bin/env python3

import sys
import os

sys.path.append(os.getcwd() + '/..')
from JJAsolver.network import network
from JJAsolver.data import datafile, datafolder

import copy
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
Nx = 16
Ny = 16

folder = datafolder('long_range_model')
data = datafile(folder, file="data.dat", params=['Nx', 'Ny', 'phi0', 'f0', 'phi', 'f'])

def cpr_x(gamma):
    return np.sin(gamma)

def f_x(gamma):
    return 1  - np.cos(gamma)


alpha = 1.8

@njit()
def numba_free_energy(phi_matrix, Nx, Ny):
    f = 0
    for i0 in range(Nx):
        for j0 in range(Ny):
            for i1 in range(i0 + 1, Nx):
                for j1 in range(j0 + 1, Ny):
                    r = (i1 - i0)**2 + (j1 - j0)**2
                    f += 1/r**alpha * ( 1 - np.cos(phi_matrix[i1,j1] - phi_matrix[i0, j0]))
    return f

def free_energy(network):
    phi_matrix = network.phi_matrix
    Nx = network.Nx
    Ny = network.Ny
    return numba_free_energy(phi_matrix, Nx, Ny)
   
def optimize(n, tol=1e-6, max_iter=10000, **kwargs):
    for i in range(max_iter):
        delta = optimization_step(n,  **kwargs)
        print("delta = %g" % delta)
        if delta < tol:
            break
    return delta

@njit()
def numba_optimization_step(phi_matrix, Nx, Ny, optimize_leads):
    I_norm = 0
    epsilon = 0.2
    if optimize_leads == False:
        i_range = range(1, Nx-1)
    else:
        i_range = range(Nx)

    for i in i_range:
        for j in range(Ny):
            f_prime = 0
            for i1 in range (Nx):
                for j1 in range(Ny):
                    if i != i1 or j != j1:
                        r = (i1 - i)**2 + (j1 - j)**2
                        f_prime += 1/r**alpha * np.sin(phi_matrix[i,j] - phi_matrix[i1, j1])
            phi_matrix[i,j] = phi_matrix[i,j] - epsilon * f_prime
            I_norm += np.abs(f_prime)
    return I_norm / (Nx * Ny)

def optimization_step(n, optimize_leads=True):
    Nx = n.Nx
    Ny = n.Ny
    phi_matrix = n.phi_matrix
    return numba_optimization_step(phi_matrix, Nx, Ny, optimize_leads)

n = network(
    Nx, Ny,
    cpr_x=cpr_x,
    cpr_y=cpr_x,
    free_energy_x = f_x,
    free_energy_y = f_x,
)

n0 = network(
    Nx, Ny,
    cpr_x=cpr_x,
    cpr_y=cpr_x,
    free_energy_x = f_x,
    free_energy_y = f_x,
)

    
N_currents = 400
delta_phi = 0.05 / N_currents
for sign in (+1, -1):
    n.reset_network()
    n0.reset_network()
    n.add_vortex(7.5, 7.5)
    optimize(n, optimize_leads=True)
    for i in range(N_currents):
        phi = n.phi_matrix[-1,0] - n.phi_matrix[0,0]
        phi0 = n0.phi_matrix[-1,0] - n0.phi_matrix[0,0]

        optimize(n, max_iter=1000, optimize_leads=False)
        optimize(n0, optimize_leads=False)

        f = free_energy(n)
        f0 = free_energy(n0)
        print("phi = %g π, f = %g" %(phi/np.pi, f))
        data.log(
            {'Nx': Nx,
             'Ny': Ny,
             'phi': phi,
             'phi0': phi0,
             'f': f,
             'f0': f0
             })
        n.add_phase_gradient(sign * delta_phi)
        n0.add_phase_gradient(sign * delta_phi)

n.reset_network()
n.add_vortex(7.5, 7)

f = free_energy(n)
print("free energy: ", f)
