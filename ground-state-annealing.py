#!/usr/bin/env python3

import sys
import os

sys.path.append(os.getcwd() + '/..')
from JJAsolver.data import datafile, datafolder

import copy
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import time
# from numba import njit



Nx = 50

Ny = 50
phi_matrix = np.zeros((Nx, Ny)) 
phi_matrix_prev = np.zeros((Nx, Ny))


I_x = np.zeros((Nx -1, Ny))
I_y = np.zeros((Nx, Ny-1))
A_x = np.zeros((Nx -1, Ny))


x_coords, y_coords = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")

def plot_currents():
    x_current_xcoords, x_current_ycoords = np.meshgrid(np.arange(Nx-1), np.arange(Ny), indexing="ij")
    x_current_xcoords = x_current_xcoords.astype('float64')
    x_current_ycoords = x_current_ycoords.astype('float64')
    x_current_xcoords += 0.5
    y_current_xcoords, y_current_ycoords = np.meshgrid(np.arange(Nx), np.arange(Ny-1), indexing="ij")
    y_current_xcoords = y_current_xcoords.astype('float64')
    y_current_ycoords = y_current_ycoords.astype('float64')
    y_current_ycoords += 0.5
    plt.clf()
    plt.quiver(x_current_xcoords, x_current_ycoords, I_x, np.zeros(I_x.shape),
               scale = 0.02, scale_units='dots',
#               pivot='mid', # units='width', scale=5*Nx, width=1/(30*Nx)
    )
    plt.quiver(y_current_xcoords, y_current_ycoords, np.zeros(I_y.shape), I_y,
               scale = 0.02, scale_units='dots',
#               pivot='mid', # units='width', scale=5*Nx, width=1/(30*Nx)
    )
    # plt.scatter(x_coords, y_coords, marker='s', c='b', s=5)
    #plt.show()

def plot_phases():
    plt.clf()
    m = phi_matrix.copy()
    m = np.flip(m, axis=1)
    m = np.swapaxes(m, 0, 1)
    plt.imshow(m/np.pi, aspect='equal', cmap='gray')
    plt.colorbar(format="%.1f", label='φ')
    plt.show()
    
def plot_flux():
    plt.clf()
    m = np.zeros((Nx - 1, Ny - 1))
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            m[i,j] = I_x[i, j] - I_x[i,j+1]  + I_y[i+1,j] - I_y[i,j]
            
    m /= 4
    m = np.flip(m, axis=1)
    m = np.swapaxes(m, 0, 1)

    plt.imshow(m, aspect='equal', cmap='gray')
    plt.colorbar(format="%.1f", label='flux')
    #plt.show()

# folder = datafolder('current_bias_array')
# data = datafile(folder, file="data.dat", params=['Nx', 'Ny', 'phi0', 'f0', 'phi', 'f'])

# def cpr_x(gamma):
#     return np.sin(gamma)

# def f_x(gamma):
#     return 1  - np.cos(

def update_current():
    global phi_matrix, I_x, I_y, A_x
    I_x[:,:] = np.sin(phi_matrix[1:,:] - phi_matrix[:-1,:] + A_x[:,:])
    I_y[:,:] = np.sin(phi_matrix[:,1:] - phi_matrix[:,:-1])

def free_energy():
    global I_x, I_y
    f = np.sum(1 - np.sqrt(1 - I_x**2)) + np.sum(1 - np.sqrt(1 - I_y**2))
    return f / (Nx * Ny)

def optimization_step(epsilon=0.2, temp=0):
    global phi_matrix, phi_matrix_prev, I_x, I_y , A_x
    
    
    update_current()

    # d/dt Φ_i = ε sum_j I(j->i)
    # equivalent to d/dt Φ = ε grad F(Φ) with the free energy F(Φ)
    # similar to a time-dependent Ginzburg-Landau equation

    # x-direction
    phi_matrix[0,:] += epsilon * I_x[0,:]
    phi_matrix[1:-1,:] += epsilon * (I_x[1:,:] - I_x[:-1,:])
    phi_matrix[-1,:] += epsilon * (-I_x[-1,:])

    # y-direction
    phi_matrix[:,0] += epsilon * I_y[:,0]
    phi_matrix[:,1:-1] += epsilon * (I_y[:,1:] - I_y[:,:-1])
    phi_matrix[:,-1] += epsilon * (-I_y[:,-1])
    if temp > 0:
        phi_matrix += temp * numpy.random.rand(Nx, Ny)
    delta = np.sum(np.abs(phi_matrix- phi_matrix_prev))
    phi_matrix_prev[:,:] = phi_matrix

    return delta

def calculate_vector_potential(f):
    global A_x
    for i in range(Ny):
        A_x[:,i] = -2*np.pi * (f * i - (Ny-1)/2 * f)
    

def add_vortex(x0, y0, vorticity=1):
    global phi_matrix
    
    phi_matrix += vorticity * np.arctan2(y_coords - y0, x_coords - x0)

# for i in range(100):
#     x = numpy.random.rand() * Nx
#     y= numpy.random.rand() * Ny
#     add_vortex(x, y)

calculate_vector_potential(0.49)

t_start = 2
num_steps = 1000000
for i in range(num_steps):
    print(i)
    epsilon = 0.2
    t = t_start * (num_steps - i) / num_steps
    delta = optimization_step(epsilon, temp=t)
    f = free_energy()
    print("t = %.4g, f = %.4g" % (t, f))

plot_flux()
plt.show()

