#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

Nx = 200
Ny = 200
a = 0.1 # lattice parameter

from numba import njit

@njit()
def time_step_with_A(phi, A_y):
    # assume that div A = 0,
    # A * e_x = 0
    
    Nx = phi.shape[0]
    Ny = phi.shape[1]
    alpha = -1
    beta = 1
#    rhs = np.zeros((Nx, Ny))
    rhs = alpha * phi  + beta * np.abs(phi)**2 * phi
    
    laplace = np.zeros((Nx, Ny)) + 0*1j
    # x direction
    laplace[0,:] += (phi[1,:] - phi[0,:])
    for i in range(1, Nx-1):
        laplace[i,:] += phi[i+1,:] + phi[i-1,:] - 2*phi[i,:]
    laplace[Nx-1,:] += phi[Nx-2,:] - phi[Nx-1,:]
    # y direction
    laplace[:,0] += phi[:,1] - phi[:,0]
    for i in range(1, Ny-1):
        laplace[:,i] += phi[:,i+1] + phi[:,i-1] - 2*phi[:,i]
    laplace[:,Ny-1] += phi[:,Ny-2] - phi[:,Ny-1]

    A_times_grad_phi = np.zeros((Nx, Ny)) + 0*1j
    # A * Δphi = A_y d/dy phi
    for i in range(Ny-1):
        A_times_grad_phi[:,i] = (phi[:,i+1] - phi[:,i]) * A_y[:,i]
    A_times_grad_phi[:,-1] = A_times_grad_phi[:,-2]
    
    
    rhs += -laplace/a**2 + 2*1j * A_times_grad_phi/a + A_y**2 * phi
    epsilon = 0.2 * a**2
    phi -= epsilon * rhs
    

@njit()
def time_step(phi):
    Nx = phi.shape[0]
    Ny = phi.shape[1]
    alpha = -1
    beta = 1
#    rhs = np.zeros((Nx, Ny))
    rhs = alpha * phi  + beta * np.abs(phi)**2 * phi
    laplace = np.zeros((Nx, Ny)) + 0*1j
    # x direction
    laplace[0,:] += (phi[1,:] - phi[0,:])
    for i in range(1, Nx-1):
        laplace[i,:] += phi[i+1,:] + phi[i-1,:] - 2*phi[i,:]
    laplace[Nx-1,:] += phi[Nx-2,:] - phi[Nx-1,:]
    # y direction
    laplace[:,0] += phi[:,1] - phi[:,0]
    for i in range(1, Ny-1):
        laplace[:,i] += phi[:,i+1] + phi[:,i-1] - 2*phi[:,i]
    laplace[:,Ny-1] += phi[:,Ny-2] - phi[:,Ny-1]

    laplace *= 1/a**2
    rhs -= laplace
    epsilon = 0.2 * a**2
    phi -= epsilon * rhs
    
def gen_A(B):
    A_y = np.zeros((Nx, Ny))
    for i in range(Nx):
        A_y[i,:] = a * B * i
    return A_y

def plot(phi):
    m = phi.copy()
    m = np.flip(m, axis=1)
    m = np.swapaxes(m, 0, 1)
    plt.imshow(np.abs(m), aspect='equal', cmap='gray')
    plt.colorbar(format="%.1f", label='|Φ|')
    plt.show()


order_parameter = np.ones((Nx, Ny)) + 0*1j # complex order parameter
A_y = gen_A(0.1)

# for i in range(Nx):
#     order_parameter[i,:] = np.exp(1j * i / 15)
    
phi_r = order_parameter[-1,0]
for i in range(100000):
    print("i = ", i)
    time_step_with_A(order_parameter, A_y)
    if i % 1000 == 0:
        plot(order_parameter)
        
