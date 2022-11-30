#!/usr/bin/env python3

import numpy as np
import numpy.linalg
import numpy.random
import scipy.sparse

import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
np.set_printoptions(linewidth=200)

# JJA lattice:
#                N junctions long
#             # #  ...                #  M junctions wide
#             # #  ...                # 
# φ_l = 0     # #  ...                # φ_r (variable)
#             # #  ...                #
#


# Kirchoff law for each square "i":
# sum_j sin(φ_j - φ_i - A_ij) + d/dt(φ_j - φ_i) = 0

# => sum_j (d/dt φ_j - d/dt φ_i) = - sum_j sin(φ_j - φ_i - A_ij)


# for the right lead we enforce a fixed current 
# -I = sum_j sin( φ_j - φ_r) + d/dt(φ_j - φ_r)

# written in the linear system of equations

#               A * d/dt(φ) = f(φ)
# or           (-A) * d/dt(φ) = -f(φ) (to have a positive-definite system)

# update φ: φ -> φ + τ * d/dt(φ)





N = 100 # x-axis
M = 32 # y-axis
 
#                   <- M islands -> 
# matrix: a_00 a_01 a_02 ...
#         a_10 a_11 a_12 ...
#         ...
#   
#         a_(N-1)0 ...     a_(N-1)(M-1)

# vector: a_00, a_01, a_02, ..., a_0(M-1), a_10, a_11, a_12, ..., a_(N-1)0, ..., a_(N-1)(M-1), phi_r


def _normalize_phase(phi):
    phi = np.fmod(phi, 2 * np.pi)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    # normalize to (-pi, pi)
    phi = np.where(phi > np.pi, phi - 2*np.pi, phi)
    return phi


# lattice constant = 1
def vector_to_matrix(x):
    return np.reshape(x[:N*M], (N,M), order='C'), x[-1]

    
print("setting up sparse matrix")

A = scipy.sparse.lil_matrix((N*M+1, N*M+1)) # N blocks of size (MxM) + one 1x1 block for phi_r

print("adding main diagonal")
for i in range(N*M):
    A[i,i] = 4

A[-1,-1] = M

print("adding inner diagonals")

for i in range(N):
    for j in range(M-1):
        A[i*M+j,i*M +j+1] = -1
        A[i*M+j+1,i*M +j] = -1
print("adding outer diagnoals")

print("adding connections to right busbar")
for i in range(M):
    A[(N-1)*M + i, N*M] = -1
    A[N*M, (N-1)*M+i] = -1
    

for i in range(M*(N-1)):
    A[i,i+M] = -1
    A[i+M,i] = -1

print("manipulating diagonals")

for i in range(N):
    A[i*M,i*M] = 3
    A[(i+1)*M -1, (i+1)*M - 1] = 3
    
print("converting to csr format")
A = scipy.sparse.csr_matrix(A)

# B = A.toarray()

# print(B)
# print("print symm: ", np.sum(B - B.T))
# print("evals: ", numpy.linalg.eigvalsh(B))

phi_vector = 2 * np.pi * np.random.rand(N*M + 1)
rhs = np.zeros((N*M + 1)) 
phi_dot = np.zeros((N*M + 1))
I_x_matrix = np.zeros((N+1, M))
I_y_matrix = np.zeros((N, M-1))

def plot_phi_matrix():
    m, phi_r = vector_to_matrix(phi_vector)
    plt.imshow(_normalize_phase(m))

def plot_currents():
    global I_x_matrix, I_y_matrix

    island_x_coords, island_y_coords = np.meshgrid(np.arange(N), np.arange(M), indexing="ij")
    x_current_xcoords, x_current_ycoords = np.meshgrid(np.arange(N+1), np.arange(M), indexing="ij")
    x_current_xcoords = x_current_xcoords.astype('float64')
    x_current_ycoords = x_current_ycoords.astype('float64')
    x_current_xcoords += -0.5
    y_current_xcoords, y_current_ycoords = np.meshgrid(np.arange(N), np.arange(M-1), indexing="ij")
    y_current_xcoords = y_current_xcoords.astype('float64')
    y_current_ycoords = y_current_ycoords.astype('float64')
    y_current_ycoords += 0.5
    plt.clf()
    plt.quiver(x_current_xcoords, x_current_ycoords,
               I_x_matrix, np.zeros(I_x_matrix.shape),
               pivot='mid', units='width', scale=5*N, width=1/(30*N))
    plt.quiver(y_current_xcoords, y_current_ycoords,
               np.zeros(I_y_matrix.shape), I_y_matrix,
               pivot='mid', units='width', scale=5*N, width=1/(30*N))
    plt.scatter(island_x_coords, island_y_coords, marker='s', c='b', s=5)

def plot_flux():
    plt.clf()
    global I_x_matrix, I_y_matrix
    m = np.zeros((N - 1, M - 1))
    for i in range(N - 1):
        for j in range(M - 1):
            m[i,j] = I_x_matrix[i+1, j] - I_x_matrix[i+1,j+1]  + I_y_matrix[i+1,j] - I_y_matrix[i,j]
            
    m /= 4
    m = np.flip(m, axis=1)
    m = np.swapaxes(m, 0, 1)

    plt.imshow(m, aspect='equal', cmap='gray')
    plt.colorbar(format="%.1f", label='flux')
def update_rhs(I, frustration):
    global phi_vector
    global rhs
    # rhs_i = - sum_j sin(φ_j - φ_i)
    rhs *= 0
    print("I = ", I)
    print("frustration = ", frustration)
    for i in range(N):
        for j in range(M):
            A = 2*np.pi * frustration * (j - M/2)
            ind = i*M + j
            phi = phi_vector[ind]
            
            # upper and lower neighbours
            if j > 0:
                rhs[ind] += np.sin(phi_vector[ind-1] - phi) 
            if j < M-1:
                rhs[ind] += np.sin(phi_vector[ind+1] - phi)

            # left and right neighbours    
            if i > 0:
                rhs[ind] += np.sin(phi_vector[ind - M] - phi - A)
            else:
                rhs[ind] +=  np.sin(- phi - A)
                
            if i < N-1:
                rhs[ind] += np.sin(phi_vector[ind + M] - phi + A)
            else:
                rhs[ind] +=  np.sin(phi_vector[N*M] - phi + A)

    phi_r = phi_vector[-1]
    rhs[-1] = I
    
    for j in range(M):
        rhs[-1] += np.sin(phi_vector[(N-1)*M + j] - phi_r - 2*np.pi*frustration*(j-M/2))
    
def update_current():
    global phi_vector, phi_dot, I_x_matrix, I_y_matrix

    phi_matrix, phi_right_lead = vector_to_matrix(phi_vector)
    phi_dot_matrix, phi_right_lead = vector_to_matrix(phi_dot)
    # I_x
    for i in range(N+1):
        for j in range(M):
            A = 2*np.pi * frustration * (j - M/2)
            if i == 0:
                phi_l = 0
            else:
                phi_l = phi_matrix[i-1, j]
            if i == N:
                phi = phi_right_lead
            else:
                phi = phi_matrix[i, j]
                
            I_x_matrix[i,j] = np.sin(phi - phi_l + A)
    # I_y
    for i in range(N):
        for j in range(M-1):
            I_y_matrix[i,j] = np.sin(phi_matrix[i,j+1] - phi_matrix[i,j])
        
def time_step(I, tau, frustration):
    global phi_vector
    global rhs
    global phi_dot
    
    update_rhs(I, frustration)
    phi_dot, info = scipy.sparse.linalg.cg(A, rhs, x0=phi_dot)
    phi_vector += tau * phi_dot


i_vals = []
phi_r_vals = []
I_vals = []

I = 0
for i in range(100000):
    print(i)
    i_vals.append(i)
    phi_r_vals.append(phi_vector[-1])
    #I += 0.001
    I_vals.append(I)
    tau = 0.1
    frustration = 0.5
    time_step(M/10 * 2, tau, frustration)
    if i % 100 == 5:
        plt.clf()
        #plt.plot(range(N), phi_vector[:N*M:M])
        update_current()
        plot_flux()
#        plot_currents()
       # plt.show()
        #plot_phi_matrix()
        plt.pause(0.1)
