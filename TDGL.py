#!/usr/bin/env python3

import numpy as np
import scipy.sparse

import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
np.set_printoptions(linewidth=200)

# Φ: complex order parameter
# φ: scalar electric potential

# TDGL equation:

# d/dt Φ + i φΦ = ΔΦ + Γ Φ - |Φ|²Φ

# current:

# I = I_SC + I_N
# I_SC = i/2(Φ grad Φ* - grad Φ Φ*)

# I_N = -σ grad φ

# current conservation:

# div(I_SC + I_N) = 0

# =>  Δφ = 1/σ * div(I_SC) = i/(2σ) (Φ ΔΦ* - ΔΦ Φ*) = -1/(2σ) Im(Φ ΔΦ* - ΔΦ Φ*)

# boundary conditions for φ:
#  - fixed value at left and right boundary (voltage bias)
#  - zero derivative normal to boundary at top and bottom boundaries

# A = -laplace

N = 100
conductivity = 1

# lattice constant = 1
def vector_to_matrix(x):
    return np.reshape(x, (N,N), order='F')

def matrix_to_vector(x):
    return np.reshape(x, N*N, order='F')

phi_l = 0.1

potential_vector_linear_part = np.zeros((N, N))
for i in range(N):
    potential_vector_linear_part[i,:] = phi_l * (1 - i/N)
potential_vector_linear_part = matrix_to_vector(potential_vector_linear_part)
potential_vector = np.zeros((N*N))
potential_vector += potential_vector_linear_part

order_param_matrix = np.zeros((N, N)) + 0.1j
for i in range(N):
    for j in range(N):
        order_param_matrix[i,j] = np.exp(1j * np.arctan2(j-N/2, i-N/2))

order_param_vector = matrix_to_vector(order_param_matrix)


div_I_SC = np.zeros((N * N)) 
gamma_matrix = np.ones((N, N))
gamma_matrix[0:30,:] = 0
gamma_matrix[-30:,:] = 0
#gamma_matrix[:,:10] = 0
#gamma_matrix[:,-10:] = 0

gamma_vector = matrix_to_vector(gamma_matrix)

print("setting up sparse matrix")
# A = -Δ
A = scipy.sparse.lil_matrix((N*N, N*N))

print("adding main diagonal")
for i in range(N*N):
    A[i,i] = 4

print("adding inner diagonals")

for i in range(N*N -1):
    A[i,i+1] = -1
    A[i+1,i] = -1

print("adding outer diagnoals")

for i in range(N*(N-1)):
    A[i,i+N] = -1
    A[i+N,i] = -1

print("manipulating diagonals")

for i in range(N):
    A[i*N,i*N] = 1
    A[i*N + N-1, i*N + N-1] = 1

for i in range(N-1):
    A[i*N, (i+1)*N] = 0
    A[(i+1)*N, i*N] = 0
    
    A[(i+1)*N-1, (i+2)*N-1] = 0
    A[(i+2)*N-1, (i+1)*N-1] = 0
    
    A[(i+1)*N-1, (i+1)*N] = 0
    A[(i+1)*N, (i+1)*N-1] = 0
print("converting to csr format")
A = scipy.sparse.csr_matrix(A)


print(A.__class__)

def time_step(delta_t):
    global order_param_vector, A, potential_vector, div_I_SC, gamma_vector

    t0 = time.time()
    
    order_param_vector -= delta_t * 1j * potential_vector * order_param_vector
    order_param_vector += delta_t * -A.dot(order_param_vector)
    order_param_vector += delta_t * gamma_vector  * order_param_vector
    order_param_vector += delta_t * -np.abs(order_param_vector)**2 * order_param_vector
    div_I_SC = -1/(2*conductivity) * np.imag(order_param_vector * -A.dot(np.conjugate(order_param_vector)) - np.conjugate (order_param_vector) * -A.dot(order_param_vector))
    x, info = scipy.sparse.linalg.cg(A, -div_I_SC, x0=potential_vector-potential_vector_linear_part)
    print("info = ", info)
    t1 = time.time()
    potential_vector = x + potential_vector_linear_part
    print("t = ", t1 - t0)
    

for i in range(10000):
    phi_matrix = vector_to_matrix(order_param_vector)
    potential_matrix = vector_to_matrix(potential_vector)
    print("i = ", i)
    if i % 100 == 0:
        print("running imshow")
        plt.imshow(potential_matrix)
        plt.pause(0.1)
    time_step(0.1)

