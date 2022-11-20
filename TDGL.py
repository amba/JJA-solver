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
# I_SC = Im(Φ* grad Φ)

# I_N = -σ grad φ

# current conservation:

# div(I_SC + I_N) = 0

# =>  Δφ = 1/σ * div(I_SC) = i/(2σ) (Φ ΔΦ* - ΔΦ Φ*) = -1/(2σ) Im(Φ ΔΦ* - ΔΦ Φ*) = 1/σ Im(Φ* ΔΦ)

# boundary conditions for φ:
#  - fixed value at left and right boundary at x=0, x = N-1 (voltage bias)
#  - zero derivative normal to boundary at top and bottom boundaries y=0, y = M-1

# A = -laplace

N = 150 # x-axis
M = 150 # y-axis

conductivity = 10
#                    y ->
# matrix: a_00 a_01 a_02 ...
#         a_10 a_11 a_12 ...
#   x      ...
#   |
#         a_(N-1)0 ...     a_(N-1)(M-1)

# vector: a_00, a_01, a_02, ..., a_0(M-1), a_10, a_11, a_12, ..., a_(N-1)0, ..., a_(N-1)(M-1)




# lattice constant = 1
def vector_to_matrix(x):
    return np.reshape(x, (N,M), order='C')

def matrix_to_vector(x):
    return np.reshape(x, N*M, order='C')

phi_l = 1

potential_vector_linear_part = np.zeros((N, M))
for i in range(N):
    potential_vector_linear_part[i,:] = phi_l * (1 - i/N)
potential_vector_linear_part = matrix_to_vector(potential_vector_linear_part)
potential_vector = np.zeros((N*M))
potential_vector += potential_vector_linear_part

x_edge = 30
y_edge = 30

order_param_matrix = np.zeros((N, M)) + 0.1j
for i in range(N):
    for j in range(M):
        order_param_matrix[i,j] = 0.01 # np.exp(1j * np.arctan2(j-M/2, i-N/2))

order_param_matrix[0:x_edge,:] = 0
order_param_matrix[-x_edge:,:] = 0
order_param_matrix[:,0:y_edge] = 0
order_param_matrix[:,-y_edge:] = 0

order_param_vector = matrix_to_vector(order_param_matrix)


div_I_SC = np.zeros((N * M)) 
gamma_matrix = np.ones((N, M))
for i in range(N):
    gamma_matrix[i,:] *= (np.arctan((i-x_edge)/2) + np.pi/2) / np.pi
    gamma_matrix[i,:] *= (np.arctan(((N-x_edge)-i)/2) + np.pi/2) / np.pi
for j in range(M):
    gamma_matrix[:,j] *= (np.arctan((j-y_edge)/2) + np.pi/2) / np.pi
    gamma_matrix[:,j] *= (np.arctan(((M-y_edge)-j)/2) + np.pi/2) / np.pi

gamma_vector = matrix_to_vector(gamma_matrix)

print("setting up sparse matrix")
# A = -Δ
A = scipy.sparse.lil_matrix((N*M, N*M)) # N blocks of size (MxM)

print("adding main diagonal")
for i in range(N*M):
    A[i,i] = 4

print("adding inner diagonals")

for i in range(N*M -1):
    A[i,i+1] = -1
    A[i+1,i] = -1

print("adding outer diagnoals")

for i in range(M*(N-1)):
    A[i,i+M] = -1
    A[i+M,i] = -1

print("manipulating diagonals")

for i in range(N):
    A[i*M,i*M] = 1
    A[(i+1)*M -1, (i+1)*M - 1] = 1

for i in range(N-1):
    A[(i+1)*M-1, (i+1)*M] = 0
    A[(i+1)*M, (i+1)*M-1] = 0
    
    A[i*M, (i+1)*M] = 0
    A[(i+1)*M, i*M] = 0
    
    A[(i+1)*M-1, (i+2)*M-1] = 0
    A[(i+2)*M-1, (i+1)*M-1] = 0
    
print("converting to csr format")
A = scipy.sparse.csr_matrix(A)
print(A.toarray())
print(A.__class__)

def time_step(delta_t):
    global order_param_vector, A, potential_vector, div_I_SC, gamma_vector

    t0 = time.time()
    
    order_param_vector -= delta_t * 1j * potential_vector * order_param_vector
    order_param_vector += delta_t * -A.dot(order_param_vector)
    order_param_vector += delta_t * gamma_vector  * order_param_vector
    order_param_vector += delta_t * -np.abs(order_param_vector)**2 * order_param_vector
    div_I_SC = 1/conductivity * np.imag(np.conjugate(order_param_vector) * -A.dot(order_param_vector))
    x, info = scipy.sparse.linalg.cg(A, -div_I_SC, x0=potential_vector-potential_vector_linear_part)
    print("info = ", info)
    t1 = time.time()
    potential_vector = x + potential_vector_linear_part
    print("t = ", t1 - t0)

def currents():
    global potential_vector, order_param_vector
    phi_matrix = vector_to_matrix(order_param_vector)
    potential_matrix = vector_to_matrix(potential_vector)
    Ix_N = -conductivity * np.gradient(potential_matrix, axis=0)
    Iy_N = -conductivity * np.gradient(potential_matrix, axis=1)
    Ix_S = np.imag(np.conjugate(phi_matrix) * np.gradient(phi_matrix, axis=0))
    Iy_S = np.imag(np.conjugate(phi_matrix) * np.gradient(phi_matrix, axis=1))
    return Ix_N, Iy_N, Ix_S, Iy_S

    
for i in range(10000):
    phi_matrix = vector_to_matrix(order_param_vector)
    potential_matrix = vector_to_matrix(potential_vector)
    print("i = ", i)
    if i % 10 == 0:
        Ix_N, Iy_N, Ix_S, Iy_S = currents()
        
        print("running imshow")
        #plt.imshow(potential_matrix)
        plt.imshow(np.abs(phi_matrix))
        #plt.imshow(np.sqrt((Ix_N + Ix_S)**2 + (Iy_N + Iy_S)**2))
        #plt.imshow(gamma_matrix)
        plt.pause(0.1)
    time_step(0.1)

