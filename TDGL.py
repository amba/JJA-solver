#!/usr/bin/env python3

import numpy as np
import numpy.random
import scipy.sparse

import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
np.set_printoptions(linewidth=200)

# Φ: complex order parameter
# φ: scalar electric potential

# TDGL equation:

# d/dt Φ + i φΦ = ΔΦ + α Φ - β|Φ|²Φ 

# with the coherence length ξ = 1/sqrt(α)
# and Φ_\infty = sqrt(α/β)

# the GL critical current density:
#  j_c = |Φ_\intfy|^2 * 2/3 * sqrt(4 α /3) = 0.77 * |Φ_\infty|^2 * sqrt(α)
# at |Φ_c| = 0.8165 * |Φ_\infty|


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

N = 200 # x-axis
M = 200 # y-axis

conductivity = 1
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

x_edge = 20
y_edge = 20

j = 0.5

potential_vector = np.zeros((N*M))

order_param_matrix = 0.1 * numpy.random.rand(N, M) + 0.1j * numpy.random.rand(N, M)

order_param_matrix[0:x_edge:,:] = 0
order_param_matrix[-x_edge:,:] = 0
order_param_matrix[:,0:y_edge] = 0
order_param_matrix[:,-y_edge:] = 0

order_param_vector = matrix_to_vector(order_param_matrix)


div_I_SC = np.zeros((N * M))
alpha = (1.0/2.0)**2
beta = alpha

alpha_matrix = -5 * alpha * np.ones((N, M))
alpha_matrix[x_edge:-x_edge,y_edge:-y_edge] = alpha

B = 0.01
A_x_matrix = np.zeros((N, M))
for j in range(M):
    A_x_matrix[:,j] = (j - int(M/2))*B

A_x_vector = matrix_to_vector(A_x_matrix)

#alpha_matrix[int(N/2),int(M/2)] = -3*alpha
# for i in range(N):
#     for j in range(M):
#         order_param_matrix[i,j] =  alpha * np.exp(1j * np.arctan2(j-M/2, i-N/2))


#order_param_matrix[0:x_edge,:] = 0

#alpha_matrix += alpha/10 *numpy.random.rand(N, M) - 0.1


for i in range(N):
    for j in range(M):
        if i % 10 == 0 or j % 10 == 0:
            alpha_matrix[i,j] -=  3* alpha

# for i in range(N):
#     for j in range(M):
#         if i < int(N/2) and (i - N/2)**2 + j**2 < (4*y_edge)**2:
#             alpha_matrix[i,j] = -alpha
#         if i < int(N/2) and (i - N/2)**2 + (M-j)**2 < (4*y_edge)**2:
#             alpha_matrix[i,j] = -alpha
            
# alpha_matrix[int(N/2),0:3*y_edge] = -1
# alpha_matrix[int(N/2)+1,0:3*y_edge] = -1
# alpha_matrix[int(N/2),-3*y_edge:] = -1
# alpha_matrix[int(N/2)+1,-3*y_edge:] = -1
# choose alpha(x,y) > 0 for superconducting region and
#        alpha(x,y) < 0 for normal regions


alpha_vector = matrix_to_vector(alpha_matrix)

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
    A[i*M,i*M] = 1 # ??? why not 3
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

grad_x = scipy.sparse.lil_matrix((N*M, N*M))
for i in range(M * (N-1)):
    grad_x[i, i+M] = 1
    grad_x[i+M, i] = -1
grad_x = scipy.sparse.csr_matrix(grad_x)

def time_step(phi):
    global order_param_vector, A, potential_vector, div_I_SC, alpha_vector
    delta_t = 0.1
    t0 = time.time()
    
    order_param_vector -= delta_t * 1j * potential_vector * order_param_vector
    order_param_vector += delta_t * -A.dot(order_param_vector)
    order_param_vector += delta_t * order_param_vector * (alpha_vector - beta * np.abs(order_param_vector)**2)
    order_param_vector -= delta_t * (2*1j * grad_x.dot(order_param_vector) * A_x_vector + A_x_vector**2 * order_param_vector)
    
    div_I_SC = -1/conductivity * np.imag(np.conjugate(order_param_vector) * A.dot(order_param_vector))
    
    for i in range(N):
        potential_vector[i*M:(i+1)*M] -= phi*(i-N/2) / N
    x, info = scipy.sparse.linalg.cg(A, -div_I_SC, x0=potential_vector, maxiter=10)
   # print("info = ", info)
    t1 = time.time()
    potential_vector = x
    for i in range(N):
        potential_vector[i*M:(i+1)*M] += phi*(i-N/2) / N
    
#    print("t = ", t1 - t0)

def currents():
    global potential_vector, order_param_vector
    phi_matrix = vector_to_matrix(order_param_vector)
    potential_matrix = vector_to_matrix(potential_vector)
    Ix_N = -conductivity * np.gradient(potential_matrix, axis=0)
    Iy_N = -conductivity * np.gradient(potential_matrix, axis=1)
    Ix_S = np.imag(np.conjugate(phi_matrix) * np.gradient(phi_matrix, axis=0))
    Iy_S = np.imag(np.conjugate(phi_matrix) * np.gradient(phi_matrix, axis=1))
    return Ix_N, Iy_N, Ix_S, Iy_S

phi = 0# 0.15 * 2 * x_edge / conductivity
print("phi = ", phi)
for i in range(1000000):
    print(i)
    # phi += 0.00001
    #print("phi = ", phi)
    phi_matrix = vector_to_matrix(order_param_vector)
    potential_matrix = vector_to_matrix(potential_vector)
#    print("|Φ| = ", np.abs(phi_matrix[int(N/2),int(M/2)]))
 #   print("i = ", i)
    if i % 50  == 0:
        Ix_N, Iy_N, Ix_S, Iy_S = currents()
        print("j = ", Ix_N[1, int(M/2)])
        print("j = ", Ix_N[-1, int(M/2)])
        
        print("running imshow")
        #plt.imshow(potential_matrix)
        #plt.clf()
        #plt.plot(range(N), np.angle(phi_matrix[:,int(M/2)]) / np.pi)
       # plt.plot(range(N), np.abs(phi_matrix[:,int(M/2)]))
        plt.imshow(np.abs(phi_matrix))
        #plt.imshow(np.sqrt((Ix_N)**2 + (Iy_N)**2))
        #plt.imshow(np.sqrt((Ix_N + Ix_S)**2 + (Iy_N + Iy_S)**2))
        #plt.imshow(alpha_matrix)
        plt.pause(0.01)
    time_step(phi)
    potential_vector *= 0

