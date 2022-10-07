#!/usr/bin/env python3

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import time
from numba import njit

width = 1
N_intervals=30
delta_tol = 1e-2
Ic_upper = 1
phi0_upper =  0#np.pi
phi0_lower = 0# 0.3 * np.pi# phi0_upper * np.cos(30 *np.pi / 180)

# γ = phi_l - phi_r + k * x
# I = integral_x cpr(γ) = integral_x cpr(phi_l - phi_r + k*x)
# with x = 0 ... width
@njit()
def cpr(gamma):
    return np.sin(gamma)

@njit()
def current(phi_l, phi_r, k, phi0):
    points = np.linspace(0, k * width, N_intervals)
    rv = np.sum(cpr(phi_r - phi_l -phi0 + points))
    return rv/N_intervals


# phi1    |       phi2         |           phi4
#       phi3            phi3 + 2pi Φ/Φ0

@njit()
def network_current(phases, k):
    phi1 = phases[0]
    phi2 = phases[1]
    phi3 = phases[2]
    return Ic_upper * current(phi1, phi2, k, phi0_upper) + \
        current(phi1, phi3, -k, phi0_lower)

@njit()
def optimization_step(phases, k, flux):
    # phase vector = [phi1, phi2, phi3, phi4]
    # flux = Φ/Φ_0
    # phi1 and phi4 are fixed
    # optimize phi3 and phi4
    phi1 = phases[0]
    phi2 = phases[1] # phi2A = phi2B
    phi3 = phases[2] #
    phi4 = phases[3]

    phi_3B = phi3 + 2*np.pi * flux
    
    # get derivative of free energy with respect to phi2
    dF2 = 0
    dF2 += Ic_upper * current(phi1, phi2, k, phi0_upper)
    dF2 += current(phi3, phi2, -k, phi0_lower)
    dF2 += -current(phi2, phi_3B, -k, phi0_lower)
    dF2 += -Ic_upper * current(phi2, phi4, k, phi0_upper)

    # get derivative of free energy with respect to phi3
    dF3 = 0
    dF3 += current(phi1, phi3, -k, phi0_lower)
    dF3 += -current(phi3, phi2, -k, phi0_lower)
    dF3 += current(phi2, phi_3B, -k, phi0_lower)
    dF3 += -current(phi_3B, phi4, -k, phi0_lower)


    # do gradient descent of free energy
    epsilon = 0.2
    phi2_new = phi2 - epsilon * dF2
    phi3_new = phi3 - epsilon * dF3

    phases[1] = phi2_new
    phases[2] = phi3_new

    # use dF to check for convergence
    return np.abs(dF2) + np.abs(dF3)

@njit()
def optimize_phases(phi, k, flux):
    for i in range(2000):
        delta = optimization_step(phi, k, flux)
        if delta < delta_tol:
            return 0
    print("did not converge for k = ", k)
    
k_vals = np.linspace(-6*np.pi, 6*np.pi, 1000)
Iplus_vals = []
Iminus_vals = []
t0 = time.time()
for k in k_vals:
    print("k = ", k)
    flux = 2* k / (2*np.pi)
    phi_vals = np.linspace(0, 2*np.pi, 50)
    Iphi_vals = []
    phi2 = 2 * np.pi * numpy.random.rand()
    phi3 = 2 * np.pi * numpy.random.rand()
    phi = np.array([0, phi2, phi3, 0])
    
    for phi4 in phi_vals:
        phi[3] = phi4
        optimize_phases(phi, k, flux)
        I = network_current(phi, k)
        Iphi_vals.append(I)
    Iplus_vals.append(np.amax(Iphi_vals))
    Iminus_vals.append(np.amin(Iphi_vals))

print("time: ", time.time() - t0)
    
k_vals = k_vals / (2*np.pi)
plt.plot(k_vals, Iplus_vals)
plt.plot(k_vals, Iminus_vals)
plt.show()
    
    
    
    
    
    
    

# phi_vals = np.linspace(0, 2*np.pi, 100)
# k_vals = np.linspace(0, 30, 1000)
# Ic_vals = []
# for k in k_vals:
#     Iphi = []
#     for phi in phi_vals:
#         I = current(0, phi, k=k)
#         Iphi.append(I)
#     Ic_vals.append(np.amax(Iphi))







# plt.plot(k_vals, Ic_vals)
# plt.show()


