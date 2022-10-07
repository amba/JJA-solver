#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt



# γ = phi_l - phi_r + k * x
# I = integral_x cpr(γ) = integral_x cpr(phi_l - phi_r + k*x)
# with x = 0 ... width

def cpr(gamma):
    return np.sin(gamma)

def current(phi_l, phi_r, width=1, k=0, N_intervals=100):
    points = np.linspace(0, k * width, N_intervals)
    rv = np.sum(cpr(phi_r - phi_l + points))
    return rv/N_intervals


# phi1    |       phi2         |           phi4
#       phi3            phi3 + 2pi Φ/Φ0


def network_current(phases, k, *args, **kwargs):
    phi1 = phases[0]
    phi2 = phases[1]
    phi3 = phases[2]
    return current(phi1, phi2, k=k, *args, **kwargs) + \
        current(phi1, phi3, k=-k, *args, **kwargs)

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
    dF2 += current(phi1, phi2, k=k)
    dF2 += current(phi3, phi2, k=-k)
    dF2 += -current(phi2, phi_3B, k=-k)
    dF2 += -current(phi2, phi4, k=k)

    # get derivative of free energy with respect to phi3
    dF3 = 0
    dF3 += current(phi1, phi3, k=-k)
    dF3 += -current(phi3, phi2, k=-k)
    dF3 += current(phi2, phi_3B, k=-k)
    dF3 += -current(phi_3B, phi4, k=-k)


    # do gradient descent of free energy
    epsilon = 0.2
    phi2_new = phi2 - epsilon * dF2
    phi3_new = phi3 - epsilon * dF3

    phases[1] = phi2_new
    phases[2] = phi3_new

    # use dF to check for convergence
    return np.abs(dF2) + np.abs(dF3)

k_vals = np.linspace(0, 3*np.pi, 400)
I_vals = []
for k in k_vals:
    print("k = ", k)
    flux = 3*k
    phi_vals = np.linspace(-np.pi, np.pi, 100)
    Iphi_vals = []

    for phi4 in phi_vals:
        phi = np.array([0, phi4/2, phi4/2, phi4])
        for i in range(1000):
            delta = optimization_step(phi, k, flux)
#            print("delta = ", delta)
 #           print(phi_0)
            if delta < 1e-3:
                print("i = ", i)
                break
        I = network_current(phi, k)
        Iphi_vals.append(I)
    I_vals.append(np.amax(Iphi_vals))

plt.plot(k_vals, I_vals)
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


