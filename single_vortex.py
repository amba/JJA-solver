#!/usr/bin/env python3

from JJAsolver.network import network, jj_cpr_ballistic, jj_free_energy_ballistic

import numpy as np
import matplotlib.pyplot as plt
Nx = 16
Ny = 16


def cpr_x(gamma):
    return np.sin(gamma)
    # #return np.sin(gamma) + sin2_term * np.sin(2*gamma) + cos_term*np.cos(gamma)
    # return jj_cpr_ballistic(gamma, tau)

                                            
#def cpr_y(gamma):
    
  #  return np.sin(gamma) + sin2_term * np.sin(2*gamma)
#    return jj_cpr_ballistic(gamma, tau)

def f_x(gamma):
    return 1 -np.cos(gamma)# - 0.5 * sin2_term * np.cos(2 * gamma) +\
#        cos_term * np.sin(gamma)
#    return jj_free_energy_ballistic(gamma, tau)

#def f_y(gamma):
#    return 1 -np.cos(gamma) - 0.5 * sin2_term * np.cos(2 * gamma)
    return jj_free_energy_ballistic(gamma, tau)


n = network(
    Nx, Ny,
    cpr_x=cpr_x,
    cpr_y=cpr_x,
    free_energy_x = f_x,
    free_energy_y = f_x,
)

n.add_vortex(0.5 * (Nx - 1), 0.5 * (Nx - 1))
n.set_frustration(5 / (Nx * Ny))
n.phi_r = 0
n.phi_l = 0

for i in range(200):
    delta =  n.optimization_step(optimize_leads=True)
    print("i = %d, delta = %g" % (i, delta))        
    if abs(delta) < 1e-3:
        break
    print("F = ", n.free_energy())
    print("I = ", n.get_current())

print("phi_l = %g π, phi_r = %g π" % (n.phi_l/np.pi, n.phi_r/np.pi))
n.plot_currents()
plt.show()
