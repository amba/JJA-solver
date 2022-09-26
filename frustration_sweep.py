#!/usr/bin/env python3

from JJAsolver.network import network, jj_cpr_ballistic, jj_free_energy_ballistic
import copy
import numpy as np
import matplotlib.pyplot as plt
Nx = 16
Ny = 16

tau = 0.99
cos_term = 0#0.3
sin2_term = 0#-0.3

def cpr_x(gamma):
    return np.sin(gamma) + sin2_term * np.sin(2*gamma) + cos_term*np.cos(gamma)
#    return jj_cpr_ballistic(gamma, tau)

                                            
def cpr_y(gamma):
    return np.sin(gamma) + sin2_term * np.sin(2*gamma)
   # return jj_cpr_ballistic(gamma, tau)

def f_x(gamma):
    return 1 -np.cos(gamma) - 0.5 * sin2_term * np.cos(2 * gamma) +\
        cos_term * np.sin(gamma)
    #return jj_free_energy_ballistic(gamma, tau)

def f_y(gamma):
    return 1 -np.cos(gamma) - 0.5 * sin2_term * np.cos(2 * gamma)
    #return jj_free_energy_ballistic(gamma, tau)


phi_vals = np.linspace(-np.pi, np.pi, 100)
plt.grid()
plt.plot(phi_vals / np.pi, cpr_x(phi_vals))
plt.plot(phi_vals / np.pi, f_x(phi_vals))
plt.show()

I_vals = (0,)# np.linspace(0,)
I_meas_vals = []
F_vortex_vals = []

n = network(
    Nx, Ny,
    cpr_x=cpr_x,
    cpr_y=cpr_y,
    free_energy_x = f_x,
    free_energy_y = f_y,
)

for i in range(Nx-1):
    for j in range(Ny-1):
        if i % 2  == 0 and j % 2 == 0:
            n.add_vortex(i + 0.5, j+0.5)
# n.add_vortex(5 + 0.5, 5 + 0.5)
# n.add_vortex(6 + 0.5, 6 + 0.5)
# n.add_vortex(7 + 0.5, 7 + 0.5)
#n.add_vortex(5 + 0.5, 5 + 0.5)
F_vals = []

# for i in range(Nx):
#     for j in range(Ny):
#         if i % 2 == 0 and j % 2 == 0:
#             n.phi_matrix[i,j] = 0
#         if i % 2 == 0 and j % 2 == 1:
#             n.phi_matrix[i,j] = -np.pi/2
#         if i % 2 == 1 and j % 2 == 0:
#             n.phi_matrix[i,j] = np.pi/2
#         if i % 2 == 1 and j % 2 == 1:
#             n.phi_matrix[i,j] = np.pi

f = 1.0 / 4
I_vals = []
n.set_frustration(f)
n.plot_currents()
print("current = ", n.get_current())
plt.show()
for x in range(50):
#    m = copy.deepcopy(n)
    #n.add_vortex(0.5 * (Nx - 1), 0.5 * (Nx - 1))
    n.set_current(0.1)
    for i in range(1000):
        delta =  n.optimization_step()
        print("f = %g, i = %d, delta = %g" % (f, i, delta))        
        if abs(delta) < 1e-2:
            break
    F = n.free_energy()
    I = n.get_current()
    print("F = ", F)
    print("I = ", I)
   # n.plot_currents()
  #  plt.show()
    F_vals.append(F)
    I_vals.append(I)
    
plt.plot(I_vals, F_vals, 'x')
plt.show()
        

# for sign in (+1, -1):
#     last_free_energy = 0
#     for I in sign * I_vals:
#         n.reset_network()
#        # n.add_vortex(0.5 * (Nx - 1), 0.5 * (Nx - 1))
#         n.set_current(I)
#         n.set_frustration(f)
        
#         #    last_delta = 1000
#         for i in range(2000):
#             print("I_meas = %g" % (n.get_current(),))
#             delta =  n.optimization_step()
#             print("I = %g, i = %d, delta = %g" % (I, i, delta))        
#             if abs(delta) < 1e-3:
#                 break
#         n.plot_phases()
#         plt.show()
#         n.plot_currents()
#         plt.show()
                          
                          
#         free_energy = n.free_energy()
#         I_meas_vals.append(n.get_current())
#         F_vortex_vals.append(n.free_energy())
#         if free_energy < last_free_energy:
#             break
#         last_free_energy = free_energy

# # I0_meas_vals = []
# # F0_vortex_vals = []


# # for I in I_vals:
# #     n.reset_network()
# #     n.set_current(I)
# #     n.set_frustration(0)

# #     for i in range(300):
# #         print("I_meas = %g" % (n.get_current(),))
# #         delta =  n.optimization_step()
# #         print("I = %g, i = %d, delta = %g" % (I, i, delta))
# #         if abs(delta) < 1e-2:
# #             break
# #     I0_meas_vals.append(n.get_current())
# #     F0_vortex_vals.append(n.free_energy())

# I_meas_vals = np.array(I_meas_vals)
# #I0_meas_vals = np.array(I0_meas_vals)
# F_vortex_vals = np.array(F_vortex_vals)
# #F0_vortex_vals = np.array(F0_vortex_vals)

# p_vortex = np.polyfit(I_meas_vals, F_vortex_vals, 2)
# #p_0 = np.polyfit(I0_meas_vals, F0_vortex_vals, 2)


# # L_N1 = p_vortex[0]
# # L_N0 = p_0[0]
# # print("L_N1 = %g, L_N0 = %g" % (L_N1, L_N0))
# # print("L_V / L_JJ = ", (L_N1/L_N0 - 1) * Nx * Ny)

# #I_vals = np.linspace(I_meas_vals[0], I_meas_vals[-1], 200)
# #fit_vals = p_vortex[0]*I_vals**2 +  p_vortex[1] * I_vals
# #print("I_meas_vals: ", I_meas_vals)
# plt.plot(I_meas_vals/Ny, F_vortex_vals, 'x', label="vortex")
# # plt.plot(I_vals, fit_vals, label="quadratic fit")
# #plt.plot(I0_meas_vals/Ny, F0_vortex_vals, 'x', label="no vortex")

# plt.legend()
# plt.grid()
# plt.show()
