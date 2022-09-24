#!/usr/bin/env python3

from JJAsolver.network import network
import numpy as np
import matplotlib.pyplot as plt
Nx = 16
Ny = 16

tau = 0.4

def cpr_ballistic(gamma):
    return np.sin(gamma) / np.sqrt(1 - tau * np.sin(gamma/2)**2)


def free_energy_ballistic(gamma):
    return 4 / tau * (1 - np.sqrt(1 - tau * np.sin(gamma/2)**2))

# phi_vals = np.linspace(-np.pi, np.pi, 200)
# I_vals = cpr_ballistic(phi_vals)
# F_vals = free_energy_ballistic(phi_vals)

# plt.plot(phi_vals / np.pi, I_vals, label="I")
# plt.plot(phi_vals / np.pi, F_vals, label="F")

# plt.show()

def cpr(gamma):
    return np.sin(gamma)

def free_energy(gamma):
    return 1 - np.cos(gamma)

I_vals = np.linspace(-0.02 * Ny, 0.02 * Ny, 5) + 0.09
I_meas_vals = []
F_vortex_vals = []

for I in I_vals:
    n = network(Nx, Ny, cpr_x=cpr_ballistic, cpr_y=cpr_ballistic, free_energy_x = free_energy_ballistic, free_energy_y = free_energy_ballistic)

    n.add_vortex(0.5 * (Nx - 1), 0.5 * (Nx - 1))
    n.set_current(I)
    n.set_frustration(1 / (Nx * Ny))

    for i in range(1000):
        delta =  n.optimization_step()
        print("I = %g, i = %d, delta = %g" % (I, i, delta))
        if abs(delta) < 1e-2:
            break
    I_meas_vals.append(n.get_current())
    F_vortex_vals.append(n.free_energy())

print("I_meas: ", I_meas_vals)
I0_meas_vals = []
F0_vortex_vals = []


for I in I_vals:
    n = network(Nx, Ny, cpr_x=cpr_ballistic, cpr_y=cpr_ballistic, free_energy_x = free_energy_ballistic, free_energy_y = free_energy_ballistic)

    n.set_current(I)
    n.set_frustration(0)

    for i in range(1000):
        delta =  n.optimization_step()
        print("I = %g, i = %d, delta = %g" % (I, i, delta))
        if abs(delta) < 1e-2:
            break
    I0_meas_vals.append(n.get_current())
    F0_vortex_vals.append(n.free_energy())

L_vortex = np.polyfit(I_meas_vals, F_vortex_vals, 2)[0]
L0 = np.polyfit(I0_meas_vals, F0_vortex_vals, 2)[0]

plt.plot(I_meas_vals, F_vortex_vals, label="vortex")
plt.plot(I0_meas_vals, F0_vortex_vals)
plt.legend()
plt.grid()
plt.show()
print(L_vortex, L0)
print("L1 / L0 = ", L_vortex / L0)
print("L_V / L_JJ = ", (L_vortex/L0 - 1) * Nx * Ny)
    
