import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import scipy.optimize

def jj_cpr_ballistic(gamma, tau):
    return np.sin(gamma) / np.sqrt(1 - tau * np.sin(gamma/2)**2)

def jj_free_energy_ballistic(gamma, tau):
    return 4 / tau * (1 - np.sqrt(1 - tau * np.sin(gamma/2)**2))

class network:
    def __init__(self, Nx, Ny, *, cpr_x, cpr_y, free_energy_x, free_energy_y):
        self.Nx = Nx
        self.Ny = Ny
        self.cpr_x = cpr_x
        self.cpr_y = cpr_y
        self.free_energy_x = free_energy_x
        self.free_energy_y = free_energy_y

        
        self.island_x_coords, self.island_y_coords = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
        self.phi_matrix = np.zeros((Nx, Ny))
        self.phi_r = 0
        self.phi_l = 0
        self.set_frustration(0)
        
    def reset_network(self):
        self.phi_matrix *= 0
        self.phi_r = 0
        self.phi_l = 0
        self.set_frustration(0)

    def set_random_state(self):
        self.phi_matrix = 2 * np.pi * numpy.random.rand(self.Nx, self.Ny)
        self.phi_r = 2 * np.pi * numpy.random.rand()
        self.phi_l = 2 * np.pi * numpy.random.rand()
        
    def set_frustration(self, f):
        Nx = self.Nx
        Ny = self.Ny
        A_x = np.linspace(0, -(Ny-1) * f, Ny) + (Ny - 1)/2 * f
        A_x = np.tile(A_x, (Nx + 1, 1))
        A_y = np.linspace(0, (Nx-1) * f, Nx) - (Nx-1)/2 * f
        A_y = np.tile(A_y, (Ny - 1, 1)).T

        self.A_x = -np.pi * A_x
        self.A_y = -np.pi * A_y

    def set_current(self, I):
        Nx = self.Nx
        Ny = self.Ny
        j = I / Ny # current per junction

        def f(x):
            return self.cpr_x(x) - j

        gamma = scipy.optimize.brentq(f, -np.pi/2, np.pi/2, rtol=1e-6, xtol=1e-6)
        
        print("gamma = %g pi" % (gamma/ np.pi))

        phi = np.linspace(gamma, Nx * gamma, Nx)
        phi = np.tile(phi, (Ny, 1)).T
        phi_r = (Nx + 1) * gamma
        
        self.phi_matrix += phi
        self.phi_r += phi_r

    def add_vortex(self, x0, y0, vorticity=1):
        self.phi_matrix += np.arctan2(y0 - self.island_y_coords,
                                      x0 - self.island_x_coords)
        self.phi_r += np.pi


    def get_gamma_matrices(self):
        Nx = self.Nx
        Ny = self.Ny
        gamma_x = np.zeros((Nx + 1, Ny))
        gamma_y = np.zeros((Nx, Ny - 1))
        
        gamma_x += self.A_x
        gamma_y += self.A_y

        phi_matrix = self.phi_matrix
        gamma_x[1:-1,:] += phi_matrix[1:,:] - phi_matrix[:-1,:]
        gamma_x[0,:] += phi_matrix[0,:] - self.phi_l
        gamma_x[-1,:] += self.phi_r - phi_matrix[-1,:]
        
        gamma_y += phi_matrix[:,1:] - phi_matrix[:,:-1]
        return (gamma_x, gamma_y)
        
    def get_current_matrices(self):
        gamma_x, gamma_y = self.get_gamma_matrices()
        return self.cpr_x(gamma_x), self.cpr_y(gamma_y)
    
    def get_current(self):
        I_x, I_y = self.get_current_matrices()
        return np.sum(I_x[0,:])
    
    def free_energy(self):
        gamma_x, gamma_y = self.get_gamma_matrices()
        return np.sum(self.free_energy_x(gamma_x)) + \
            np.sum(self.free_energy_y(gamma_y))

    def plot_phases(self):
        plt.clf()
        m = self.phi_matrix.copy()
        m = np.flip(m, axis=1)
        m = np.swapaxes(m, 0, 1)
        plt.imshow(m/np.pi, aspect='equal', cmap='gray')
        plt.colorbar(format="%.1f", label='φ')


    def plot_currents(self):
        Nx = self.Nx
        Ny = self.Ny
        x_currents, y_currents = self.get_current_matrices()
        
        x_current_xcoords, x_current_ycoords = np.meshgrid(np.arange(Nx+1), np.arange(Ny), indexing="ij")
        x_current_xcoords = x_current_xcoords.astype('float64')
        x_current_ycoords = x_current_ycoords.astype('float64')
        x_current_xcoords -= 0.5
        y_current_xcoords, y_current_ycoords = np.meshgrid(np.arange(Nx), np.arange(Ny-1), indexing="ij")
        y_current_xcoords = y_current_xcoords.astype('float64')
        y_current_ycoords = y_current_ycoords.astype('float64')
        y_current_ycoords += 0.5
        plt.clf()
        plt.quiver(x_current_xcoords, x_current_ycoords,
           x_currents, np.zeros(x_currents.shape),
           pivot='mid', units='width', scale=5*Nx, width=1/(30*Nx))
        plt.quiver(y_current_xcoords, y_current_ycoords,
           np.zeros(y_currents.shape), y_currents,
           pivot='mid', units='width', scale=5*Nx, width=1/(30*Nx))
        plt.scatter(self.island_x_coords, self.island_y_coords, marker='s', c='b', s=5)
    
        

    def optimization_step(self):
        # minimize free energy f(phi) using Newton's method
        # phi -> phi - ε f'(phi)
        
        Nx = self.Nx
        Ny = self.Ny
        phi_matrix = self.phi_matrix
        phi_l = self.phi_l
        phi_r = self.phi_r
        
        A_x = self.A_x
        A_y = self.A_y
        cpr_x = self.cpr_x
        cpr_y = self.cpr_y

        epsilon = 0.5
        delta_phi = 0

        for i in range(Nx):
            for j in range(Ny):
                f_prime = 0
                phi_i_j = phi_matrix[i,j]
                # y-component
                if j > 0:
                    f_prime += cpr_y(phi_i_j - phi_matrix[i,j-1] + A_y[i, j-1])
                if j < Ny - 1:
                    f_prime += -cpr_y(-phi_i_j + phi_matrix[i,j+1] + A_y[i,j])

                # x-component
                if i == 0:
                    f_prime += cpr_x(phi_i_j - phi_l + A_x[0, j])
                    f_prime += -cpr_x(-phi_i_j + phi_matrix[i+1, j] + A_x[1,j])
                elif i == Nx - 1:
                    f_prime += -cpr_x(-phi_i_j + phi_r + A_x[i+1, j])
                    f_prime += cpr_x(phi_i_j - phi_matrix[i-1, j] + A_x[i,j])
                else:
                    f_prime += -cpr_x(-phi_i_j + phi_matrix[i+1,j]+ A_x[i+1, j])
                    f_prime += cpr_x(phi_i_j - phi_matrix[i-1, j]+ A_x[i,j])
                    
                new_phi = phi_i_j - epsilon * f_prime
                phi_matrix[i, j] = new_phi
                delta_phi += np.abs(phi_i_j- new_phi)
        return delta_phi

