import numpy as np

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
        

    def set_frustration(self, f):
        Nx = self.Nx
        Ny = self.Ny
        A_x = np.linspace(0, -(Ny-1) * f, Ny) + Ny/2 * f
        A_x = np.tile(A_x, (Nx + 1, 1))
        A_y = np.linspace(0, (Nx-1) * f, Nx) - Nx/2 * f
        A_y = np.tile(A_y, (Ny - 1, 1)).T

        self.A_x = -np.pi * A_x
        self.A_y = -np.pi * A_y

    def set_current(self, I):
        Nx = self.Nx
        Ny = self.Ny
        j = I / Ny # current per junction

        # find solution cpr_x(gamma) = j by linear fitting of cpr
        j0 = cpr_x(0)
        delta = 0.01 * np.pi
        cpr_prime = (cpr_x(delta) - j0) / delta
        gamma = (j -j0) / cpr_prime

        phi = np.linspace(gamma, Nx * gamma, Nx)
        phi = np.tile(phi, (Ny, 1)).T
        phi_r = (Nx + 1) * gamma
        
        self.phi_matrix += phi
        self.phi_r += phi_r

    def add_vortex(self, x0, y0, vorticity=1):
        return 0

    def get_current(self):
        return 0

    
    
    def free_energy(self):
        return 0

    def plot_phases(self):
        return 0

    def plot_currents(self):
        return 0


    
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
                    f_prime += cpr_y(phi_i_j - phi_matrix[i,j+1] - A_y[i,j])

                # x-component
                if i == 0:
                    f_prime += cpr_x(phi_i_j - phi_l + A_x[0, j])
                    f_prime += cpr_x(phi_i_j - phi_matrix[i+1, j] - A_x[1,j])
                elif i == Nx - 1:
                    f_prime += cpr_x(phi_i_j - phi_r- A_x[i+1, j])
                    f_prime += cpr_x(phi_i_j - phi_matrix[i-1, j] + A_x[i,j])
                else:
                    f_prime += cpr_x(phi_i_j - phi_matrix[i+1,j]- A_x[i+1, j])
                    f_prime += cpr_x(phi_i_j - phi_matrix[i-1, j]+ A_x[i,j])
                    
                new_phi = phi_i_j - epsilon * f_prime
                phi_matrix[i, j] = new_phi
                delta_phi += np.abs(phi_i_j- new_phi)
        return delta_phi

