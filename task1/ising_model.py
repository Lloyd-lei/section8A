import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
from tqdm import tqdm

class IsingModel:
    """
    A class implementing the 2D classical Ising model.
    """
    def __init__(self, L, J=1.0, B=0.0, T=1.0):
        """
        Initialize the Ising model.
        
        Parameters:
        -----------
        L : int
            Size of the lattice (L x L)
        J : float
            Coupling constant
        B : float
            External magnetic field
        T : float
            Temperature
        """
        self.L = L
        self.J = J
        self.B = B
        self.T = T
        self.beta = 1.0 / T
        
        # Initialize random spin configuration
        self.spins = np.random.choice([-1, 1], size=(L, L))
        
    def initialize_spins(self, config=None):
        """
        Initialize the spin configuration.
        
        Parameters:
        -----------
        config : str
            Configuration type: 'random', 'all_up', 'all_down', or None for random
        """
        if config == 'all_up':
            self.spins = np.ones((self.L, self.L))
        elif config == 'all_down':
            self.spins = -np.ones((self.L, self.L))
        else:  # Random configuration
            self.spins = np.random.choice([-1, 1], size=(self.L, self.L))
    
    def energy(self):
        """
        Calculate the total energy of the current spin configuration.
        
        Returns:
        --------
        float
            Total energy
        """
        # Calculate nearest-neighbor interaction energy
        # Using roll to implement periodic boundary conditions
        nn_sum = (self.spins * np.roll(self.spins, 1, axis=0) + 
                  self.spins * np.roll(self.spins, -1, axis=0) + 
                  self.spins * np.roll(self.spins, 1, axis=1) + 
                  self.spins * np.roll(self.spins, -1, axis=1))
        
        # Calculate energy from external field
        field_sum = np.sum(self.spins)
        
        # Total energy
        return -self.J * np.sum(nn_sum) / 2 - self.B * field_sum
    
    def energy_change(self, i, j):
        """
        Calculate the energy change if the spin at position (i, j) is flipped.
        
        Parameters:
        -----------
        i, j : int
            Position of the spin to flip
        
        Returns:
        --------
        float
            Energy change
        """
        L = self.L
        s = self.spins[i, j]
        
        # Calculate the sum of neighboring spins (with periodic boundary conditions)
        neighbors_sum = (self.spins[(i+1)%L, j] + 
                         self.spins[(i-1)%L, j] + 
                         self.spins[i, (j+1)%L] + 
                         self.spins[i, (j-1)%L])
        
        # Energy change = -2 * J * s * sum_neighbors - 2 * B * s
        return 2 * self.J * s * neighbors_sum + 2 * self.B * s
    
    def magnetization(self):
        """
        Calculate the magnetization of the current spin configuration.
        
        Returns:
        --------
        float
            Magnetization per spin
        """
        return np.sum(self.spins) / (self.L * self.L)
    
    def specific_heat(self, energies):
        """
        Calculate the specific heat from a list of energies.
        
        Parameters:
        -----------
        energies : list
            List of energy values
        
        Returns:
        --------
        float
            Specific heat
        """
        N = self.L * self.L
        return (self.beta**2 / N) * (np.mean(np.array(energies)**2) - np.mean(energies)**2)
    
    def magnetic_susceptibility(self, magnetizations):
        """
        Calculate the magnetic susceptibility from a list of magnetizations.
        
        Parameters:
        -----------
        magnetizations : list
            List of magnetization values
        
        Returns:
        --------
        float
            Magnetic susceptibility
        """
        N = self.L * self.L
        return (self.beta / N) * (np.mean(np.array(magnetizations)**2) - np.mean(magnetizations)**2)
    
    def visualize_spins(self, title=None, save_path=None):
        """
        Visualize the current spin configuration.
        
        Parameters:
        -----------
        title : str
            Title for the plot
        save_path : str
            Path to save the figure
        """
        # Create a custom colormap for better visualization
        cmap = colors.ListedColormap(['#1E3A8A', '#EF4444'])  # Dark blue for -1, Red for +1
        bounds = [-1.5, 0, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(self.spins, cmap=cmap, norm=norm, interpolation='nearest')
        plt.colorbar(ticks=[-1, 1], label='Spin')
        
        if title:
            plt.title(title, fontsize=14)
        else:
            plt.title(f'Ising Model L={self.L}, T={self.T}, J={self.J}, B={self.B}', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def animate_spins(self, frames, interval=100, save_path=None):
        """
        Create an animation of the spin evolution.
        
        Parameters:
        -----------
        frames : list
            List of spin configurations
        interval : int
            Interval between frames in milliseconds
        save_path : str
            Path to save the animation
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create a custom colormap
        cmap = colors.ListedColormap(['#1E3A8A', '#EF4444'])  # Dark blue for -1, Red for +1
        bounds = [-1.5, 0, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        img = ax.imshow(frames[0], cmap=cmap, norm=norm, interpolation='nearest')
        plt.colorbar(img, ax=ax, ticks=[-1, 1], label='Spin')
        
        title = ax.set_title(f'Ising Model L={self.L}, T={self.T}, J={self.J}, B={self.B}, Step: 0', fontsize=14)
        
        def update(i):
            img.set_array(frames[i])
            title.set_text(f'Ising Model L={self.L}, T={self.T}, J={self.J}, B={self.B}, Step: {i}')
            return [img, title]
        
        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)
        
        if save_path:
            ani.save(save_path, writer='pillow', fps=10)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        return ani 