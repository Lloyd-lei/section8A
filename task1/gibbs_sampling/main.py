import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from tqdm import tqdm

# Add parent directory to path to import IsingModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ising_model import IsingModel

class GibbsSampler:
    """
    Gibbs sampler for the 2D Ising model.
    """
    def __init__(self, L, J=1.0, B=0.0, T=1.0):
        """
        Initialize the Gibbs sampler.
        
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
        self.model = IsingModel(L, J, B, T)
        self.L = L
        self.J = J
        self.B = B
        self.T = T
        self.beta = 1.0 / T
        
    def initialize_spins(self, config=None):
        """
        Initialize the spin configuration.
        
        Parameters:
        -----------
        config : str
            Configuration type: 'random', 'all_up', 'all_down', or None for random
        """
        self.model.initialize_spins(config)
    
    def conditional_probability(self, i, j):
        """
        Calculate the conditional probability P(S_ij = 1 | S_-ij).
        
        Parameters:
        -----------
        i, j : int
            Position of the spin
        
        Returns:
        --------
        float
            Probability that the spin is +1 given all other spins
        """
        # Temporarily flip the spin to -1 (if it's not already -1)
        original_spin = self.model.spins[i, j]
        if original_spin == 1:
            self.model.spins[i, j] = -1
            energy_minus = self.model.energy()
            self.model.spins[i, j] = 1
            energy_plus = self.model.energy()
        else:
            self.model.spins[i, j] = 1
            energy_plus = self.model.energy()
            self.model.spins[i, j] = -1
            energy_minus = self.model.energy()
        
        # Restore the original spin
        self.model.spins[i, j] = original_spin
        
        # Calculate energy difference
        delta_E = energy_plus - energy_minus
        
        # Calculate probability using Boltzmann factor
        p_plus = 1.0 / (1.0 + np.exp(self.beta * delta_E))
        
        return p_plus
    
    def gibbs_step(self):
        """
        Perform one Gibbs sampling step (update all spins once).
        """
        L = self.L
        
        # Update each spin sequentially
        for i in range(L):
            for j in range(L):
                # Calculate conditional probability
                p_plus = self.conditional_probability(i, j)
                
                # Sample new spin value
                self.model.spins[i, j] = 1 if np.random.random() < p_plus else -1
    
    def run_chain(self, n_steps, burn_in=0, thin=1):
        """
        Run the Gibbs sampler for n_steps.
        
        Parameters:
        -----------
        n_steps : int
            Number of Gibbs steps to perform
        burn_in : int
            Number of initial steps to discard
        thin : int
            Thinning factor (store every 'thin' samples)
        
        Returns:
        --------
        list
            List of sampled configurations
        list
            List of energies
        list
            List of magnetizations
        """
        # Initialize lists to store samples and observables
        samples = []
        energies = []
        magnetizations = []
        
        # Run the chain
        for step in tqdm(range(n_steps + burn_in), desc="Running Gibbs sampler"):
            # Perform one Gibbs step
            self.gibbs_step()
            
            # Store samples and observables after burn-in
            if step >= burn_in and (step - burn_in) % thin == 0:
                samples.append(self.model.spins.copy())
                energies.append(self.model.energy())
                magnetizations.append(self.model.magnetization())
        
        return samples, energies, magnetizations

def plot_observables(energies, magnetizations, J, B, T, burn_in, save_dir=None):
    """
    Plot the evolution of energy and magnetization.
    
    Parameters:
    -----------
    energies : list
        List of energies
    magnetizations : list
        List of magnetizations
    J : float
        Coupling constant
    B : float
        External magnetic field
    T : float
        Temperature
    burn_in : int
        Number of burn-in steps
    save_dir : str
        Directory to save the figures
    """
    # Convert to numpy arrays
    energies = np.array(energies)
    magnetizations = np.array(magnetizations)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot energy
    ax1.plot(energies, color='#3B82F6', linewidth=1.5)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title(f'Energy Evolution (J={J}, B={B}, T={T})', fontsize=14)
    ax1.grid(alpha=0.3)
    
    # Plot magnetization
    ax2.plot(magnetizations, color='#EF4444', linewidth=1.5)
    ax2.set_xlabel('Gibbs Steps', fontsize=12)
    ax2.set_ylabel('Magnetization', fontsize=12)
    ax2.set_title(f'Magnetization Evolution (J={J}, B={B}, T={T})', fontsize=14)
    ax2.grid(alpha=0.3)
    
    # Add vertical line for burn-in
    if burn_in > 0:
        ax1.axvline(x=burn_in, color='k', linestyle='--', alpha=0.7)
        ax2.axvline(x=burn_in, color='k', linestyle='--', alpha=0.7)
        ax1.text(burn_in + 5, ax1.get_ylim()[0] + 0.1 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]), 
                'Burn-in', fontsize=10)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'observables_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_autocorrelation(data, max_lag=100, title=None, save_path=None):
    """
    Plot the autocorrelation function of a time series.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Time series data
    max_lag : int
        Maximum lag to compute
    title : str
        Title for the plot
    save_path : str
        Path to save the figure
    """
    # Compute autocorrelation
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    
    # Normalize data
    normalized_data = (data - mean) / np.sqrt(var)
    
    # Compute autocorrelation for lags up to max_lag
    lags = range(max_lag)
    autocorr = np.zeros(max_lag)
    
    for lag in lags:
        autocorr[lag] = np.mean(normalized_data[:(n-lag)] * normalized_data[lag:])
    
    # Plot autocorrelation
    plt.figure(figsize=(10, 6))
    plt.plot(lags, autocorr, 'o-', color='#3B82F6', markersize=4, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('Autocorrelation', fontsize=12)
    
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title('Autocorrelation Function', fontsize=14)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_chain_evolution(samples, L, J, B, T, n_frames=10, save_path=None):
    """
    Visualize the evolution of the Markov chain.
    
    Parameters:
    -----------
    samples : list
        List of sampled configurations
    L : int
        Size of the lattice
    J : float
        Coupling constant
    B : float
        External magnetic field
    T : float
        Temperature
    n_frames : int
        Number of frames to show
    save_path : str
        Path to save the animation
    """
    # Select frames evenly spaced throughout the chain
    n_samples = len(samples)
    indices = np.linspace(0, n_samples-1, n_frames, dtype=int)
    selected_samples = [samples[i] for i in indices]
    
    # Create a temporary IsingModel instance for visualization
    model = IsingModel(L, J, B, T)
    
    # Create animation
    model.animate_spins(selected_samples, interval=500, save_path=save_path)

def main():
    # Parameters
    L = 16  # Lattice size (reduced from 32 to 16)
    J = 1.0  # Coupling constant
    B = 0.0  # External magnetic field
    T = 2.27  # Temperature (close to critical temperature)
    n_steps = 1000  # Number of Gibbs steps (reduced from 10000 to 1000)
    burn_in = 100  # Number of burn-in steps (reduced from 1000 to 100)
    thin = 10  # Thinning factor
    
    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Gibbs sampler
    sampler = GibbsSampler(L, J, B, T)
    
    # Initialize with random spins
    sampler.initialize_spins('random')
    
    # Start timer
    start_time = time.time()
    
    # Run the Gibbs sampler
    samples, energies, magnetizations = sampler.run_chain(n_steps, burn_in, thin)
    
    # End timer
    end_time = time.time()
    print(f"Computation time: {end_time - start_time:.2f} seconds")
    
    # Plot observables
    plot_observables(energies, magnetizations, J, B, T, burn_in=0, save_dir=output_dir)
    
    # Plot autocorrelation of magnetization
    plot_autocorrelation(magnetizations, max_lag=100, 
                         title=f'Autocorrelation of Magnetization (L={L}, J={J}, B={B}, T={T})',
                         save_path=os.path.join(output_dir, 'magnetization_autocorrelation.png'))
    
    # Visualize chain evolution
    visualize_chain_evolution(samples, L, J, B, T, n_frames=16, 
                             save_path=os.path.join(output_dir, 'chain_evolution.gif'))
    
    # Visualize final configuration
    sampler.model.visualize_spins(title=f'Final Configuration (L={L}, J={J}, B={B}, T={T})',
                                 save_path=os.path.join(output_dir, 'final_configuration.png'))
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 