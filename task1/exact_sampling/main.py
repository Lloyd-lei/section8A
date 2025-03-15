import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from itertools import product
import time

# Add parent directory to path to import IsingModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ising_model import IsingModel

def generate_all_configurations(L):
    """
    Generate all possible spin configurations for an L x L lattice.
    
    Parameters:
    -----------
    L : int
        Size of the lattice
    
    Returns:
    --------
    list
        List of all possible spin configurations
    """
    # Total number of configurations: 2^(L*L)
    n_configs = 2**(L*L)
    print(f"Generating {n_configs} configurations for L={L}...")
    
    # Generate all possible combinations of -1 and 1
    all_spins = list(product([-1, 1], repeat=L*L))
    
    # Reshape each configuration to L x L
    configurations = [np.array(config).reshape(L, L) for config in all_spins]
    
    return configurations

def calculate_energy(config, J, B):
    """
    Calculate the energy of a given spin configuration.
    
    Parameters:
    -----------
    config : numpy.ndarray
        Spin configuration
    J : float
        Coupling constant
    B : float
        External magnetic field
    
    Returns:
    --------
    float
        Energy of the configuration
    """
    L = config.shape[0]
    
    # Calculate nearest-neighbor interaction energy
    nn_sum = 0
    for i in range(L):
        for j in range(L):
            nn_sum += config[i, j] * (config[(i+1)%L, j] + config[i, (j+1)%L])
    
    # Calculate energy from external field
    field_sum = np.sum(config)
    
    # Total energy
    return -J * nn_sum - B * field_sum

def calculate_boltzmann_weights(configurations, J, B, T):
    """
    Calculate Boltzmann weights for all configurations.
    
    Parameters:
    -----------
    configurations : list
        List of all possible spin configurations
    J : float
        Coupling constant
    B : float
        External magnetic field
    T : float
        Temperature
    
    Returns:
    --------
    numpy.ndarray
        Array of Boltzmann weights
    float
        Partition function (sum of weights)
    """
    beta = 1.0 / T
    
    # Calculate energies for all configurations
    energies = np.array([calculate_energy(config, J, B) for config in configurations])
    
    # Calculate Boltzmann weights
    weights = np.exp(-beta * energies)
    
    # Calculate partition function
    Z = np.sum(weights)
    
    # Normalize weights to get probabilities
    probabilities = weights / Z
    
    return probabilities, Z, energies

def sample_from_distribution(configurations, probabilities, n_samples):
    """
    Sample configurations from the Boltzmann distribution.
    
    Parameters:
    -----------
    configurations : list
        List of all possible spin configurations
    probabilities : numpy.ndarray
        Array of probabilities for each configuration
    n_samples : int
        Number of samples to draw
    
    Returns:
    --------
    list
        List of sampled configurations
    """
    # Sample indices according to probabilities
    indices = np.random.choice(len(configurations), size=n_samples, p=probabilities)
    
    # Get the corresponding configurations
    samples = [configurations[i] for i in indices]
    
    return samples

def plot_probability_distribution(energies, probabilities, J, B, T, save_path=None):
    """
    Plot the probability distribution of energies.
    
    Parameters:
    -----------
    energies : numpy.ndarray
        Array of energies
    probabilities : numpy.ndarray
        Array of probabilities
    J : float
        Coupling constant
    B : float
        External magnetic field
    T : float
        Temperature
    save_path : str
        Path to save the figure
    """
    # Sort energies and probabilities
    sorted_indices = np.argsort(energies)
    sorted_energies = energies[sorted_indices]
    sorted_probs = probabilities[sorted_indices]
    
    # Create a visually appealing plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_energies)), sorted_probs, color='#3B82F6', alpha=0.8)
    plt.xlabel('Energy States (sorted by energy)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Probability Distribution for 2D Ising Model (L=4, J={J}, B={B}, T={T})', fontsize=14)
    plt.grid(alpha=0.3)
    
    # Add text with partition function and other info
    Z = np.sum(probabilities)
    plt.text(0.02, 0.95, f'Partition Function Z = {Z:.4f}', transform=plt.gca().transAxes, 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_samples(samples, L, J, B, T, save_dir=None):
    """
    Visualize sampled configurations.
    
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
    save_dir : str
        Directory to save the figures
    """
    # Create a custom colormap
    cmap = colors.ListedColormap(['#1E3A8A', '#EF4444'])  # Dark blue for -1, Red for +1
    bounds = [-1.5, 0, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # Create a grid of subplots
    n_samples = min(len(samples), 16)  # Show at most 16 samples
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    fig.suptitle(f'Sampled Configurations from 2D Ising Model (L={L}, J={J}, B={B}, T={T})', fontsize=16)
    
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    
    for i in range(n_samples):
        ax = axes[i]
        im = ax.imshow(samples[i], cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(f'Sample {i+1}')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[-1, 1])
    cbar.set_label('Spin')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'sampled_configurations.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    # Parameters
    L = 4  # Lattice size
    J = 1.0  # Coupling constant
    B = 0.0  # External magnetic field
    T = 1.0  # Temperature
    n_samples = 16  # Number of samples to draw
    
    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Start timer
    start_time = time.time()
    
    # Generate all possible configurations
    configurations = generate_all_configurations(L)
    
    # Calculate Boltzmann weights and partition function
    probabilities, Z, energies = calculate_boltzmann_weights(configurations, J, B, T)
    
    # Sample from the distribution
    samples = sample_from_distribution(configurations, probabilities, n_samples)
    
    # End timer
    end_time = time.time()
    print(f"Computation time: {end_time - start_time:.2f} seconds")
    
    # Plot probability distribution
    plot_probability_distribution(energies, probabilities, J, B, T, 
                                 save_path=os.path.join(output_dir, 'probability_distribution.png'))
    
    # Visualize samples
    visualize_samples(samples, L, J, B, T, save_dir=output_dir)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 