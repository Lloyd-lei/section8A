import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Add parent directory to path to import IsingModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ising_model import IsingModel

def run_simulation(params):
    """
    Run a simulation for a given set of parameters.
    
    Parameters:
    -----------
    params : tuple
        (L, J, B, T, n_steps, burn_in, thin)
    
    Returns:
    --------
    tuple
        (B, L, T, magnetization, energy)
    """
    L, J, B, T, n_steps, burn_in, thin = params
    
    # Initialize model
    model = IsingModel(L, J, B, T)
    model.initialize_spins('random')
    
    # Lists to store observables
    energies = []
    magnetizations = []
    
    # Run simulation
    for step in range(n_steps + burn_in):
        # Perform one Monte Carlo step (sweep through all spins)
        for _ in range(L * L):
            # Select a random spin
            i, j = np.random.randint(0, L, 2)
            
            # Calculate energy change if this spin is flipped
            delta_E = model.energy_change(i, j)
            
            # Metropolis acceptance criterion
            if delta_E <= 0 or np.random.random() < np.exp(-model.beta * delta_E):
                model.spins[i, j] *= -1
        
        # Store observables after burn-in
        if step >= burn_in and (step - burn_in) % thin == 0:
            energies.append(model.energy())
            magnetizations.append(model.magnetization())
    
    # Calculate observables
    mag = np.mean(magnetizations)
    energy = np.mean(energies)
    
    return B, L, T, mag, energy

def plot_magnetization_vs_field(results, save_path=None):
    """
    Plot magnetization vs magnetic field for different lattice sizes and temperatures.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    save_path : str
        Path to save the figure
    """
    # Extract data
    data = {}
    for B, L, T, mag, energy in results:
        key = (L, T)
        if key not in data:
            data[key] = {'B': [], 'mag': []}
        data[key]['B'].append(B)
        data[key]['mag'].append(mag)
    
    # Sort data by magnetic field
    for key in data:
        indices = np.argsort(data[key]['B'])
        data[key]['B'] = np.array(data[key]['B'])[indices]
        data[key]['mag'] = np.array(data[key]['mag'])[indices]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot magnetization vs magnetic field for each lattice size and temperature
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for i, ((L, T), values) in enumerate(sorted(data.items())):
        plt.plot(values['B'], values['mag'], 'o-', color=colors[i], label=f'L = {L}, T = {T}', markersize=6)
    
    # Add labels and legend
    plt.xlabel('Magnetic Field (B)', fontsize=14)
    plt.ylabel('Magnetization (M)', fontsize=14)
    plt.title('Magnetization vs Magnetic Field in 2D Ising Model', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_hysteresis(results, save_path=None):
    """
    Plot hysteresis loop for the Ising model.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    save_path : str
        Path to save the figure
    """
    # Extract data for the largest lattice size and lowest temperature
    L_max = max(L for _, L, _, _, _ in results)
    T_min = min(T for _, L, T, _, _ in results if L == L_max)
    
    # Filter data for the largest lattice size and lowest temperature
    data = {'B': [], 'mag': []}
    for B, L, T, mag, _ in results:
        if L == L_max and T == T_min:
            data['B'].append(B)
            data['mag'].append(mag)
    
    # Sort data by magnetic field
    indices = np.argsort(data['B'])
    data['B'] = np.array(data['B'])[indices]
    data['mag'] = np.array(data['mag'])[indices]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot hysteresis loop
    plt.plot(data['B'], data['mag'], 'o-', color='#3B82F6', markersize=6, linewidth=2)
    
    # Add labels
    plt.xlabel('Magnetic Field (B)', fontsize=14)
    plt.ylabel('Magnetization (M)', fontsize=14)
    plt.title(f'Hysteresis Loop for 2D Ising Model (L = {L_max}, T = {T_min})', fontsize=16)
    plt.grid(alpha=0.3)
    
    # Add horizontal and vertical lines at zero
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_configurations(results, output_dir):
    """
    Visualize spin configurations for different magnetic fields.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    output_dir : str
        Directory to save the figures
    """
    # Extract data for the largest lattice size and lowest temperature
    L_max = max(L for _, L, _, _, _ in results)
    T_min = min(T for _, L, T, _, _ in results if L == L_max)
    
    # Select a few magnetic field values
    B_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    # Run simulations for these specific values
    for B in B_values:
        # Initialize model
        model = IsingModel(L_max, J=1.0, B=B, T=T_min)
        model.initialize_spins('random')
        
        # Run simulation
        n_steps = 10000
        burn_in = 1000
        
        for step in tqdm(range(n_steps + burn_in), desc=f"Simulating B = {B}"):
            # Perform one Monte Carlo step
            for _ in range(L_max * L_max):
                i, j = np.random.randint(0, L_max, 2)
                delta_E = model.energy_change(i, j)
                if delta_E <= 0 or np.random.random() < np.exp(-model.beta * delta_E):
                    model.spins[i, j] *= -1
        
        # Visualize final configuration
        model.visualize_spins(title=f'Spin Configuration (L = {L_max}, T = {T_min}, B = {B})',
                             save_path=os.path.join(output_dir, f'configuration_B_{B}.png'))

def main():
    # Parameters
    J = 1.0  # Coupling constant
    
    # Lattice sizes
    lattice_sizes = [16, 32]
    
    # Temperatures
    temperatures = [1.0, 2.0, 3.0]
    
    # Magnetic field range
    B_min, B_max = -2.0, 2.0
    n_fields = 20
    magnetic_fields = np.linspace(B_min, B_max, n_fields)
    
    # Simulation parameters
    n_steps = 10000  # Number of Monte Carlo steps
    burn_in = 1000  # Number of burn-in steps
    thin = 10  # Thinning factor
    
    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Start timer
    start_time = time.time()
    
    # Prepare parameters for parallel execution
    params = []
    for L in lattice_sizes:
        for T in temperatures:
            for B in magnetic_fields:
                params.append((L, J, B, T, n_steps, burn_in, thin))
    
    # Run simulations in parallel
    print(f"Running {len(params)} simulations using {min(cpu_count(), 8)} processes...")
    with Pool(processes=min(cpu_count(), 8)) as pool:
        results = list(tqdm(pool.imap(run_simulation, params), total=len(params)))
    
    # End timer
    end_time = time.time()
    print(f"Computation time: {end_time - start_time:.2f} seconds")
    
    # Plot magnetization vs magnetic field
    plot_magnetization_vs_field(results, save_path=os.path.join(output_dir, 'magnetization_vs_field.png'))
    
    # Plot hysteresis loop
    plot_hysteresis(results, save_path=os.path.join(output_dir, 'hysteresis_loop.png'))
    
    # Visualize configurations for different magnetic fields
    visualize_configurations(results, output_dir)
    
    # Save results to file
    np.save(os.path.join(output_dir, 'magnetic_field_results.npy'), results)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 