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
        (T, L, energy, specific_heat)
    """
    L, J, B, T, n_steps, burn_in, thin = params
    
    # Initialize model
    model = IsingModel(L, J, B, T)
    model.initialize_spins('random')
    
    # Lists to store observables
    energies = []
    
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
    
    # Calculate observables
    energy = np.mean(energies)
    specific_heat = model.specific_heat(energies)
    
    return T, L, energy, specific_heat

def calculate_critical_temperature(J):
    """
    Calculate the critical temperature for the 2D Ising model.
    
    Parameters:
    -----------
    J : float
        Coupling constant
    
    Returns:
    --------
    float
        Critical temperature
    """
    return 2 * J / np.log(1 + np.sqrt(2))

def plot_specific_heat_vs_temperature(results, J, save_path=None):
    """
    Plot specific heat vs temperature for different lattice sizes.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    J : float
        Coupling constant
    save_path : str
        Path to save the figure
    """
    # Extract data
    data = {}
    for T, L, energy, specific_heat in results:
        if L not in data:
            data[L] = {'T': [], 'Cv': []}
        data[L]['T'].append(T)
        data[L]['Cv'].append(specific_heat)
    
    # Sort data by temperature
    for L in data:
        indices = np.argsort(data[L]['T'])
        data[L]['T'] = np.array(data[L]['T'])[indices]
        data[L]['Cv'] = np.array(data[L]['Cv'])[indices]
    
    # Calculate critical temperature
    Tc = calculate_critical_temperature(J)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot specific heat vs temperature for each lattice size
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for i, (L, values) in enumerate(sorted(data.items())):
        plt.plot(values['T'], values['Cv'], 'o-', color=colors[i], label=f'L = {L}', markersize=6)
    
    # Add vertical line at critical temperature
    plt.axvline(x=Tc, color='r', linestyle='--', label=f'$T_c$ = {Tc:.4f}')
    
    # Add labels and legend
    plt.xlabel('Temperature (T)', fontsize=14)
    plt.ylabel('Specific Heat ($C_v$)', fontsize=14)
    plt.title('Specific Heat vs Temperature in 2D Ising Model', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # Add text with critical temperature
    plt.text(0.02, 0.95, f'$T_c$ = {Tc:.4f}', transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_energy_vs_temperature(results, J, save_path=None):
    """
    Plot energy vs temperature for different lattice sizes.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    J : float
        Coupling constant
    save_path : str
        Path to save the figure
    """
    # Extract data
    data = {}
    for T, L, energy, _ in results:
        if L not in data:
            data[L] = {'T': [], 'E': []}
        data[L]['T'].append(T)
        data[L]['E'].append(energy)
    
    # Sort data by temperature
    for L in data:
        indices = np.argsort(data[L]['T'])
        data[L]['T'] = np.array(data[L]['T'])[indices]
        data[L]['E'] = np.array(data[L]['E'])[indices]
    
    # Calculate critical temperature
    Tc = calculate_critical_temperature(J)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot energy vs temperature for each lattice size
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for i, (L, values) in enumerate(sorted(data.items())):
        plt.plot(values['T'], values['E'] / (L * L), 'o-', color=colors[i], label=f'L = {L}', markersize=6)
    
    # Add vertical line at critical temperature
    plt.axvline(x=Tc, color='r', linestyle='--', label=f'$T_c$ = {Tc:.4f}')
    
    # Add labels and legend
    plt.xlabel('Temperature (T)', fontsize=14)
    plt.ylabel('Energy per Spin (E/N)', fontsize=14)
    plt.title('Energy vs Temperature in 2D Ising Model', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    # Parameters
    J = 1.0  # Coupling constant
    B = 0.0  # External magnetic field
    
    # Lattice sizes
    lattice_sizes = [10, 20, 32, 40]
    
    # Temperature range
    T_min, T_max = 1.0, 4.0
    n_temps = 30
    temperatures = np.linspace(T_min, T_max, n_temps)
    
    # Add more points near the critical temperature
    Tc = calculate_critical_temperature(J)
    T_critical_range = np.linspace(Tc - 0.3, Tc + 0.3, 15)
    temperatures = np.sort(np.unique(np.concatenate([temperatures, T_critical_range])))
    
    # Simulation parameters
    n_steps = 20000  # Number of Monte Carlo steps
    burn_in = 5000  # Number of burn-in steps
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
            params.append((L, J, B, T, n_steps, burn_in, thin))
    
    # Run simulations in parallel
    print(f"Running {len(params)} simulations using {min(cpu_count(), 8)} processes...")
    with Pool(processes=min(cpu_count(), 8)) as pool:
        results = list(tqdm(pool.imap(run_simulation, params), total=len(params)))
    
    # End timer
    end_time = time.time()
    print(f"Computation time: {end_time - start_time:.2f} seconds")
    
    # Plot specific heat vs temperature
    plot_specific_heat_vs_temperature(results, J, save_path=os.path.join(output_dir, 'specific_heat_vs_temperature.png'))
    
    # Plot energy vs temperature
    plot_energy_vs_temperature(results, J, save_path=os.path.join(output_dir, 'energy_vs_temperature.png'))
    
    # Save results to file
    np.save(os.path.join(output_dir, 'specific_heat_results.npy'), results)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 