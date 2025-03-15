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
        (T, L, magnetization, susceptibility)
    """
    L, J, B, T, n_steps, burn_in, thin = params
    
    # Initialize model
    model = IsingModel(L, J, B, T)
    model.initialize_spins('random')
    
    # Lists to store observables
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
            magnetizations.append(model.magnetization())
    
    # Calculate observables
    mag = np.mean(magnetizations)
    susceptibility = model.magnetic_susceptibility(magnetizations)
    
    return T, L, mag, susceptibility

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

def plot_susceptibility_vs_temperature(results, J, save_path=None):
    """
    Plot magnetic susceptibility vs temperature for different lattice sizes.
    
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
    for T, L, mag, susceptibility in results:
        if L not in data:
            data[L] = {'T': [], 'chi': []}
        data[L]['T'].append(T)
        data[L]['chi'].append(susceptibility)
    
    # Sort data by temperature
    for L in data:
        indices = np.argsort(data[L]['T'])
        data[L]['T'] = np.array(data[L]['T'])[indices]
        data[L]['chi'] = np.array(data[L]['chi'])[indices]
    
    # Calculate critical temperature
    Tc = calculate_critical_temperature(J)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot susceptibility vs temperature for each lattice size
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for i, (L, values) in enumerate(sorted(data.items())):
        plt.plot(values['T'], values['chi'], 'o-', color=colors[i], label=f'L = {L}', markersize=6)
    
    # Add vertical line at critical temperature
    plt.axvline(x=Tc, color='r', linestyle='--', label=f'$T_c$ = {Tc:.4f}')
    
    # Add labels and legend
    plt.xlabel('Temperature (T)', fontsize=14)
    plt.ylabel('Magnetic Susceptibility ($\\chi$)', fontsize=14)
    plt.title('Magnetic Susceptibility vs Temperature in 2D Ising Model', fontsize=16)
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

def plot_susceptibility_vs_field(results, save_path=None):
    """
    Plot magnetic susceptibility vs magnetic field for different lattice sizes.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    save_path : str
        Path to save the figure
    """
    # Extract data for the largest lattice size
    L_max = max(L for _, L, _, _ in results)
    
    # Filter data for the largest lattice size
    data = {'T': [], 'chi': []}
    for T, L, mag, susceptibility in results:
        if L == L_max:
            data['T'].append(T)
            data['chi'].append(susceptibility)
    
    # Sort data by temperature
    indices = np.argsort(data['T'])
    data['T'] = np.array(data['T'])[indices]
    data['chi'] = np.array(data['chi'])[indices]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot susceptibility vs temperature
    plt.plot(data['T'], data['chi'], 'o-', color='#3B82F6', markersize=6, linewidth=2)
    
    # Add labels
    plt.xlabel('Temperature (T)', fontsize=14)
    plt.ylabel('Magnetic Susceptibility ($\\chi$)', fontsize=14)
    plt.title(f'Magnetic Susceptibility vs Temperature for L = {L_max}', fontsize=16)
    plt.grid(alpha=0.3)
    
    # Calculate critical temperature
    Tc = calculate_critical_temperature(1.0)  # J = 1.0
    plt.axvline(x=Tc, color='r', linestyle='--', label=f'$T_c$ = {Tc:.4f}')
    
    plt.legend(fontsize=12)
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
    
    # Plot susceptibility vs temperature
    plot_susceptibility_vs_temperature(results, J, save_path=os.path.join(output_dir, 'susceptibility_vs_temperature.png'))
    
    # Plot susceptibility vs field for the largest lattice size
    plot_susceptibility_vs_field(results, save_path=os.path.join(output_dir, 'susceptibility_vs_field.png'))
    
    # Save results to file
    np.save(os.path.join(output_dir, 'magnetic_susceptibility_results.npy'), results)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 