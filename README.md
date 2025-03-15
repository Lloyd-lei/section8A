# 2D Classical Ising Model with MCMC

This project implements the 2D classical Ising model using Monte Carlo Markov Chain (MCMC) methods. The project is structured into several tasks, each exploring different aspects of the Ising model.

## Project Structure

```
.
├── task1/
│   ├── exact_sampling/         # Exact sampling for L=4 system
│   ├── gibbs_sampling/         # Gibbs sampling implementation
│   ├── phase_transition/       # Magnetization vs temperature (phase transition)
│   ├── magnetic_field/         # Magnetic field dependence
│   ├── specific_heat/          # Specific heat calculations
│   └── magnetic_susceptibility/ # Magnetic susceptibility calculations
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Tasks Overview

1. **Exact Sampling**: Implement exact sampling from the Boltzmann distribution for a small (L=4) Ising model.
2. **Gibbs Sampling**: Implement a Gibbs sampler for the Ising model.
3. **Phase Transition**: Visualize the ferromagnetic to paramagnetic phase transition.
4. **Magnetic Field Dependence**: Study the effect of external magnetic field on magnetization.
5. **Specific Heat**: Calculate and visualize the specific heat as a function of temperature.
6. **Magnetic Susceptibility**: Calculate and visualize the magnetic susceptibility.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy
- tqdm (for progress bars)

## How to Run

Each task has its own directory with a main script that can be run independently:

```bash
# For example, to run the exact sampling task:
python task1/exact_sampling/main.py
``` 