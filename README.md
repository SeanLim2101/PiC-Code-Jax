# 1D PiC Code for Simulating Plasmas with Google JAX

**An Imperial College London Undergraduate Research Opportunities Project by Sean Lim, supervised by Dr Aidan Crilly**

JAX enables users to solve multiple equations vectorially (with its vmap function), and compiles code in a GPU-friendly manner, making it very useful in performing Particle-in-Cell simulations efficiently.



## Something about PiC codes and choices made in this code?

Stuff about Boris, yee grid, tldr averaging good

Stuff about 1d to reduce noise when histogramming 

Stuff about solving j using charge conservation eqn & feeding into Maxwell eqn instead of gauss’ law to ensure charge conservation and not solve gauss’ law because quote “poisson solvers are huge faffs”, must ensure gauss’ law obeyed at t=0

Insert diagram of one cycle?

Note: only periodic BCs supported currently



## Modules

### particle_mover.py

This module contains functions to solve equations of motions of particles. It uses the Boris algorithm, staggering position and velocity in time. It also includes a function for moving the particles when they leave the boundary, and a function for finding the E and B fields based on the particle’s position.

### particles_to_grid.py

This module contains functions to write the system’s charge density and current density onto the grid based on the particles’ positions. Particles have a triangular shape function spanning 2dx.

### EM_solver.py

This module contains functions for solving Maxwell’s Equations. This is done in 2 steps of dt/2 each, first updating the E-field before the B-field, then vice versa, because averaging good. It also contains a function, find_E0_by_matrix to help check if the initial conditions are correct (this may provide the wrong answer by a constant, hence it is recommended to manually calculate the E-field values).

### diagnostics.py

Contains functions to find system properties eg total kinetic energy, momentum, temperature at each cell point, etc.

### simulation_module.py

Simulation function is called with arguments (steps per snapshot, total steps, ICs, ext_fields, dx,dt). For neatness, the arguments in simulation are wrapped into an initial conditions sequence containing 3 sequences, (box size, particle parameters, fields). Box size contains (Lx,Ly,Lz). Particle parameters contains (particle initial positions, velocities, qs, ms, q/ms, the number of pseudo electrons, and the particle weights). Fields contains (array of E-fields,array of B-fields). ext_fields also contains (array of E-fields,array of B-fields).

It has 2 options to save data, either by returning a dictionary, or by saving into a set of csv files. This is specified with the argument write_to_file (default false)

The dictionary keys are:

{'Time’,’Kinetic Energy’,’Positions’,’Velocities’,’E-fields’,’E-field Energy’,’B-fields’,’B-field Energy’, 'Charge Densities’,’Temperature'}

Temperature is a tuple containing electron and ion temperature (so its shape is Tx2xM)

CSV file names are 'time.csv','kinetic_energy.csv','E_x.csv','E_y.csv','E_z.csv','E_energy_densities.csv','B_x.csv','B_y.csv','B_z.csv','B_energy_densities.csv','chargedens.csv','electron_no_densities.csv','ion_no_densities.csv','electron_temp.csv','ion_temp.csv'



## Initialising the Simulation

For N particles and M cells,
Grid should be a length M array containing the centre of each cell.
Particle initial positions and velocities should be an Nx3 array.
Qs,ms and q/ms are Nx1 arrays.
E-fields and B-fields are Mx3 arrays.

Some precautions: dx should be on the order of Debye length to avoid numerical heating. Make functions as smooth as possible, eg for EM waves, apply gaussian envelope or ensure no cutoff of EM waves. For particles, ensure left and right side of system match.


## Examples

In the examples folder there are some example simulations showing typical plasma behaviour. Includes plasma oscillations, plasma waves, plasma waves, 2-stream instability, weibel instability, hybrid oscillations.



Current problems: 

plasma oscillations: ke increasing, will test further next week
& test and add other plasma effects