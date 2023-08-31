# 1D PiC Code for Simulating Plasmas with Google JAX

**An Imperial College London Undergraduate Research Opportunities Project by Sean Lim, supervised by Dr Aidan Crilly**

JAX enables users to solve multiple equations vectorially (with its vmap function), and compiles code in a GPU-friendly manner, making it very useful in performing Particle-in-Cell simulations efficiently.



## Choices Made

The code uses many staples in PiC codes, such as the Boris Algorithm to push particles, a triangular shape function for the pseudoparticles, a staggered Yee Grid for the EM fields, and more. A detailed explanation is given below.

The core of the simulation consists of four parts: 
<ol>
    <li> The particle pusher
    <li> Copying the particles' properties to the grid
    <li> The EM solver
    <li> Returning the EM fields' values to the particles
</ol>

The schematic of one cycle of the simulation is shown:
![diagram of one cycle of the simulation](/Images/cycle.png)

The Equations to be solved are:

### 1. The Particle Pusher
The particle pusher functions are contained in the particle_mover.py module.

The Boris algorithm staggers the position and velocity in time to make it pseudo-symplectic. The equations used are:
eqn 1
eqn 2
This was taken from [1].

### 2. Particles to Grid
These functions are contained in the particles_to_grid.py module.

Particles are taken as pseudoparticles with a weight $\Omega$ such that number density $n=\frac{N_{p}\Omega}{L}$ where $N_{p}$ is the number are pseudoparticles. This is in agreement with the 1D grid, where $\Omega$ carries an 'areal weight' on top of a normal weight (units of no. of actual particles/$m^2$). The pseudoparticles have a triangular shape function of width 2dx, as used in EPOCH [2]. This smooths out the properties on the grid to reduce numerical noise.
![shape function of particles](/Images/shapefunction.png)

The current density is found using the equation $eqn here$, as done in Villasenor and Buneman [3] and EPOCH [4]. This is done by sweeping the grid from left to right. In one timestep, each particle can travel at most 1 cell (since the simulation becomes unstable as $\frac{dx}{dt}\to3x10^8$), so with the shape function, we only need to sweep between -3 to 2 spaces from the particle's initial cell, where the first cell is empty as the starting point for the sweeping.
![current sweeping method](/images/current_sweep.png)

The current in y and z direction use $j=nqv$, or more precisely $j=\frac{N_p\rho v}$.

### 3. The EM solver
The EM solver is contained in the EM_solver.py module.A staggered Yee grid is used, where E-fields are defined on right-side cell edges and B-fields are defined on cell centres. 
![yee grid](/Images/yee_grid.png)

The equations to solve are $Ampere$ and $Faraday$. We do not solve Gauss' Law directly, as Poisson solvers can lead to numerical issues, and Gauss' Law is automatically obeyed if we use the charge conservation equation, provided Gauss' Law was satisfied at the start.

In a 1D PiC code, $de/dt = curlB$ and $db/dt = curlE$ solve transverse EM wave components, while $j_x$ updates longitudinal E-field, and $j_y$ and $j_z$ feed into the equations to create EM waves.

The solver takes 2 steps of dt/2 each, first updating the E-field before the B-field, then vice versa. 


### 4. Fields to Particles
The function to return the fields to the particles is found in the particle_mover.py module. Taking into account the particle spanning several cells due to its shape, the total force it experiences adding each part is $eqn$ [2].

### Boundary Conditions
Boundary conditions are found in the boundary_conditions.py module.

Boundary conditions are specified by moving the particles and changing their velocities as desired after they have left the box, and applying ghost cells for fields.

The code supports 3 particle BC modes, and 3 field BC modes, to be specified on each side. They are displayed in this table :
Particle table:
mode/BC/new particle position/new particle velocity/ghost cell experienced by particle (in fields to particles)
![table of particle BC modes](/Images/part_BC_table.png)
Note the need to use 2 ghost cells on the left due to the leftmost edges of particles in the first half cell undefined when using the staggered grid  while finding E-field experienced.
note y and z BCs are always periodic

Field table:
![table of field BC modes](Images/field_BC_table.png)

### Diagnostics
Apart from the core solver, there is an additional diagnostics.py module for returning useful output. In it are functions to find the system's total kinetic energy, E-field density, B-field density, temperature at each cell and velocity histogram. These are returned in the output.

### The simulation.py module
Finally, the simulation.py module puts it all together. It defines one step in the cycle, which is called in an n_cycles function so we can take many steps before performing diagnosis for long simulations where timescales of phenomenon are much longer than the dt required to maintain stability ($\frac{dx}{dt}<3x10^8$). 

This outermost function n_cycles, as well as any other outermost functions in the simulation function, are decorated with @jit for jax to compile the function and any other function called inside it, as well as block_until_ready statements placed where necessary to run on GPUs. 

## Using the Simulation

### Initialising the Simulation

In the simulation.py module, the main function 'simulation' is called with arguments (steps per snapshot, total steps, ICs, ext_fields,dx,dt,BC_left,BC_right). For neatness, the arguments in simulation are wrapped to be called, and then unwrapped within the simulation function. 

For N particles and M cells,
<ol>
    <li>ICs is an initial conditions sequence containing 3 sequences, (box_size, particle_parameters, fields). 
      <ol>
        <li>box_size contains (Lx,Ly,Lz). </li>
        <li>particle_parameters contains (particle positions, velocities, qs, ms, q/ms, number of each pseudospecies, weights).
          <ol>
            <li>Particle positions and velocities should both be an Nx3 array.</li>
            <li>Qs,ms and q/ms should all be Nx1 arrays. Note it has to be Nx1 and not N to be compatible with JAX's vmap function. Also note the use of q/m to reduce floating point errors as JAX is single-precision.</li>
            <li>number of each pseudospecies should be an iterable of the number of each pseudospecies, e.g. if I had 5000 electrons and 1000 protons, it would be (5000,1000)</li>
            <li>weights should be an integer/float
          </ol>
        </li>
        <li> fields contains (array of E-fields,array of B-fields) where both are Mx3 arrays specifying initial E- and B- fields. In EM_solver.py there is a function, find_E0_by_matrix to help check if the initial conditions are correct (this may provide the wrong answer by a constant, hence it is recommended to manually calculate the E-field values). </li>
      </ol>
    </li>
    <li>ext_fields contains (array of E-fields,array of B-fields) where both are Mx3 arrays specifying external E- and B- fields.

Note the staggered grid when dealing with E-fields, which are defined on the edges of cells.

Some precautions: dx should be on the order of Debye length to avoid numerical heating. Make functions as smooth as possible, eg for EM waves, apply gaussian envelope or ensure no cutoff of EM waves. For particles, ensure left and right side of system match.


### Output
CSV file names are 'time.csv','kinetic_energy.csv','E_x.csv','E_y.csv','E_z.csv','E_energy_densities.csv','B_x.csv','B_y.csv','B_z.csv','B_energy_densities.csv','chargedens.csv','electron_no_densities.csv','ion_no_densities.csv','electron_temp.csv','ion_temp.csv'


## Examples

In the examples folder there are some example simulations showing typical plasma behaviour. Includes plasma oscillations, plasma waves, plasma waves, 2-stream instability, weibel instability, hybrid oscillations.

# References
[1] Why is Boris so good
[2] EPOCH shape function
[3] V&B
[4] EPOCH current
[5] Silver-Muller BCs