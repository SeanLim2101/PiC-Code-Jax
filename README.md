# 1D PiC Code for Simulating Plasmas with Google JAX

**An Imperial College London Undergraduate Research Opportunities Project by Sean Lim, supervised by Dr Aidan Crilly**

## Using the Simulation

### Initialising the Simulation

In the simulation.py module, the main function 'simulation' is called with arguments (steps per snapshot, total steps, ICs, ext_fields,dx,dt,BCs). For neatness, the arguments in simulation should be wrapped when called, and then unwrapped within the simulation function. 

For N particles and M cells (defined by an $M$ length array of cell centres), where all arrays should be JAX DeviceArrays,
<ol>
    <li>ICs is an initial conditions sequence containing 3 sequences, (box_size, particle_parameters, fields). 
      <ul>
        <li>box_size contains ($L_x,L_y,L_z$), each an integer representing the $x$, $y$ and $z$ dimensions. </li>
        <li>particle_parameters contains (particle_positions, velocities, qs, ms, q/ms, Number of each pseudospecies, Weights).
          <ul>
            <li>particle_positions and velocities should both be $N\times3$ arrays.</li>
            <li>qs,ms and q/ms should all be Nx1 arrays. Note it has to be $N\times1$ and not $N$ to be compatible with JAX's vmap function. Also note the use of $\frac{q}{m}$ to reduce floating point errors as JAX is single-precision.</li>
            <li>Number of each pseudospecies should be a sequence of the number of each pseudospecies. It is used to split up the particles when outputting . E.g. if I had 5000 electrons, 1000 protons and 4000 Deuterium ions in that order when initialising the particles, it would be (5000,1000,4000).</li>
            <li>Weights should be a float.
          </ul>
        </li>
        <li> fields contains (array of E-fields,array of B-fields) where both are $M\times3$ arrays specifying initial E- and B- fields. Note the staggered grid when dealing with E-fields, which are defined on the edges of cells. In EM_solver.py there is a function, find_E0_by_matrix to help check if the initial conditions are correct (this may provide the wrong answer by a constant, hence it is recommended to manually calculate the E-field values).  </li>
      </ul>
    </li>
    <li>ext_fields contains (array of E-fields,array of B-fields) where both are $M\times3$ arrays specifying external E- and B- fields.
    <li>BCs is a 4-integer tuple representing (left particle BC, right particle BC, left field BC, right field BC). Particle BCs are 0 for periodic, 1 for reflective and 2 for destructive. Field BCs are 0 for periodic, 1 for reflective, 2 for trasnsmissive and 3 for laser. Detailed information on how these BCs work can be found below. If 3 for field BCs is selected, the laser magnitude and wavenumber must be specified with the arguments laser_mag and laser_k (both default 0). </li>
</ol>

In the Examples folder example_script.py gives a skeleton for the initialisation.

### Output
The simulation supports 2 forms of output, as a returned dictionary variable or by saving CSV files. This is defined with the write_to_file argument (default false).

For smaller simulations, the code saves all particle positions and velocities as a $N_t\times N\times3$ array for more flexibility in manipulation, for example for the 2D phase-space histogramming in the 2-stream instability example. The dictionary keys are: 'Time',
'Kinetic Energy','Positions','Velocities','E-fields','E-field Energy','B-fields','B-field Energy','Charge Densities','Temperature' where 'Temperature' returns a $2\times M$ array for the first 2 species.

For larger simulations, the $x$-positions are histogrammed by cell and $x$-velocities are histogrammed in 30 bins from $-3v_{rms}$ to $3v_{rms}$. The path to save the files can be defined by the path_to_file argument, default in the current working directory. CSV file names are 'time.csv', 'kinetic_energy.csv', 'E_x.csv', 'E_y.csv', 'E_z.csv', 'E_energy_densities.csv', 'B_x.csv', 'B_y.csv', 'B_z.csv', 'B_energy_densities.csv', 'chargedens.csv'. For each species there will be a 'speciesi_no_densities.csv', 'speciesi_temp.csv' and 'speciesi_vx_dist.csv' where i is an integer.

## Why JAX?
JAX is a Python module utilising the XLA (accelerated Linear Algebra) compiler to create efficient machine learning code. The github repository can be found <a href='https://github.com/google/jax'>here</a>. So why are we using it to write PIC code? 

1. JAX's compiler allows Python code to be passed onto GPUs to run. Given the parallel nature of PIC codes (advancing many particles at once with the same equations of motion), on top of JAX's vmap function to perform calculations vectorially, the code is well-suited to run on GPUs, utilising parallel computing to run quickly.

2. By writing our code in accordance with JAX's restrictions, we can use JAX's jit function to compile code efficiently and get large speed increases. As a quick test for how much of a speed increase we can get, I ran the current calculation code on 500/5000/50000/500000 particles and 100 grid cells on my local PC.
After removing the ```@jit``` decorator from our ```find_j``` function, running 
```python
import timeit

string = '''
import jax.numpy as jnp
import jax
from particles_to_grid import find_j

dx = 1
dt = dx/(2*3e8)
grid = jnp.arange(0.5,100.5,dx)
grid_start = grid[0]-dx/2
no_particles = 500/5000/50000/500000
xs_n = jax.random.uniform(jax.random.PRNGKey(100),shape=(no_particles,3),minval=0,maxval=100)
vs_n = jax.random.normal(jax.random.PRNGKey(100),shape=(no_particles,3))
xs_nminushalf = xs_n - vs_n*dt/2
xs_nplushalf = xs_n + vs_n*dt/2
qs = 1.6e-19*jnp.ones(shape=(no_particles,1))
'''

timeit.timeit(stmt='find_j(xs_nminushalf,xs_n,xs_nplushalf,vs_n,qs,dx,dt,grid,grid_start,0,0)',setup=string,number=100)
```
returns 44.4/44.5/46.6/67.5s.

Adding the ```@jit``` decorator back and the ```.block_until_ready()``` command behind ```find_j```, with the first 500 particle run taking 0.62s (due to compilation time), the output is now about 0.05/0.97/4.59/40.6s.

As another example, for the boris step with 500/5000/50000/500000 particles,
```python
import timeit

string = '''
import jax.numpy as jnp
import jax
from particle_mover import boris_step

dx = 1
dt = dx/(2*3e8)
no_particles = 500/5000/50000/500000
xs_nplushalf = jax.random.uniform(jax.random.PRNGKey(100),shape=(no_particles,3),minval=0,maxval=100)
vs_n = jax.random.normal(jax.random.PRNGKey(100),shape=(no_particles,3))
q_ms = jnp.ones(shape=(no_particles,1))
E_fields_at_x = jnp.ones(shape=(no_particles,3))
B_fields_at_x = jnp.ones(shape=(no_particles,3))
'''

timeit.timeit(stmt='boris_step(dt,xs_nplushalf,vs_n,q_ms,E_fields_at_x,B_fields_at_x)',setup=string,number=100)
```
gave us outputs of 0.47/0.45/0.70/1.95s. 

Jitting the function (and using ```jax.block_until_ready()```) gave us 0.0044/0.13/0.20/0.81s.

This is only on my local PC: While we are still trying to run it on GPUs, it would provide another speed boost to our simulation.

However, perhaps what this project best provides is a PIC code which is much more accessible, one which does not require knowledge of old and relatively unknown languages like Fortran. Even undergraduates can use or develop the code to their needs just by getting used to JAX's slightly different syntax. 

The code could even be used as a learning tool to visualise plasma effects in Plasma Physics courses, albeit only 1D effects in its current iteration. Several of these effects are shown in the Examples folder (see below for more details).

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
![diagram of one cycle of the simulation](./Images/cycle.png)

The Equations to be solved are:

![#ff0000](https://placehold.co/15x15/ff0000/ff0000.png) EM Solver:<ol>
<li>$\frac{\partial B}{\partial t} = -\nabla\times E$</li>
<li>$\frac{\partial E}{\partial t} = c^2\nabla\times B-\frac{j}{\varepsilon_0}$</li>
</ol>

![#009933](https://placehold.co/15x15/009933/009933.png) Particles to Grid:<ol>
<li>(in $x$) $\nabla\cdot j = -\frac{\partial\rho}{\partial t}$</li>
<li>(in $y,z$) $j=nqv$</li>
</ol>

![#ff6633](https://placehold.co/15x15/ff6633/ff6633.png) Particle Pusher:<ol>
<li>$\frac{dv}{dt}=q(E+v\times B)$</li>
<li>$\frac{dx}{dt}=v$</li>
</ol>

### 1. The Particle Pusher
The particle pusher functions are contained in the particle_mover.py module.

The Boris algorithm staggers the position and velocity in time. The equations used are:
$$v^-=v_t+\frac{q}{m}E_t\frac{\Delta t}{2}$$
$$\frac{v^+-v^-}{\Delta t}=\frac{q}{2m}(v^++v^-)\times B_t$$
$$v_{t+\Delta t}=v^++\frac{q}{m}E_t\frac{\Delta t}{2}$$
This was taken from [1].

To solve the second equation, if $P=P\times Q + R$, then $P=\frac{R+R\times Q+(Q\cdot R)Q}{1+Q\cdot Q}$ [2]. Applying this to our equations gives us $Q=\frac{q\Delta t}{2m}B_t$ and $R=v^-+\frac{q\Delta t}{2m}(v^-\times B_t)$.

### 2. Particles to Grid
These functions are contained in the particles_to_grid.py module.

Particles are taken as pseudoparticles with a weight $\Omega$ such that number density $n=\frac{N_{p}\Omega}{L}$ where $N_{p}$ is the number of pseudoparticles. This is in agreement with the 1D grid, where $\Omega$ carries an 'areal weight' on top of a normal weight (so $\Omega$ has units of no. of actual particles/$m^2$ ). The pseudoparticles have a triangular shape function of width $2\Delta x$, as used in EPOCH [3]. This smooths out the properties on the grid to reduce numerical noise.

![shape function of particles](./Images/shapefunction.png)

Thus when copying particle charges onto the grid, the charge density in cell $i$ is, where $x_i$ is the ith cell's $x$-position, and $X$ is the particle's $x$-position:

-For $|X-x_i|\leq\frac{\Delta x}{2}$ (particle is in cell), $\rho=\frac{q}{\Delta x}\left(\frac{3}{4}-\frac{(X-x_i)^2}{\Delta x^2}\right)$.

-For $\frac{\Delta x}{2}\leq|X-x_i|\leq\frac{3\Delta x}{2}$ (particle is in the next cell), $\rho = \frac{q}{2\Delta x}\left(\frac{3}{2}-\frac{|X-x_i|}{\Delta x}\right)^2$.

-For $\frac{3\Delta x}{2}\geq|X-x_i|$ (particle is at least 2 cells away), $\rho=0$.

The current density is found using the equation $\frac{\partial j}{\partial x} = -\frac{\partial\rho}{\partial t}$, as in Villasenor and Buneman [4] and EPOCH [5]. This is done by sweeping the grid from left to right. In one timestep, each particle can travel at most 1 cell (since the simulation becomes unstable as $\frac{dx}{dt}\to3\times10^8$), so with the shape function, we only need to sweep between -3 to 2 spaces from the particle's initial cell, where the first cell is always empty as the starting point for the sweeping.

![current sweeping method](./Images/current_sweep.png)

The current in y and z direction use $j=nqv$, or more precisely $j=N_p\rho v$.

### 3. The EM solver
The EM solver is contained in the EM_solver.py module. The equations to solve are $\frac{\partial B}{\partial t} = -\nabla\times E$ and $\frac{\partial E}{\partial t} = c^2\nabla\times B-\frac{j}{\varepsilon_0}$. A staggered Yee grid is used, where E-fields are defined on right-side cell edges and B-fields are defined on cell centres. 

![yee grid](./Images/yee_grid.png)

 This increases the accuracy of the Curl calculations. We do not solve Gauss' Law directly, as Poisson solvers can lead to numerical issues, and Gauss' Law is automatically obeyed if we use the charge conservation equation, provided Gauss' Law was satisfied at the start.

In a 1D PiC code, $\frac{dE}{dt} = \nabla\times B$ and $\frac{dB}{dt} = \nabla\times E$ solve transverse EM wave components. $j_x$ updates longitudinal E-field, and $j_y$ and $j_z$ feed into $E_y$ and $E_z$ to create EM waves.

The solver takes 2 steps of $\frac{dt}{2}$ each, first updating the E-field before the B-field, then vice versa, to create another layer of averaging. 


### 4. Fields to Particles
The function to return the fields to the particles is found in the particle_mover.py module. Taking into account the particle spanning several cells due to its shape, the total force it experiences adding each part is, where $i$ is the particle cell number, $x_i$ is the ith cell's $x$-position, and $X$ is the particle's $x$-position [3], 
$$F_{on part} = \frac{1}{2}F_{i-1}\left(\frac{1}{2}+\frac{x_i-X}{\Delta x}\right)^2 + F_{i}\left(\frac{3}{4}-\frac{(x_i-X)^2}{\Delta x^2}\right) + \frac{1}{2}F_{i+1}\left(\frac{1}{2}-\frac{x_i-X}{\Delta x}\right)^2$$
Note that in the code, the indices of the the forces are shifted by 1 due to the ghost cells. Also note that to account for particles in the first half cell (which when using the staggered grid, are out of the grid), an extra grid cell has to be added.

### Boundary Conditions
Boundary conditions are found in the boundary_conditions.py module.

Boundary conditions are specified by moving the particles and changing their velocities as desired after they have left the box, and applying ghost cells for fields.

The code supports 3 particle BC modes, and 4 field BC modes, to be specified on each side. 

For particles:

| Mode | BC | Particle position	| Particle velocity |	Force experienced by particle in ghost cells GL1/GL2/GR|
|---|---|---|---|---|
| 0 | Periodic | Move particle back to other side of box. This is done with the modulo function to find the distance left from the cell. | No change. | GL1 = 2nd last cell </br> GL2 = Last cell </br> GR = First cell |
| 1 | Reflective | Move particle back the excess distance. | Multiply $x$-component by -1. | GL1 = 2nd cell </br> GL2 = First cell </br> GR = Last cell |
| 2 | Destructive | Park particles on either side outside the box. JAX needs fixed array lengths, so removing particles causes it to recompile each time and increase the code runtime. </br></br> Set their position outside of the box, currently at L-Δx for the left and R+2.5Δx for the right, where L/R is the left/right $x$-position of the box. (When calling jnp.arange to produce the grid, the rightmost cells begin producing some numerical deviation, hence when particles leave the right border, parking the particle exactly on the next ghost cell produces some numerical issues. However, when indexing beyond the length of the array, JAX will take the last element of the array. Thus we can park the particle a few $\Delta x$'s away.) </br></br> Also set q and q/m to 0 so they do not contribute any charge density/current. | Set to 0. | GL1 = 0 </br> GL2 = 0 </br> GR = 0 |

Note the need to use 2 ghost cells on the left due to the leftmost edges of particles in the first half cell exceeding the grid when using the staggered grid. Also note $y$ and $z$ BCs are always periodic.

For fields:

| Mode | BC | Ghost cells GL/GR|
|---|---|---|
| 0 | Periodic | GL = Last cell </br> GR = First cell |
| 1 | Reflective | GL = First cell </br> GR = Last cell |
| 2 | Transmissive | Silver-Mueller BCs [6]. By applying conditions for a left-propagating wave for the left cell ($E_y=-cB_z,E_z=cB_y$) and a right-propagating wave for the right ($E_y=cB_z,E_z=-cB_y$),  and with a simple averaging to account for the staggering (for example $\frac{E_L+E_0}{2}=B_0$), we get: </br></br> $E_{yL}=-E_{y0}-2cB_{z0}$ </br> $E_{zL}=-E_{z0}+2cB_{y0}$ </br> $B_{yL}=3B_{y0}-\frac{2}{c}E_{z0}$ </br> $B_{zL}=3B_{z0}+\frac{2}{c}E_{y0}$ </br> </br> $E_{yR}=3E_{y,-1}-2cB_{z,-1}$ </br> $E_{zR}=3E_{z,-1}+2cB_{y,-1}$ </br> $B_{yR}=-B_{y,-1}-\frac{2}{c}E_{z,-1}$ </br> $B_{zR}= -B_{z,-1}+\frac{2}{c}E_{y,-1}$ </br> </br> This gives us a zero-order approximation for transmissive BCs. |
| 3 | Laser | For laser amplitude A and wavenumber k defined at the start, </br></br> $E_{yL}=Asin(kct)$ </br> $B_{zL}=\frac{A}{c} sin(kct)$ </br> $E_{yR}=Asin(kct)$ </br> $B_{zR}=-\frac{A}{c} sin(kct)$ |

### Diagnostics
Apart from the core solver, there is an additional diagnostics.py module for returning useful output. In it are functions to find the system's total kinetic energy, E-field density, B-field density, temperature at each cell and velocity histogram. These are returned in the output.

Temperature in each cell is calculated first by finding and subtracting any drift velocity $&lt v&gt$ from the particles in the cell, then using $\frac{1}{2}mv^2=\frac{3}{2}kT$ for each particle and adding up the temperatures.

This module also contains a function to perform Fourier transforms on number density data.

### The simulation.py module
Finally, the simulation.py module puts it all together. It defines one step in the cycle, which is called in an n_cycles function so we can take many steps before performing diagnosis (for long simulations where timescales of phenomenon are much longer than the dt required to maintain stability, $\frac{dx}{dt}<3\times10^8$). 

This outermost function n_cycles, as well as any other outermost functions in the simulation function, are decorated with @jit so jax compiles the function and any other function called inside it. block_until_ready statements are placed where necessary to run on GPUs. 

## Examples
In the examples folder there are some example simulations showing typical plasma behaviour, mostly set out by Langdon and Birdsall [7]. They are, with their approximate runtime on my local PC and some notes based on how far I got during the project:
<ol>
<li> Plasma oscillations (16s). Frequency agrees with theoretical frequency of $\omega=\sqrt{\frac{ne^2}{m_e\varepsilon_0}}=\sqrt{\frac{N_p\Omega e^2}{Lm_e\varepsilon_0}}$.</li>
<li> Plasma waves (130s). A Fourier transform was performed to find the dominant modes in the simulation. While the FT plot takes the shape of the dispersion relation, there are strong modes in the entire area below the line as well. Note that $\Delta x$ must be on the order of the Debye length to prevent numerical heating, as is the case for any thermal effects. </li>
<li> Hybrid oscillations (43s). Elliptical motion of particles can be seen, and frequency agrees with theoretical frequency of $\omega_H^2=\omega_C^2+\omega_P^2$ where $\omega_C$ is cyclotron frequency and $\omega_P$ is plasma frequency. Note particles have to be initialised with a velocity based on their position to see the elliptical motion, and this velocity must be $&lt&lt c$ to ensure the system is electrostatic. </li>
<li> 2-stream instability (225s). A 2D histogram on the position and velocity was performed to plot the system in phase space. 2 configurations were tested, one with counterstreaming electrons in a sea of protons, and one with counterstreaming positrons and electrons. Changing the grid resolution changes the modes that can be captured by the simulation, leading to different patterns in phase space. The last 2 cells plot the system's energy, and the conversion of kinetic energy to electric field energy can be seen, as well as the point where the instability starts becoming saturated in the log plot (energy plots off by a factor for positron-electron example). </li>
<li> Weibel instability (100s). With 2 groups of electrons, one moving in $+z$ direction and one in $-z$ direction, B-fields can clearly be seen growing, and current filaments forming and merging. The last 2 cells show energy plots, and a log plot showing the growth and saturation of the instability.</li>
<li> Precursor (110s). A laser travels into an underdense plasma, and a small attenuation can be seen. However, a portion of the wave appears to be reflected. One can also try an overdense plasma and see that most of the wave is reflected and in the plasma, it is shorted out by the plasma.


# References
[1] H. Qin, S. Zhang, J. Xiao, J. Liu, Y. Sun, W. M. Tang (2013, August). "Why is Boris algorithm so good." Physics of Plasmas [Online]. vol. 20, issue 8. Available: https://doi.org/10.1063/1.4818428.

[2] A. Hakim (2021). "Computational Methods in Plasma Physics Lecture II." PPPL Graduate Summer School 2021 [PowerPoint slides]. slide 19. Available: https://cmpp.readthedocs.io/en/latest/_static/lec2-2021.pdf.

[3] C. Brady, K. Bennett, H. Schmitz, C. Ridgers (2021, June). Section 4.3.1 "Particle Shape Functions" in "Developers Manual for the EPOCH PIC Codes." Version 4.17.0. Latest version available: https://github.com/Warwick-Plasma/EPOCH_manuals/releases.

[4] J. Villasenor, O. Buneman (1992, March). "Rigorous charge conservation for local electromagnetic field solvers." Computer Physics Communications [Online]. vol. 69, issues 2–3, pages 306-316. Available: https://doi.org/10.1016/0010-4655(92)90169-Y.

[5] C. Brady, K. Bennett, H. Schmitz, C. Ridgers (2021, June). Section 4.3.2 "Current Calculation" in "Developers Manual for the EPOCH PIC Codes." Version 4.17.0. Latest version available: https://github.com/Warwick-Plasma/EPOCH_manuals/releases.

[6] R. Lehe (2016, June). "Electromagnetic wave propagation in Particle-In-Cell codes." US Particle Accelerator School (USPAS) Summer Session 2016 [PowerPoint slides]. slides 18-24. Available: https://people.nscl.msu.edu/~lund/uspas/scs_2016/lec_adv/A1b_EM_Waves.pdf.

[7] C.K. Birdsall, A.B. Langdon. Plasma Physics via Computer Simulation (1st ed.). Bristol: IOP Publishing Ltd, 1991.
