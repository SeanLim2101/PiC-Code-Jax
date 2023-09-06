# 1D PiC Code for Simulating Plasmas with Google JAX

**An Imperial College London Undergraduate Research Opportunities Project by Sean Lim, supervised by Dr Aidan Crilly**

JAX enables users to solve multiple equations vectorially (with its vmap function), and compiles code in a GPU-friendly manner, making it very useful in performing Particle-in-Cell simulations efficiently.

## Using the Simulation

### Initialising the Simulation

In the simulation.py module, the main function 'simulation' is called with arguments (steps per snapshot, total steps, ICs, ext_fields,dx,dt,BCs). For neatness, the arguments in simulation are wrapped to be called, and then unwrapped within the simulation function. 

For N particles and M cells,
<ol>
    <li>ICs is an initial conditions sequence containing 3 sequences, (box_size, particle_parameters, fields). 
      <ol>
        <li>box_size contains ($L_x,L_y,L_z$), each an integer representing the $x$, $y$ and $z$ dimensions. </li>
        <li>particle_parameters contains (particle positions, velocities, qs, ms, q/ms, number of each pseudospecies, weights).
          <ol>
            <li>Particle positions and velocities should both be an $N\times3$ array.</li>
            <li>Qs,ms and q/ms should all be Nx1 arrays. Note it has to be $N\times1$ and not $N$ to be compatible with JAX's vmap function. Also note the use of $\frac{q}{m}$ to reduce floating point errors as JAX is single-precision.</li>
            <li>number of each pseudospecies should be an iterable of the number of each pseudospecies, e.g. if I had 5000 electrons and 1000 protons, it would be (5000,1000)</li>
            <li>weights should be an integer/float
          </ol>
        </li>
        <li> fields contains (array of E-fields,array of B-fields) where both are $M\times3$ arrays specifying initial E- and B- fields. In EM_solver.py there is a function, find_E0_by_matrix to help check if the initial conditions are correct (this may provide the wrong answer by a constant, hence it is recommended to manually calculate the E-field values). </li>
      </ol>
    </li>
    <li>ext_fields contains (array of E-fields,array of B-fields) where both are $M\times3$ arrays specifying external E- and B- fields.

Note the staggered grid when dealing with E-fields, which are defined on the edges of cells.

BCs is a 4-integer tuple representing (left particle BC, right particle BC, left field BC, right field BC). Particle BCs are 0 for periodic, 1 for reflective and 2 for destructive. Field BCs are 0 for periodic, 1 for reflective, 2 for trasnsmissive and 3 for laser. Detailed information on how these BCs work can be found below.

If 3 for field BCs is selected, the laser magnitude and wavenumber must be specified with the arguments laser_mag and laser_k (both default 0).

In the Examples folder example_script.py gives a skeleton for the initialisation.

### Output
The simulation supports 2 forms of output, as a returned dictionary variable or by saving files into a folder. This is defined with the write_to_file argument (default false).

For smaller simulations, the code saves all particle $x$-positions and velocities as a $N_t\times N\times3$ array for more flexibility in manipulation, for example for the 2D-histogramming in the 2-stream instability example. The dictionary keys are: 'Time',
'Kinetic Energy','Positions','Velocities','E-fields','E-field Energy','B-fields','B-field Energy','Charge Densities','Temperature' where 'Temperature' returns a $2\times M$ array for the first 2 species.

For larger simulations, the $x$-positions are histogrammed by cell and velocities are histogrammed in 30 bins from $-3v_{rms}$ to $3v_{rms}$. The path to save the files can be defined by the path_to_file argument, default in the current working directory. CSV file names are 'time.csv','kinetic_energy.csv','E_x.csv','E_y.csv','E_z.csv','E_energy_densities.csv','B_x.csv','B_y.csv','B_z.csv','B_energy_densities.csv','chargedens.csv'. For each species there will be a 'species_no_densities.csv', 'species_temp.csv' and 'species_vx_dist.csv'.

## Why JAX?
JAX is a Python module utilising the XLA (accelerated Linear Algebra) compiler to create efficient machine learning code. The github repository can be found <a href='https://github.com/google/jax'>here</a>. So why are we using it to write PIC code? 

1. Given the parallel nature of PIC codes (advancing many particles at once with the same equations of motion), we can use JAX's vmap function to perform calculations vectorially and hence utilise parallel computing on GPUs.
2. By writing our code in accordance with JAX's restrictions, we can use JAX's jit function to compile code efficiently and get massive speed increases. As a quick test for how much of a speed increase we can get, I ran the current calculation code on 500/5000/50000/500000 particles and 100 grid cells on my local PC.
After removing the ```@jit``` decorator from our ```find_j``` function,
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

Adding the ```@jit``` decorator back and add the ```.block_until_ready()``` command behind ```find_j```, with the first 500 particle run taking 0.62s (due to compilation time), the output is now about 0.05/0.97/4.59/40.6s.

As another example, for the boris step with 500/5000/50000/500000 particles,
```python
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

Adding the ```@jit``` decorator and replacing the last line with ```python timeit.timeit(stmt='jax.block_until_ready(boris_step(dt,xs_nplushalf,vs_n,q_ms,E_fields_at_x,B_fields_at_x))',setup=string,number=100)``` gave us 0.0044/0.13/0.20/0.81s.

Some of these speed-ups make it feasible to run our PIC codes on Python to begin with. Thus we have a PIC code which is much more accessible, one which does not require knowledge of old and relatively unknown languages like Fortran. 

This accessibility is what makes the code so useful as even undergraduates can use or develop the code to their needs, on GPUs or on their local PCs, just by getting used to JAX's slightly different syntax. 

The code could even be used as a learning tool to visualise plasma effects in Plasma Physics courses, albeit only 1D effects in its current iteration. Several plasma effects are already shown in the examples folder.

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
![diagram of one cycle of the simulation](Images/cycle.png)

The Equations to be solved are:

![#ff0000](https://placehold.co/15x15/ff0000/ff0000.png):<ol>
<li>$\frac{\partial B}{\partial t} = -\nabla\times E$</li>
<li>$\frac{\partial E}{\partial t} = c^2\nabla\times B-\frac{j}{\varepsilon_0}$</li>
</ol>

![#009933](https://placehold.co/15x15/009933/009933.png):<ol>
<li>(in $x$) $\nabla j = -\frac{\partial\rho}{\partial t}$</li>
<li>(in $y,z$) $j=nqv$</li>
</ol>

![#ff6633](https://placehold.co/15x15/ff6633/ff6633.png):<ol>
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

Particles are taken as pseudoparticles with a weight $\Omega$ such that number density $n=\frac{N_{p}\Omega}{L}$ where $N_{p}$ is the number of pseudoparticles. This is in agreement with the 1D grid, where $\Omega$ carries an 'areal weight' on top of a normal weight (units of no. of actual particles/ $m^2$ ). The pseudoparticles have a triangular shape function of width $2\Delta x$, as used in EPOCH [3]. This smooths out the properties on the grid to reduce numerical noise.

![shape function of particles](Images/shapefunction.png)

Thus when copying particle charges onto the grid, the charge density in cell $i$ is:

-For $|X-x_i|\leq\frac{\Delta x}{2}$ (particle is in cell), $\rho=\frac{q}{\Delta x}\left(\frac{3}{4}-\frac{(X-x_i)^2}{\Delta x^2}\right)$.

-For $\frac{\Delta x}{2}\leq|X-x_i|\leq\frac{3\Delta x}{2}$ (particle is in the next cell), $\rho = \frac{q}{2\Delta x}\left(\frac{3}{2}-\frac{|X-x_i|}{\Delta x}\right)^2$.

-For $\frac{3\Delta x}{2}\geq|X-x_i|$ (particle is at least 2 cells away), $\rho=0$.

The current density is found using the equation $\frac{\partial j}{\partial x} = -\frac{\partial\rho}{\partial t}$, as in Villasenor and Buneman [4] and EPOCH [5]. This is done by sweeping the grid from left to right. In one timestep, each particle can travel at most 1 cell (since the simulation becomes unstable as $\frac{dx}{dt}\to3\times10^8$), so with the shape function, we only need to sweep between -3 to 2 spaces from the particle's initial cell, where the first cell is empty as the starting point for the sweeping.

![current sweeping method](Images/current_sweep.png)

The current in y and z direction use $j=nqv$, or more precisely $j=N_p\rho v$.

### 3. The EM solver
The EM solver is contained in the EM_solver.py module.A staggered Yee grid is used, where E-fields are defined on right-side cell edges and B-fields are defined on cell centres. 

![yee grid](Images/yee_grid.png)

The equations to solve are $Ampere$ and $Faraday$. We do not solve Gauss' Law directly, as Poisson solvers can lead to numerical issues, and Gauss' Law is automatically obeyed if we use the charge conservation equation, provided Gauss' Law was satisfied at the start.

In a 1D PiC code, $\frac{dE}{dt} = \nabla\times B$ and $\frac{dB}{dt} = \nabla\times E$ solve transverse EM wave components, while $j_x$ updates longitudinal E-field, and $j_y$ and $j_z$ feed into the equations to create EM waves.

The solver takes 2 steps of $\frac{dt}{2}$ each, first updating the E-field before the B-field, then vice versa. 


### 4. Fields to Particles
The function to return the fields to the particles is found in the particle_mover.py module. Taking into account the particle spanning several cells due to its shape, the total force it experiences adding each part is, where $i$ is the particle cell number, $x_i$ is the ith cell's $x$-position, and $X$ is the particle's $x$-position, 
$$F_{on part} = \frac{1}{2}F_{i-1}\left(\frac{1}{2}+\frac{x_i-X}{\Delta x}\right)^2 + F_{i}\left(\frac{3}{4}-\frac{(x_i-X)^2}{\Delta x^2}\right) + \frac{1}{2}F_{i+1}\left(\frac{1}{2}-\frac{x_i-X}{\Delta x}\right)^2$$ [3].
Note that in the code, the indeces of the the forces are shifted by 1 due to the ghost cells.

### Boundary Conditions
Boundary conditions are found in the boundary_conditions.py module.

Boundary conditions are specified by moving the particles and changing their velocities as desired after they have left the box, and applying ghost cells for fields.

Boundary conditions are also specified to find charge densities based on chosen particle BCs. Note the method of these calculations differ from the above charge density calculations as they use changing $X$ position rather than changing grid cell position.

The code supports 3 particle BC modes, and 3 field BC modes, to be specified on each side. They are displayed in this table :
Particle table:

| Mode | BC | Particle position	| Particle velocity |	Force experienced by particle in ghost cells GL1/GL2/GR|
|---|---|---|---|---|
| 0 | Periodic | Move particle back to other side of box. This is done with the modulo function to find the distance left from the cell. | No change. | GL1 = 2nd last cell </br> GL2 = Last cell </br> GR = First cell |
| 1 | Reflective | Move particle back the excess distance. | Multiply x-component by -1. | GL1 = 2nd cell </br> GL2 = First cell </br> GR = Last cell |
| 2 | Destructive | Park particles on either side outside the box. JAX needs fixed array lengths, so removing particles causes it to recompile functions each time and increases the code runtime. </br></br> Arbitrarily set their position outside of the box, currently at L-Δx for the left and R+2.5Δx for the right, where L/R is left/right x-position of box. (When calling jnp.arange to produce the grid, the elements towards the end start producing some numerical deviation, parking the particle exactly on the next ghost cell produces some issues. However, when indexing beyond the length of the array, JAX will take the last element of the array. Thus we can park the particle a few $\Delta x$'s away.) </br></br> Also set q and q/m to 0 so they do not contribute any charge density/current. | Set to 0. | GL1 = 0 </br> GL2 = 0 </br> GR = 0 |

Note the need to use 2 ghost cells on the left due to the leftmost edges of particles in the first half cell undefined when using the staggered grid  while finding E-field experienced.
Also note $y$ and $z$ BCs are always periodic.

Field table:

| Mode | BC | Ghost cells GL/GR|
|---|---|---|
| 0 | Periodic | GL = Last cell </br> GR = First cell |
| 1 | Reflective | GL = First cell </br> GR = Last cell |
| 2 | Transmissive | Silver-Mueller BCs [6]. By applying conditions for a left-propagating wave for the left cell (E_y=-cB_z,E_z=cB_y) and a right-propagating wave for the right (E_y=cB_z,E_z=-cB_y),  and with a simple averaging to account for the staggering (for example $\frac{E_{-1}+E_0}{2}=B_0$), we get: </br></br> $E_{yL}=-E_{y0}-2cB_{z0}$ </br> $E_{zL}=-E_{z0}+2cB_{y0}$ </br> $B_{yL}=3B_{y0}-\frac{2}{c}E_{z0}$ </br> $B_{zL}=3B_{z0}+\frac{2}{c}E_{y0}$ </br> </br> $E_{yR}=3E_{y,-1}-2cB_{z,-1}$ </br> $E_{zR}=3E_{z,-1}+2cB_{y,-1}$ </br> $B_{yR}=-B_{y,-1}-\frac{2}{c}E_{z,-1}$ </br> $B_{zR}= -B_{z,-1}+\frac{2}{c}E_{y,-1}$ </br> </br> This gives us a zero-order approximation for transmissive BCs. |
| 3 | Laser | For laser amplitude A and wavenumber k defined at the start, </br></br> $E_{yL}=Asin(kct)$ </br> $B_{zL}=\frac{A}{c} sin(kct)$ </br> $E_{yR}=Asin(kct)$ </br> $B_{zR}=-\frac{A}{c} sin(kct)$ |

### Diagnostics
Apart from the core solver, there is an additional diagnostics.py module for returning useful output. In it are functions to find the system's total kinetic energy, E-field density, B-field density, temperature at each cell and velocity histogram. These are returned in the output.

Temperature is calculated in each cell first by finding and subtracting any drift velocity $<v>$ from the particles in the cell, then using $\frac{1}{2}mv^2=\frac{3}{2}kT$ for each particle and adding up the temperatures.

In this module is also a function to perform Fourier transforms on number density data.

### The simulation.py module
Finally, the simulation.py module puts it all together. It defines one step in the cycle, which is called in an n_cycles function so we can take many steps before performing diagnosis for long simulations where timescales of phenomenon are much longer than the dt required to maintain stability ($\frac{dx}{dt}<3\times10^8$). 

This outermost function n_cycles, as well as any other outermost functions in the simulation function, are decorated with @jit for jax to compile the function and any other function called inside it, as well as block_until_ready statements placed where necessary to run on GPUs. 

## Examples
In the examples folder there are some example simulations showing typical plasma behaviour, mostly set out by Langdon and Birdsall [7]. They are, with their approximate runtime on my local PC and some notes based on how far I got on them during the project:
<ol>
<li> Plasma oscillations (16s). </li>
<li> Plasma waves (130s). A Fourier transform was performed to find the dominant modes in the simulation (pre-FT plot is also available on line 141). While the FT plot takes the shape of the dispersion relation, there are strong modes in the entire area below the line as well. </li>
<li> Hybrid oscillations (43s). Elliptical motion of particles can be seen, and frequency agrees with theoretical frequency of $\omega_H=\omega_C^2+\omega_P^2$ where $\omega_C$ is cyclotron frequency and $\omega_P$ is plasma frequency. Note particles have to be initialised with a velocity based on their position to see the elliptical motion, and this velocity must be $&lt&lt c$ to ensure the system is electrostatic. </li>
<li> 2-stream instability (225s). A 2D histogram on the position and velocity was performed to plot the system in phase space. 2 configurations were tested, one with 2 2 groups of electrons with opposite velocities in a sea of protons, and one with positrons and electrons travelling in opposite directions. Changing the grid resolution changes the modes that can be captured by the simulation, leading to different patterns in phase space. The last 2 cells plot the system's energy, and the conversion of kinetic energy to electric field energy can be seen, as well as the point where the instability starts becoming saturated. </li>
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
