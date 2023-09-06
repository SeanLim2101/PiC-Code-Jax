#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:34:17 2023

@author: seanlim
"""

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from simulation_module import simulation
import time

#Creating box and grid
box_size_x = 1e-2
box_size_y = 1e-2
box_size_z = 1e-2
box_size = (box_size_x,box_size_y,box_size_z)

dx=5e-5
grid = jnp.arange(-box_size_x/2+dx/2,box_size_x/2+dx/2,dx)
staggered_grid = grid + dx/2

#Creating particle ICs, with xs defined half time step in front
no_pseudoelectrons = 10000
L=0.4*box_size_x
seed = 1701
key = jax.random.PRNGKey(seed)

x_start = -0.004
x_end = x_start+L
xs = jnp.array([jnp.linspace(x_start,x_end,no_pseudoelectrons)])
electron_xs = xs
electron_ys = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_y/2,maxval=box_size_y/2)
electron_zs = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_z/2,maxval=box_size_z/2)
electron_xs_array = jnp.transpose(jnp.concatenate((electron_xs,electron_ys,electron_zs)))

ion_xs = xs
ion_ys = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_y/2,maxval=box_size_y/2)
ion_zs = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_z/2,maxval=box_size_z/2)
ion_xs_array = jnp.transpose(jnp.concatenate((ion_xs,ion_ys,ion_zs)))

particle_xs_array = jnp.concatenate((electron_xs_array,ion_xs_array))
no_pseudoparticles = len(particle_xs_array)

particle_vs_array = jnp.zeros(shape=(no_pseudoparticles,3))

lamb = 1e-3
k = 2*jnp.pi/lamb
w0 = 3e8*k

wp = 0.5*w0
weight = 3.15e-4*wp**2*L/(no_pseudoelectrons)
q_es = -1.6e-19*weight*jnp.ones(shape=(no_pseudoelectrons,1))
q_ps = 1.6e-19*weight*jnp.ones(shape=(no_pseudoelectrons,1))
qs = jnp.concatenate((q_es,q_ps))
m_es = 9.1e-31*weight*jnp.ones(shape=(no_pseudoelectrons,1))
m_ps = 1.67e-27*weight*jnp.ones(shape=(no_pseudoelectrons,1))
ms = jnp.concatenate((m_es,m_ps))
q_mes = -1.76e11*jnp.ones(shape=(no_pseudoelectrons,1))
q_mps = 9.56e7*jnp.ones(shape=(no_pseudoelectrons,1))
q_ms = jnp.concatenate((q_mes,q_mps))

particles = (particle_xs_array,particle_vs_array,qs,ms,q_ms,
             (no_pseudoelectrons,no_pseudoparticles-no_pseudoelectrons),
             weight)

#Creating initial fields
E_fields = jnp.zeros(shape=(len(grid),3))
B_fields = jnp.zeros(shape=(len(grid),3))


fields = (E_fields,B_fields)

ICs = (box_size,particles,fields)

ext_E = jnp.zeros(shape=(len(grid),3))
ext_B = jnp.zeros(shape=(len(grid),3))
ext_fields = (ext_E,ext_B)
#%%
plt.title('Initial distribution of particles')
x_to_plot = jnp.linspace(-L/2,L/2,100)
plt.xlim([-box_size_x/2,box_size_x/2])
plt.hist(particle_xs_array[no_pseudoelectrons:,0],jnp.arange(-box_size_x/2,box_size_x/2+dx,dx),color='red',label='ions')
plt.hist(particle_xs_array[:no_pseudoelectrons,0],jnp.arange(-box_size_x/2,box_size_x/2+dx,dx),color='blue',label='electrons')
plt.legend()
plt.show()

plt.title('Initial Fields')
plt.xlim([-box_size_x/2,box_size_x/2])
plt.plot(grid+dx/2,E_fields[:,1],label='Ey')
plt.plot(grid,B_fields[:,2]*3e8,label='Bz')
plt.legend()
plt.show()

#%%
#Simulation
dt = dx/(2*3e8)
steps_per_snapshot=1
total_steps=300

start = time.perf_counter()
Data = simulation(steps_per_snapshot,total_steps,ICs,ext_fields,dx,dt,(2,2,3,2),
                  laser_mag = 5, laser_k = k)
end = time.perf_counter()
print('Simulation complete, time taken: '+str(end-start)+'s')

t = jnp.array(Data['Time'])
#%%
E_fields_t = jnp.array(Data['E-fields'])
B_fields_t = jnp.array(Data['B-fields'])
if wp>w0:
    skin_depth = 3e8/jnp.sqrt(wp**2-w0**2)

for i in range(len(t)):
    plt.title('Fields at timestep '+str(i*steps_per_snapshot))
    plt.xlim([-box_size_x/2,x_end])
    plt.ylim([-8,8])
    plt.xlabel(r'$x/m')
    plt.ylabel('Laser amplitude')
    plt.plot(grid+dx/2,E_fields_t[i,:,1],label='Ey',color='red')
    plt.plot(grid,B_fields_t[i,:,2]*3e8,label='Bz*3e8',color='green')
    plt.axvspan(x_start,x_end,label='Particles')
    if wp>w0:
        plt.axvline(x_start+jnp.log(10)*skin_depth,label='0.1 Attenuation',color='yellow')
    plt.legend()
    plt.pause(0.1)
    plt.cla()



