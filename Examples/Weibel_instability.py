#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:47:51 2023

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

dx=3e-4
grid = jnp.arange(-box_size_x/2+dx/2,box_size_x/2+dx/2,dx)
staggered_grid = grid + dx/2

ext_E = jnp.zeros(shape=(len(grid),3))
ext_B = jnp.zeros(shape=(len(grid),3))

ext_fields = (ext_E,ext_B)

#Creating particle ICs, with xs defined half time step in front
no_pseudoelectrons = 5000
L= box_size_x
xs = jnp.array([jnp.linspace(-L/2,L/2,no_pseudoelectrons)])
seed = 1701
key = jax.random.PRNGKey(seed)
electron_ys = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_y/2,maxval=box_size_y/2)
electron_zs = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_z/2,maxval=box_size_z/2)
electron_xs_array = jnp.transpose(jnp.concatenate((xs,electron_ys,electron_zs)))

'Electron-electron stream with stationary ions'
ion_ys = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_y/2,maxval=box_size_y/2)
ion_zs = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_z/2,maxval=box_size_z/2)
ion_xs_array = jnp.transpose(jnp.concatenate((xs,ion_ys,ion_zs)))

particle_xs_array = jnp.concatenate((electron_xs_array,ion_xs_array))
no_pseudoparticles = len(particle_xs_array)

alternating_ones = (-1)**jnp.array(range(0,no_pseudoelectrons))
v0=1e8
electron_vzs = v0*alternating_ones
ion_vzs = jnp.zeros(no_pseudoelectrons)
vzs = jnp.concatenate((electron_vzs,ion_vzs))
vys = jnp.zeros(no_pseudoparticles)
vxs = jnp.zeros(no_pseudoparticles)
particle_vs_array = jnp.transpose(jnp.concatenate((jnp.array([vxs]),jnp.array([vys]),jnp.array([vzs]))))

weight = 1e13
q_es = -1.6e-19*weight*jnp.ones(shape=(no_pseudoelectrons,1))
q_ps = 1.6e-19*weight*jnp.ones(shape=(no_pseudoelectrons,1))
qs = jnp.concatenate((q_es,q_ps))
m_es = 9.1e-31*weight*jnp.ones(shape=(no_pseudoelectrons,1))
m_ps = 1.67e-27*weight*jnp.ones(shape=(no_pseudoelectrons,1))
ms = jnp.concatenate((m_es,m_ps))
q_mes = -1.76e11*jnp.ones(shape=(no_pseudoelectrons,1))
q_mps = 9.56e7*jnp.ones(shape=(no_pseudoelectrons,1))
q_ms = jnp.concatenate((q_mes,q_mps))

particles = (particle_xs_array,particle_vs_array,qs,ms,q_ms,no_pseudoelectrons,weight)

E_fields = jnp.zeros(shape=(len(grid),3))
B_fields = jnp.zeros(shape=(len(grid),3))

fields = (E_fields,B_fields)

ICs = (box_size,particles,fields)

#Simulation
dt = dx/(2*3e8)
steps_per_snapshot=5
total_steps=2000

start = time.perf_counter()
Data = simulation(steps_per_snapshot,total_steps,ICs,ext_fields,dx,dt)
end = time.perf_counter()
print('Simulation complete, time taken: '+str(end-start)+'s')

t = jnp.array(Data['Time'])
#%%
xs_over_time = jnp.array(Data['Positions'])
B_field_densities = jnp.array(Data['B-field Energy'])

for i in range(len(t)):
    plt.title('Trajectory of particles at time '+str(i*steps_per_snapshot))
    plt.scatter(xs_over_time[i,:no_pseudoelectrons:int(no_pseudoelectrons/100),0],xs_over_time[i,:no_pseudoelectrons:int(no_pseudoelectrons/100),2],color='red')
    plt.scatter(xs_over_time[i,1:no_pseudoelectrons-1:int(no_pseudoelectrons/100),0],xs_over_time[i,1:no_pseudoelectrons-1:int(no_pseudoelectrons/100),2],color='blue')    
    plt.imshow(jnp.tile(B_field_densities[i],reps=(10,1)),extent=[-box_size_x/2,box_size_x/2,-box_size_z/2,box_size_z/2],origin='lower',interpolation='bilinear',vmin=0)
    cb = plt.colorbar(label='B-field density')
    plt.xlim([-box_size_x/2,box_size_x/2])
    plt.ylim([-box_size_z/2,box_size_z/2])
    plt.xlabel('x')
    plt.ylabel('z')
    plt.pause(0.1)
    cb.remove()
    plt.cla()

#%%
plt.title('Energy densities over time')
ke = jnp.array(Data['Kinetic Energy'])
B_field_densities = jnp.array(Data['B-field Energy'])
E_field_densities = jnp.array(Data['E-field Energy'])
field_energy = jnp.sum(B_field_densities,axis=1) + jnp.sum(E_field_densities,axis=1)
plt.plot(t,ke,label='kinetic energy')
plt.plot(t,field_energy,label='field energies')
plt.legend()

#%%
plt.title('Log plot')
plt.axvspan(0,0.3e-9,color='yellow',label='Linear growth')
plt.axvspan(0.3e-9,1e-9,color='purple',label='Instability saturated')
plt.plot(t,jnp.log(ke),label='kinetic energy')
plt.plot(t,jnp.log(field_energy),label='field energies')
plt.legend()
