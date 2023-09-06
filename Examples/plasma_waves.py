#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:42:03 2023

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

dx=1e-4
grid = jnp.arange(-box_size_x/2+dx/2,box_size_x/2+dx/2,dx)
staggered_grid = grid + dx/2

#Creating particle ICs
no_pseudoelectrons = 10000
L=box_size_x
seed = 1701
key = jax.random.PRNGKey(seed)

xs = jnp.array([jnp.linspace(-L/2,L/2,no_pseudoelectrons)])
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


seed1 = 1705
seed2 = 100
key1 = jax.random.PRNGKey(seed1)
key2 = jax.random.PRNGKey(seed2)
T = 1e5
sig_electrons = jnp.sqrt(15.1e6*T)
v_electrons = sig_electrons*jax.random.normal(key1,shape=(no_pseudoelectrons,3))
sig_ions = jnp.sqrt(8625*T)
v_ions = sig_ions*jax.random.normal(key2,shape=(no_pseudoelectrons,3))
particle_vs_array = jnp.concatenate((v_electrons,v_ions))

w0 = 3e10
weight = 3.15e-4*w0**2*L/no_pseudoelectrons 
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

debye_length = 69*jnp.sqrt(L*T/(no_pseudoelectrons*weight)) #Must be close to grid length
#%%
plt.title('Initial distribution of particles')
plt.xlim([-box_size_x/2,box_size_x/2])
plt.hist(particle_xs_array[no_pseudoelectrons:,0],jnp.linspace(-box_size_x/2,box_size_x/2,len(grid)+1),color='red',label='ions')
plt.hist(particle_xs_array[:no_pseudoelectrons,0],jnp.linspace(-box_size_x/2,box_size_x/2,len(grid)+1),color='blue',label='electrons')
plt.legend()
plt.show()

plt.title('Initial E-field')
plt.xlim([-box_size_x/2,box_size_x/2])
plt.plot(grid+dx/2,E_fields[:,0])
plt.show()
#%%
#Simulation
dt = dx/(2*3e8)
steps_per_snapshot=100
total_steps=10000

start = time.perf_counter()
Data = simulation(steps_per_snapshot,total_steps,ICs,ext_fields,dx,dt,(0,0,0,0))
end = time.perf_counter()
print('Simulation complete, time taken: '+str(end-start)+'s')

#%%
from diagnostics import get_fourier_transform

t = jnp.array(Data['Time'])
xs_over_time = jnp.array(Data['Positions'])
ne = jnp.array([jnp.zeros(len(grid))])
for i in range(len(t)):
    ne_t = jnp.histogram(xs_over_time[i,:no_pseudoelectrons,0],bins=jnp.linspace(-box_size_x/2,box_size_x/2,len(grid)+1))[0]
    ne = jnp.append(ne,jnp.array([ne_t]),axis=0)
ne = jnp.delete(ne,0,axis=0)

nhat, ks, ws = get_fourier_transform(ne,grid,t)

#%%
plt.title('Frequency space')
plt.xlabel('k')
plt.ylabel(r'$\omega$')
plt.xlim([0,3e4])
plt.ylim([0,4*w0])

vth = jnp.sqrt(2*1.38e-23*T/9.1e-31)
ks_to_plot = jnp.linspace(ks[0],ks[-1],100)
plt.plot(ks_to_plot,jnp.sqrt(w0**2+3/2*(vth*ks_to_plot)**2),label='plasma wave dispersion relation',color='blue')
plt.imshow(jnp.abs(nhat),origin='lower',extent=[ks[0],ks[-1],ws[0],ws[-1]],aspect='auto',vmax=1e3)
plt.legend()
plt.colorbar()
plt.show()

'''
FT shape seems to follow dispersion relation, but is not a line, but has an 
area below the line.
'''
#%%
plt.title('Number density of electrons in x-t space')
plt.imshow(ne,origin='lower',extent=[grid[0],grid[-1],t[0],t[-1]],aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()

#%%
t = jnp.array(Data['Time'])
Ts_over_time = jnp.array(Data['Temperature'])
for i in range(len(t)):
    plt.title('Temperature along grid at timestep %d'%(i*steps_per_snapshot))
    plt.ylim([0,2e5])
    plt.xlim([-box_size_x/2,box_size_x/2])
    plt.plot(grid,Ts_over_time[i,0,:],label='electrons')
    plt.plot(grid,Ts_over_time[i,1,:],label='ions')
    plt.legend()
    plt.pause(0.1)
    plt.cla()