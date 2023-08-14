#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:09:19 2023

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

dx=5e-4
grid = jnp.arange(-box_size_x/2+dx/2,box_size_x/2+dx/2,dx)
staggered_grid = grid + dx/2

Bz=0.1
ext_E = jnp.zeros(shape=(len(grid),3))
ext_B = jnp.zeros(shape=(len(grid),3))
ext_B = ext_B.at[:,2].set(Bz) #Uniform B-field in z

ext_fields = (ext_E,ext_B)

#Creating particle ICs, with xs defined half time step in front
no_pseudoelectrons = 5000
A=0.1
L=box_size_x
seed = 1701
key = jax.random.PRNGKey(seed)

xs = jnp.array([jnp.linspace(-L/2,L/2,no_pseudoelectrons)])
electron_xs = xs-(A*L/(2*jnp.pi))*jnp.sin(2*jnp.pi*xs/L)
electron_ys = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_y/2,maxval=box_size_y/2)
electron_zs = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_z/2,maxval=box_size_z/2)
electron_xs_array = jnp.transpose(jnp.concatenate((electron_xs,electron_ys,electron_zs)))

ion_xs = xs
ion_ys = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_y/2,maxval=box_size_y/2)
ion_zs = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_z/2,maxval=box_size_z/2)
ion_xs_array = jnp.transpose(jnp.concatenate((ion_xs,ion_ys,ion_zs)))

particle_xs_array = jnp.concatenate((electron_xs_array,ion_xs_array))
no_pseudoparticles = len(particle_xs_array)

#Note vy<<c to see hybrid oscillations, set B0 and wp as such
wc =  1.76e11*Bz
vxs = jnp.zeros(shape=(1,no_pseudoparticles))
electron_vys = wc*(electron_xs-xs)
ion_vys =  jnp.zeros(shape=(1,no_pseudoelectrons))
vys = jnp.concatenate((electron_vys,ion_vys),axis=1)
vzs = jnp.zeros(shape=(1,no_pseudoparticles))
particle_vs_array = jnp.transpose(jnp.concatenate((vxs,vys,vzs)))

wp = 0.2*jnp.pi*3e8/(25*dx)
weight = 3.15e-4*wp**2*L/(no_pseudoelectrons) #Need 1/10 to make wp correct? A not supposed to be there?
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

#Creating initial fields. Must satisfy Gauss' Law for initial distribution
E_fields = jnp.zeros(shape=(len(grid),3))
for i in range(len(grid)):
    E_fields = E_fields.at[i].set(jnp.array([-weight*1.6e-19*no_pseudoelectrons*A*jnp.sin(2*jnp.pi*(grid[i]+dx/2)/L)/(2*jnp.pi*8.85e-12),0,0]))
B_fields = jnp.zeros(shape=(len(grid),3))

fields = (E_fields,B_fields)

ICs = (box_size,particles,fields)


#%%
plt.title('Initial distribution of particles')
x_to_plot = jnp.linspace(-L/2,L/2,100)
plt.xlim([-box_size_x/2,box_size_x/2])
plt.hist(particle_xs_array[no_pseudoelectrons:,0],jnp.arange(-box_size_x/2,box_size_x/2+dx,dx),color='red',label='ions')
plt.hist(particle_xs_array[:no_pseudoelectrons,0],jnp.arange(-box_size_x/2,box_size_x/2+dx,dx),color='blue',label='electrons')
plt.plot(x_to_plot, (no_pseudoelectrons/len(grid))*(1+A*jnp.cos(2*jnp.pi*x_to_plot/L)))
plt.legend()
plt.show()

plt.title('Initial E-field')
plt.xlim([-box_size_x/2,box_size_x/2])
plt.plot(grid+dx/2,E_fields[:,0],'x')
plt.show()
#%%
#Simulation
dt = dx/(2*3e8)
steps_per_snapshot=20
total_steps=2000

start = time.perf_counter()
Data = simulation(steps_per_snapshot,total_steps,ICs,ext_fields,dx,dt,0,0,0,0)
end = time.perf_counter()
print('Simulation complete, time taken: '+str(end-start)+'s')

t = jnp.array(Data['Time'])

#%%
xs_over_time = jnp.array(Data['Positions'])
for i in range(len(t)):
    plt.title('Particle positions at timestep '+str(i*steps_per_snapshot))
    plt.ylim([0,1.5*no_pseudoelectrons/len(grid)])
    plt.xlim([-box_size_x/2,box_size_x/2])
    #plt.axvline((-3e8*dt*i*steps_per_snapshot-box_size_x/2)%box_size_x-box_size_x/2)
    plt.hist(xs_over_time[i,no_pseudoelectrons:,0],jnp.arange(-box_size_x/2,box_size_x/2+dx,dx),color='red',label='ions')
    plt.hist(xs_over_time[i,:no_pseudoelectrons,0],jnp.arange(-box_size_x/2,box_size_x/2+dx,dx),color='blue',label='electrons')
    plt.legend()
    plt.pause(0.1)
    plt.cla()
#%%
for i in range(len(t)):
    plt.title('Elliptical Trajectory of particles at time '+str(i*steps_per_snapshot))
    plt.scatter(xs_over_time[i,:no_pseudoelectrons:int(no_pseudoelectrons/10),0],xs_over_time[i,:no_pseudoelectrons:int(no_pseudoelectrons/10),1])
    plt.xlim([-box_size_x/2,box_size_x/2])
    plt.ylim([-box_size_y/2,box_size_y/2])
    plt.pause(0.1)
    plt.cla()
#%%
xs_over_time = jnp.array(Data['Positions'])
particle_no = 1000
wh = jnp.sqrt(wc**2+wp**2)
plt.plot(t,xs_over_time[:,particle_no,0],label='Particle')

def sin(x,A,omega,phi,C):
    return A*jnp.sin(omega*x+phi)+C
p0 = [-0.00015,wh,-jnp.pi/2,-0.003]
ts = jnp.linspace(0,t[-1],1000)
plt.plot(ts,sin(ts,*p0),label='wh,frequency = %d'%(p0[1]))
plt.legend()
