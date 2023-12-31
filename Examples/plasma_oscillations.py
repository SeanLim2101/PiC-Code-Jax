#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 12:00:06 2023

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

#Creating particle ICs
no_pseudoelectrons = 5000
A=0.1
L=box_size_x
k=2*jnp.pi/L
seed = 1701
key = jax.random.PRNGKey(seed)

xs = jnp.array([jnp.linspace(-L/2,L/2,no_pseudoelectrons)])
electron_xs = xs-(A/k)*jnp.sin(k*xs)
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

w0 = jnp.pi*3e8/(25*dx) #100 steps for a period
weight = 3.15e-4*w0**2*L/(no_pseudoelectrons) #From plasma frequency equation
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
             (no_pseudoelectrons,no_pseudoparticles-no_pseudoelectrons),weight)

#Creating initial fields
E_fields = jnp.zeros(shape=(len(grid),3))
for i in range(len(grid)):
    E_fields = E_fields.at[i].set(jnp.array(
        [-weight*1.6e-19*no_pseudoelectrons*A*jnp.sin(k*(grid[i]+dx/2))/(k*L*8.85e-12),0,0])
        )
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
plt.hist(particle_xs_array[no_pseudoelectrons:,0],jnp.linspace(-box_size_x/2,box_size_x/2,len(grid)+1),color='red',label='ions')
plt.hist(particle_xs_array[:no_pseudoelectrons,0],jnp.linspace(-box_size_x/2,box_size_x/2,len(grid)+1),color='blue',label='electrons')
plt.plot(x_to_plot, (no_pseudoelectrons/len(grid))*(1+A*jnp.cos(k*x_to_plot)))
plt.legend()
plt.show()

plt.title('Initial E-field')
plt.xlim([-box_size_x/2,box_size_x/2])
plt.plot(grid+dx/2,E_fields[:,0],'x')
plt.show()

#You can check E-field amplitude is correct
from EM_solver import find_E0_by_matrix
E_field_from_matrix = find_E0_by_matrix(particle_xs_array,qs,dx,grid,0,0)
plt.title('E-field from matrix')
plt.xlim([-box_size_x/2,box_size_x/2])
plt.plot(grid+dx/2,E_field_from_matrix,'x')
#%%
#Simulation
dt = dx/(2*3e8)
steps_per_snapshot=5
total_steps=400

start = time.perf_counter()
Data = simulation(steps_per_snapshot,total_steps,ICs,ext_fields,dx,dt,(0,0,0,0))
end = time.perf_counter()
print('Simulation complete, time taken: '+str(end-start)+'s')

#%%
t = jnp.array(Data['Time'])
plt.title('Kinetic energy over time')
ke_over_time = jnp.array(Data['Kinetic Energy'])
plt.ylim(0.9*min(ke_over_time),1.1*max(ke_over_time))
plt.plot(t,ke_over_time)
plt.legend()
plt.show()

#%%
from scipy.optimize import curve_fit

chargedens_over_time = jnp.array(Data['Charge Densities'])
def sin(x,A,omega,phi,C):
    return A*jnp.sin(omega*x+phi)+C
p0 = [0.08,w0,-jnp.pi/2,-0.02]
fit,cov = curve_fit(sin,t,chargedens_over_time[:,len(grid)//2],p0=p0)
plt.title(r'Charge density at x=0 over time, theoretical $\omega_0$ = %.2frad/s' %(w0/1e10))
plt.xlabel('time')
plt.ylabel('Charge density')
plt.plot(t,chargedens_over_time[:,len(grid)//2])

plt.plot(t,sin(t,*fit),label=r'Fit, $\omega_0$=%.2fe10rad/s'%(fit[1]/1e10))
plt.legend()
#%%
chargedens_over_time = jnp.array(Data['Charge Densities'])
for i in range(len(t)):
    plt.title('Charge density at time %d'%(i*steps_per_snapshot))
    plt.plot(grid,chargedens_over_time[i,:])
    plt.xlim([-box_size_x/2,box_size_x/2])
    plt.ylim([0.9*min(chargedens_over_time[0,:]),1.1*max(chargedens_over_time[0,:])])
    plt.pause(0.1)
    plt.cla()

#%%
E_fields_over_time = jnp.array(Data['E-fields'])
for i in range(len(t)):
    plt.title('E-field at time %d'%(i*steps_per_snapshot))
    plt.plot(grid+dx/2,E_fields_over_time[i,:,0])
    plt.ylim([0.9*min(E_fields_over_time[0,:,0]),1.1*max(E_fields_over_time[0,:,0])])
    plt.pause(0.1)
    plt.cla()

#%%
xs_over_time = jnp.array(Data['Positions'])
for i in range(len(t)):
    plt.title('Particle positions at timestep %d'%(i*steps_per_snapshot))
    plt.ylim([0,1.5*no_pseudoelectrons/len(grid)])
    plt.xlim([-box_size_x/2,box_size_x/2])
    plt.hist(xs_over_time[i,no_pseudoelectrons:,0],jnp.linspace(-box_size_x/2,box_size_x/2,len(grid)+1),color='red',label='ions')
    plt.hist(xs_over_time[i,:no_pseudoelectrons,0],jnp.linspace(-box_size_x/2,box_size_x/2,len(grid)+1),color='blue',label='electrons')
    plt.legend()
    plt.pause(0.1)
    plt.cla()

