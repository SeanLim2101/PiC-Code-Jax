#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 18:48:41 2023

@author: seanlim
"""

import jax.numpy as jnp
from simulation_module import simulation

#Create box and grid
box_size_x = #number
box_size_y = #number
box_size_z = #number

box_size = (box_size_x,box_size_y,box_size_z)

dx = #number
grid = jnp.arange(-box_size_x/2+dx/2,box_size_x/2+dx/2,dx)
staggered_grid = grid + dx/2


#Creating particle ICs
particle_xs_array = #Nx3 JAX DeviceArray
particle_vs_array = #Nx3 JAX DeviceArray
weight = #number
qs = #Nx1 JAX DeviceArray
ms = #Nx1 JAX DeviceArray
q_ms = #Nx1 JAX DeviceArray
particle_species= (no_species1,no_species2,...)

particles = (particle_xs_array,particle_vs_array,qs,ms,q_ms,particle_species,weight)

#Creating initial fields
E_fields = #Mx3 JAX DeviceArray
B_fields = #Mx3 JAX DeviceArray

fields = (E_fields,B_fields)

ICs = (box_size,particles,fields)

ext_E = #Mx3 JAX DeviceArray
ext_B = #Mx3 JAX DeviceArray
ext_fields = (ext_E,ext_B)

left_particle_BC = #0,1,2
right_particle_BC = #0,1,2
left_field_BC = #0,1,2,3
right_field_BC = #0,1,2,3
BCs = (left_particle_BC,right_particle_BC,left_field_BC,right_field_BC)

#Simulation
dt = dx/(2*3e8) #Max dx/c
steps_per_snapshot = #Integer
total_steps = #Integer

#If saving into a variable,
Data = simulation(steps_per_snapshot,total_steps,ICs,ext_fields,dx,dt,BCs)

#If saving into files,
simulation(steps_per_snapshot,total_steps,ICs,ext_fields,dx,dt,BCs,
           write_to_file = True, 
           path_to_file = 'path/to/file')

#If using a laser,
laser_magnitude = #Number
laser_wavenumber = #Number
simulation(steps_per_snapshot,total_steps,ICs,ext_fields,dx,dt,BCs,
           laser_mag = laser_magnitude, laser_k = laser_wavenumber)



