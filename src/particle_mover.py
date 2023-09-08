#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:50:35 2023

@author: seanlim
"""

from jax import vmap, jit
import jax.numpy as jnp
from boundary_conditions import field_2_ghost_cells


"Field-to-particle function"

def get_fields_at_x(x_n,field,dx,grid,grid_start,part_BC_left,part_BC_right):
    #First add ghost cells
    ghost_cell_L1, ghost_cell_L2, ghost_cell_R = field_2_ghost_cells(part_BC_left,part_BC_right,field)
    field = jnp.insert(field,0,ghost_cell_L2,axis=0)
    field = jnp.insert(field,0,ghost_cell_L1,axis=0)
    field = jnp.append(field,jnp.array([ghost_cell_R]),axis=0)
    x = x_n[0]
    
    #If using a staggered grid, particles at first half cell will be out of grid, so add extra cell
    grid = jnp.insert(grid,0,grid[0]-dx,axis=0) 
    i = ((x-grid_start+dx)//dx).astype(int) #new grid_start = grid_start-dx due to extra cell
    
    #Field indices +1 due to extra ghost cell
    fields_n = 0.5*field[i]*(0.5+(grid[i]-x)/dx)**2 + field[i+1]*(0.75-(grid[i]-x)**2/dx**2) + 0.5*field[i+2]*(0.5-(grid[i]-x)/dx)**2
    return fields_n


"Particle mover"

def rotation(dt,B,vsub,q_m):
    #See readme on solving Boris Algorithm
    Rvec = vsub+0.5*dt*(q_m)*jnp.cross(vsub,B)
    Bvec = 0.5*q_m*dt*B
    vplus = (jnp.cross(Rvec,Bvec)+jnp.dot(Rvec,Bvec)*Bvec+Rvec)/(1+jnp.dot(Bvec,Bvec))
    return vplus

@jit
def boris_step(dt,xs_nplushalf,vs_n,q_ms,E_fields_at_x,B_fields_at_x):
    vs_n_int = vs_n + (q_ms)*E_fields_at_x*dt/2
    vs_n_rot = vmap(lambda B_n,v_n,q_ms:rotation(dt,B_n,v_n,q_ms))(B_fields_at_x,vs_n_int,q_ms[:,0])
    vs_nplus1 = vs_n_rot + (q_ms)*E_fields_at_x*dt/2
    
    xs_nplus3_2 = xs_nplushalf + dt*vs_nplus1
    
    return xs_nplus3_2,vs_nplus1

