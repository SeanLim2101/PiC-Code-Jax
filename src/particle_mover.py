#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:50:35 2023

@author: seanlim
"""

from jax import vmap, jit
import jax.numpy as jnp

#Field-to-particle function
@jit
def get_fields_at_x(x_n,fields,dx,grid,grid_start):
    fields = jnp.insert(fields,0,fields[-1],axis=0)
    fields = jnp.append(fields,jnp.array([fields[1]]),axis=0)
    x = (x_n[0]-grid_start)%(grid[-1]-grid[0]+dx)+grid_start #If using a staggered grid, particles at half of 0th cell will be out of grid
    i = ((x-grid_start)//dx).astype(int)
    fields_n = 0.5*fields[i]*(0.5+(grid[i]-x)/dx)**2 + fields[i+1]*(0.75-(grid[i]-x)**2/dx**2) + 0.5*fields[i+2]*(0.5-(grid[i]-x)/dx)**2
    return fields_n

#Particle mover
@jit
def rotation(dt,B,vsub,q_m):
    Rvec = vsub+0.5*dt*(q_m)*jnp.cross(vsub,B)
    Bvec = 0.5*q_m*dt*B
    vplus = (jnp.cross(Rvec,Bvec)+jnp.dot(Rvec,Bvec)*Bvec+Rvec)/(1+jnp.dot(Bvec,Bvec))
    return vplus

@jit
def set_BCs(x_n,box_size_x,box_size_y,box_size_z):
    x_n0 = (x_n[0]+box_size_x/2)%(box_size_x)-box_size_x/2
    x_n1 = (x_n[1]+box_size_y/2)%(box_size_y)-box_size_y/2
    x_n2 = (x_n[2]+box_size_z/2)%(box_size_z)-box_size_z/2
    return jnp.array([x_n0,x_n1,x_n2])

@jit
def boris_step(dt,xs_nplushalf,vs_n,q_ms,E_fields_at_x,B_fields_at_x):
    vs_n_int = vs_n + (q_ms)*E_fields_at_x*dt/2
    vs_n_rot = vmap(lambda B_n,v_n,q_ms:rotation(dt,B_n,v_n,q_ms))(B_fields_at_x,vs_n_int,q_ms[:,0])
    vs_nplus1 = vs_n_rot + (q_ms)*E_fields_at_x*dt/2
    
    xs_nplus3_2 = xs_nplushalf + dt*vs_nplus1
    
    return xs_nplus3_2,vs_nplus1

