#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:45:46 2023

@author: seanlim
"""
from jax import jit
import jax
import jax.numpy as jnp

@jit
def find_chargedens_1particle(x,q,dx,grid):
    grid_noBCs =  (q/dx)*jnp.where(abs(x-grid)<=dx/2,3/4-(x-grid)**2/(dx**2),
                        jnp.where((dx/2<abs(x-grid))&(abs(x-grid)<=3*dx/2),
                                  0.5*(3/2-abs(x-grid)/dx)**2,
                                  jnp.zeros(len(grid))))
    grid_BC_left = (q/dx)*jnp.where(abs(x-grid[0])<=dx/2,0.5*(0.5+(grid[0]-x)/dx)**2,0)
    grid_BC_right = (q/dx)*jnp.where(abs(grid[-1]-x)<=dx/2,0.5*(0.5+(x-grid[-1])/dx)**2,0)
    grid_BCs = grid_noBCs.at[-1].set(grid_BC_left+grid_noBCs[-1])
    grid_BCs = grid_BCs.at[0].set(grid_BC_right+grid_noBCs[0])
    return grid_BCs

@jit
def find_chargedens_grid(xs_n,qs,dx,grid):
    chargedens = jnp.zeros(len(grid))
    def chargedens_update(i,chargedens):
        chargedens += find_chargedens_1particle(xs_n[i,0],qs[i,0],dx,grid)
        return chargedens
    chargedens = jax.lax.fori_loop(0,len(xs_n),chargedens_update,chargedens)
    return chargedens

@jit
def find_j(xs_nminushalf,xs_n,xs_nplushalf,vs_n,qs,dx,dt,grid,grid_start):
    current_dens_x = jnp.zeros(len(grid))
    current_dens_y = jnp.zeros(len(grid))
    current_dens_z = jnp.zeros(len(grid))
    
    def current_update_x(i,jx):
        x_nminushalf = xs_nminushalf[i,0]
        x_nplushalf = xs_nplushalf[i,0]
        q = qs[i,0]
        
        diff_chargedens_1particle = (find_chargedens_1particle(x_nplushalf,q,dx,grid)-find_chargedens_1particle(x_nminushalf,q,dx,grid))/dt
        j_grid_if_BC = jnp.zeros(len(grid))
        j_grid_if_BC = j_grid_if_BC.at[0].set(-(diff_chargedens_1particle[0]+diff_chargedens_1particle[-1]+diff_chargedens_1particle[-2]+diff_chargedens_1particle[-3])*dx) #Calculating first cell value if a particle is there
        j_grid = jnp.where((diff_chargedens_1particle[0]!=0) | (diff_chargedens_1particle[-1]!=0),j_grid_if_BC,jnp.zeros(len(grid))) #If particle is at BC, use the altered first cell value.
        def iterate_grid(k,j_grid):
            j_grid = j_grid.at[k+1].set(-diff_chargedens_1particle[k+1]*dx+j_grid[k])
            return j_grid
        j_grid = jax.lax.fori_loop(0,len(grid)-1,iterate_grid,j_grid)
        jx += j_grid
        return jx
    current_dens_x = jax.lax.fori_loop(0,len(xs_nminushalf),current_update_x,current_dens_x)
    
    def current_update_y(i,jy):
        x_n = xs_n[i,0]
        q = qs[i,0]
        vy_n = vs_n[i,1]
        chargedens = find_chargedens_1particle(x_n,q,dx,grid)
        jy_grid = chargedens*vy_n
        jy += jy_grid
        return jy
    current_dens_y = jax.lax.fori_loop(0,len(xs_nminushalf),current_update_y,current_dens_y)
    
    def current_update_z(i,jz):
        x_n = xs_n[i,0]
        q = qs[i,0]
        vz_n = vs_n[i,2]
        chargedens = find_chargedens_1particle(x_n,q,dx,grid)
        jz_grid = chargedens*vz_n
        jz += jz_grid
        return jz
    current_dens_z = jax.lax.fori_loop(0,len(xs_nminushalf),current_update_z,current_dens_z)   
    
    return jnp.transpose(jnp.array([current_dens_x,current_dens_y,current_dens_z]))
