#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:45:46 2023

@author: seanlim
"""
from jax import jit
import jax
import jax.numpy as jnp
from boundary_conditions import chargedens_BCs


def find_chargedens_1particle(x,q,dx,grid,BC_left,BC_right):
    #See readme for particle shape and equation
    grid_noBCs =  (q/dx)*jnp.where(abs(x-grid)<=dx/2,3/4-(x-grid)**2/(dx**2),
                         jnp.where((dx/2<abs(x-grid))&(abs(x-grid)<=3*dx/2),
                                    0.5*(3/2-abs(x-grid)/dx)**2,
                                    jnp.zeros(len(grid))))
    
    #Adding extra charge on left and right cell
    chargedens_for_L,chargedens_for_R = chargedens_BCs(BC_left,BC_right,x,dx,grid,q)
    grid_BCs = grid_noBCs.at[0].set(chargedens_for_L+grid_noBCs[0])
    grid_BCs = grid_BCs.at[-1].set(chargedens_for_R+grid_BCs[-1])
    return grid_BCs

@jit
def find_chargedens_grid(xs_n,qs,dx,grid,BC_left,BC_right):
    #Sum up particles
    chargedens = jnp.zeros(len(grid))
    def chargedens_update(i,chargedens):
        chargedens += find_chargedens_1particle(xs_n[i,0],qs[i,0],dx,grid,BC_left,BC_right)
        return chargedens
    chargedens = jax.lax.fori_loop(0,len(xs_n),chargedens_update,chargedens)
    return chargedens

@jit
def find_j(xs_nminushalf,xs_n,xs_nplushalf,vs_n,qs,dx,dt,grid,grid_start,BC_left,BC_right):
    current_dens_x = jnp.zeros(len(grid))
    current_dens_y = jnp.zeros(len(grid))
    current_dens_z = jnp.zeros(len(grid))
    
    def current_update_x(i,jx):
        x_nminushalf = xs_nminushalf[i,0]
        x_nplushalf = xs_nplushalf[i,0]
        q = qs[i,0]
        cell_no = ((x_nminushalf-grid_start)//dx).astype(int)
        
        diff_chargedens_1particle_whole = (find_chargedens_1particle(x_nplushalf,q,dx,grid,BC_left,BC_right)
                                           -find_chargedens_1particle(x_nminushalf,q,dx,grid,BC_left,BC_right))/dt
        
        #Sweep only cells -3 to 2 relative to particle's initial position. 
        #To do so, roll grid to front, perform current calculations, and roll back.
        #Roll grid so that particle's initial position is on 4th cell, and select
        #first 6 cells. See readme for diagram.
        #Note 1st cell should always be 0.
        diff_chargedens_1particle_short = jnp.roll(diff_chargedens_1particle_whole,3-cell_no)[:6]
        
        def iterate_short_grid(k,j_grid):
            j_grid = j_grid.at[k+1].set(-diff_chargedens_1particle_short[k+1]*dx+j_grid[k])
            return j_grid
        j_grid_short = jnp.zeros(6)
        j_grid_short = jax.lax.fori_loop(0,6,iterate_short_grid,j_grid_short)

        #Copy 6-cell grid back onto proper grid
        def short_grid_to_grid(n,j_grid):
            j_grid = j_grid.at[n].set(j_grid_short[n])
            return j_grid
        j_grid = jnp.zeros(len(grid))
        j_grid = jax.lax.fori_loop(0,6,short_grid_to_grid,j_grid)
        
        #Roll back to its correct position on grid
        j_grid = jnp.roll(j_grid,cell_no-3)
        
        jx += j_grid
        return jx
    current_dens_x = jax.lax.fori_loop(0,len(xs_nminushalf),current_update_x,current_dens_x)
    
    #For y and z, use j=nqv=rho*v when taking into account weight function
    def current_update_y(i,jy):
        x_n = xs_n[i,0]
        q = qs[i,0]
        vy_n = vs_n[i,1]
        chargedens = find_chargedens_1particle(x_n,q,dx,grid,BC_left,BC_right)
        jy_grid = chargedens*vy_n
        jy += jy_grid
        return jy
    current_dens_y = jax.lax.fori_loop(0,len(xs_nminushalf),current_update_y,current_dens_y)
    
    def current_update_z(i,jz):
        x_n = xs_n[i,0]
        q = qs[i,0]
        vz_n = vs_n[i,2]
        chargedens = find_chargedens_1particle(x_n,q,dx,grid,BC_left,BC_right)
        jz_grid = chargedens*vz_n
        jz += jz_grid
        return jz
    current_dens_z = jax.lax.fori_loop(0,len(xs_nminushalf),current_update_z,current_dens_z)   
    
    return jnp.transpose(jnp.array([current_dens_x,current_dens_y,current_dens_z]))
