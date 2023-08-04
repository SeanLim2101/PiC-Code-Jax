#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:55:57 2023

@author: seanlim
"""

from jax import vmap,jit
import jax
import jax.numpy as jnp

@jit
def get_system_ke(vs,ms):
    kes = 0.5*ms*vmap(jnp.dot)(vs,vs)
    return jnp.sum(kes)

@jit
def get_system_momentum(vs,ms):
    return jnp.sum(ms*vs,axis=0)

@jit
def get_E_energy(E_fields):
    return 0.5*8.85e-12*vmap(jnp.dot)(E_fields,E_fields)

@jit
def get_B_energy(B_fields):
    return 0.5*vmap(jnp.dot)(B_fields,B_fields)/(4e-7*jnp.pi)

@jit
def Ts_in_cells(xs_n,vs_n,ms,weight,species_start,species_end,dx,grid,grid_start):
    cells = ((xs_n[:,0]-grid_start)//dx).astype(int)
    
    #First calculate vdrift
    vds = jnp.zeros(shape=(len(grid),3))
    def vd_per_part(i,vd):
        v = vs_n[i]
        cell = cells[i]
        vd = vd.at[cell,0].set(vd[cell,0]+v[0])
        vd = vd.at[cell,1].set(vd[cell,1]+v[1])
        vd = vd.at[cell,2].set(vd[cell,2]+v[2])
        return vd
    vds = jax.lax.fori_loop(species_start,species_end,vd_per_part,vds)
    bin_edges = jnp.arange(-0.5,len(grid)+0.5,1)
    parts_per_cell = jnp.histogram(cells,bin_edges)[0]
    vds=vmap(jnp.divide)(vds,parts_per_cell)
    
    #Then calculate T
    Ts = jnp.zeros(len(grid))
    def T_per_part(i,T):
        v = vs_n[i]
        cell = cells[i]
        m = ms[i,0]/weight
        vnet = v-vds[cell]
        T = T.at[cell].set(T[cell]+(m/1.38e-23)*jnp.dot(vnet,vnet))
        return T
    Ts = jax.lax.fori_loop(species_start,species_end,T_per_part,Ts)
    Ts=vmap(jnp.divide)(Ts,parts_per_cell)
    
    return Ts