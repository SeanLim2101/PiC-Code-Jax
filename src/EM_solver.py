#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:45:47 2023

@author: seanlim
"""

from jax import jit
import jax.numpy as jnp
from boundary_conditions import field_ghost_cells

@jit
def find_E0_by_matrix(chargedens,dx,grid):
    matrix = jnp.diag(jnp.ones(len(grid)))-jnp.diag(jnp.ones(len(grid)-1),k=-1)
    matrix.at[0,-1].set(-1)
    E_field_from_matrix = (dx/8.85e-12)*jnp.linalg.solve(matrix,chargedens)
    return E_field_from_matrix

@jit
def curl(field,dx,roll,BC_left,BC_right):
    #First, set ghost cells
    ghost_cell_L, ghost_cell_R = field_ghost_cells(BC_left,BC_right,field)
    field = jnp.insert(field,0,ghost_cell_L,axis=0)
    field = jnp.append(field,jnp.array([ghost_cell_R]),axis=0)  
    #if taking i+1 - i, roll by -1 first. If taking i - i-1, no need to roll.
    field = jnp.roll(field,roll,axis=0)
    dFz_dx = (field[1:-1,2]-field[0:-2,2])/dx
    dFy_dx = (field[1:-1,1]-field[0:-2,1])/dx
    return jnp.transpose(jnp.array([jnp.zeros(len(dFz_dx)),-dFz_dx,dFy_dx]))

@jit
def field_update1(E_fields,B_fields,dx,dt_2,j,BC_left,BC_right):
    #First, update E
    curlB = curl(B_fields,dx,-1,BC_left,BC_right)
    E_fields += dt_2*((3e8**2)*curlB-(j/8.85e-12))
    #Then, update B
    curlE = curl(E_fields,dx,0,BC_left,BC_right)
    B_fields -= dt_2*curlE
    return E_fields,B_fields

@jit
def field_update2(E_fields,B_fields,dx,dt_2,j,BC_left,BC_right):    
    #First, update B
    curlE = curl(E_fields,dx,0,BC_left,BC_right)
    B_fields -= dt_2*curlE
    #Then, update E 
    curlB = curl(B_fields,dx,-1,BC_left,BC_right)
    E_fields += dt_2*((3e8**2)*curlB-(j/8.85e-12))
    return E_fields,B_fields