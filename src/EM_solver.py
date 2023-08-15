#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:45:47 2023

@author: seanlim
"""

from jax import jit
import jax.numpy as jnp
from boundary_conditions import field_ghost_cells_E,field_ghost_cells_B

@jit
def find_E0_by_matrix(chargedens,dx,grid):
    matrix = jnp.diag(jnp.ones(len(grid)))-jnp.diag(jnp.ones(len(grid)-1),k=-1)
    matrix.at[0,-1].set(-1)
    E_field_from_matrix = (dx/8.85e-12)*jnp.linalg.solve(matrix,chargedens)
    return E_field_from_matrix


def curlE(E_field,B_field,dx,dt,field_BC_left,field_BC_right,current_t,E0,k):
    #First, set ghost cells
    ghost_cell_L, ghost_cell_R = field_ghost_cells_E(field_BC_left,field_BC_right,E_field,B_field,dx,current_t,E0,k)
    E_field = jnp.insert(E_field,0,ghost_cell_L,axis=0)
    E_field = jnp.append(E_field,jnp.array([ghost_cell_R]),axis=0)  
    
    #If taking i - i-1 (since E-fields defined on right faces), no need to roll.
    dFz_dx = (E_field[1:-1,2]-E_field[0:-2,2])/dx
    dFy_dx = (E_field[1:-1,1]-E_field[0:-2,1])/dx
    return jnp.transpose(jnp.array([jnp.zeros(len(dFz_dx)),-dFz_dx,dFy_dx]))

def curlB(B_field,E_field,dx,dt,field_BC_left,field_BC_right,current_t,B0,k):
    #First, set ghost cells
    ghost_cell_L, ghost_cell_R = field_ghost_cells_B(field_BC_left,field_BC_right,B_field,E_field,dx,current_t,B0,k)
    B_field = jnp.insert(B_field,0,ghost_cell_L,axis=0)
    B_field = jnp.append(B_field,jnp.array([ghost_cell_R]),axis=0)  
    
    #If taking i+1 - i (since B-fields defined on edges), roll by -1 first. 
    B_field = jnp.roll(B_field,-1,axis=0)
    dFz_dx = (B_field[1:-1,2]-B_field[0:-2,2])/dx
    dFy_dx = (B_field[1:-1,1]-B_field[0:-2,1])/dx
    return jnp.transpose(jnp.array([jnp.zeros(len(dFz_dx)),-dFz_dx,dFy_dx]))


def field_update1(E_fields,B_fields,dx,dt_2,j,field_BC_left,field_BC_right,current_t,E0=0,k=0):
    #First, update E
    curl_B = curlB(B_fields,E_fields,dx,dt_2,field_BC_left,field_BC_right,current_t,E0/3e8,k)
    E_fields += dt_2*((3e8**2)*curl_B-(j/8.85e-12))
    #Then, update B
    curl_E = curlE(E_fields,B_fields,dx,dt_2,field_BC_left,field_BC_right,current_t,E0,k)
    B_fields -= dt_2*curl_E
    return E_fields,B_fields


def field_update2(E_fields,B_fields,dx,dt_2,j,field_BC_left,field_BC_right,current_t,E0=0,k=0):    
    #First, update B
    curl_E = curlE(E_fields,B_fields,dx,dt_2,field_BC_left,field_BC_right,current_t,E0,k)
    B_fields -= dt_2*curl_E
    #Then, update E 
    curl_B = curlB(B_fields,E_fields,dx,dt_2,field_BC_left,field_BC_right,current_t,E0/3e8,k)
    E_fields += dt_2*((3e8**2)*curl_B-(j/8.85e-12))
    return E_fields,B_fields