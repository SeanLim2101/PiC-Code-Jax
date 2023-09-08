#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:10:05 2023

@author: seanlim
"""

from jax import jit,vmap
import jax.numpy as jnp
#For particles,
#0 is periodic
#1 is reflective
#2 is destructive
#For EM,
#0 is periodic
#1 is reflective
#2 is transmissive
#3 is laser


def field_ghost_cells_E(field_BC_left,field_BC_right,E_field,B_field,dx,current_t,E0,k):
    #For EM solver
    field_ghost_cell_L = jnp.where(field_BC_left==0,E_field[-1],
                         jnp.where(field_BC_left==1,E_field[0],
                         jnp.where(field_BC_left==2,jnp.array([0,-2*3e8*B_field[0,2]-E_field[0,1],2*3e8*B_field[0,1]-E_field[0,2]]),
                         jnp.where(field_BC_left==3,jnp.array([0,E0*jnp.sin(3e8*k*current_t),0]),
                                   jnp.array([0,0,0])))))
    field_ghost_cell_R = jnp.where(field_BC_right==0,E_field[0],
                         jnp.where(field_BC_right==1,E_field[-1],
                         jnp.where(field_BC_right==2,jnp.array([0,3*E_field[-1,1]-2*3e8*B_field[-1,2],3*E_field[-1,2]+2*3e8*B_field[-1,1]]),
                         jnp.where(field_BC_right==3,jnp.array([0,E0*jnp.sin(3e8*k*current_t),0]),
                                   jnp.array([0,0,0])))))
    return field_ghost_cell_L, field_ghost_cell_R

def field_ghost_cells_B(field_BC_left,field_BC_right,B_field,E_field,dx,current_t,B0,k):
    #For EM solver
    field_ghost_cell_L = jnp.where(field_BC_left==0,B_field[-1],
                         jnp.where(field_BC_left==1,B_field[0],
                         jnp.where(field_BC_left==2,jnp.array([0,3*B_field[0,1]-(2/3e8)*E_field[0,2],3*B_field[0,2]+(2/3e8)*E_field[0,1]]),
                         jnp.where(field_BC_left==3,jnp.array([0,0,B0*jnp.sin(3e8*k*current_t)]),
                                   jnp.array([0,0,0])))))
    field_ghost_cell_R = jnp.where(field_BC_right==0,B_field[0],
                         jnp.where(field_BC_right==1,B_field[-1],
                         jnp.where(field_BC_right==2,jnp.array([0,-(2/3e8)*E_field[-1,2]-B_field[-1,1],(2/3e8)*E_field[-1,1]-B_field[-1,2]]),
                         jnp.where(field_BC_right==3,jnp.array([0,0,-B0*jnp.sin(3e8*k*current_t)]),
                                   jnp.array([0,0,0])))))
    return field_ghost_cell_L, field_ghost_cell_R

def field_2_ghost_cells(part_BC_left,part_BC_right,field):
    #For returning fields to particles
    field_ghost_cell_L2 = jnp.where(part_BC_left==0,field[-2],
                          jnp.where(part_BC_left==1,field[1],
                          jnp.where(part_BC_left==2,jnp.array([0,0,0]),
                                    jnp.array([0,0,0]))))
    field_ghost_cell_L1 = jnp.where(part_BC_left==0,field[-1],
                          jnp.where(part_BC_left==1,field[0],
                          jnp.where(part_BC_left==2,jnp.array([0,0,0]),
                                    jnp.array([0,0,0]))))
    
    field_ghost_cell_R = jnp.where(part_BC_right==0,field[0],
                         jnp.where(part_BC_right==1,field[-1],
                         jnp.where(part_BC_right==2,jnp.array([0,0,0]),
                                   jnp.array([0,0,0]))))

    return field_ghost_cell_L2, field_ghost_cell_L1, field_ghost_cell_R

def chargedens_BCs(part_BC_left,part_BC_right,x,dx,grid,q):
    #If particle is on left/right cell, how much charge is out of grid?
    rem_charge_left = (q/dx)*jnp.where(abs(x-grid[0])<=dx/2,0.5*(0.5+(grid[0]-x)/dx)**2,0)
    rem_charge_right = (q/dx)*jnp.where(abs(grid[-1]-x)<=dx/2,0.5*(0.5+(x-grid[-1])/dx)**2,0)
    #Where should extra charge go?
    charge_for_L = jnp.where(part_BC_left==0,rem_charge_right,
                   jnp.where(part_BC_left==1,rem_charge_left,
                   jnp.where(part_BC_left==2,0,
                             0)))
    charge_for_R = jnp.where(part_BC_right==0,rem_charge_left,
                   jnp.where(part_BC_right==1,rem_charge_right,
                   jnp.where(part_BC_right==2,0,
                             0)))
    return charge_for_L, charge_for_R


def set_BCs(x_n,v_n,q,q_m,dx,grid,box_size_x,box_size_y,box_size_z,part_BC_left,part_BC_right):
    #Periodic in y and z
    x_n1 = (x_n[1]+box_size_y/2)%(box_size_y)-box_size_y/2
    x_n2 = (x_n[2]+box_size_z/2)%(box_size_z)-box_size_z/2
            
    x_n0 = jnp.where(x_n[0]<-box_size_x/2, #set left BC
                     jnp.where(part_BC_left==0,(x_n[0]+box_size_x/2)%(box_size_x)-box_size_x/2,
                     jnp.where(part_BC_left==1,-box_size_x-x_n[0],
                     jnp.where(part_BC_left==2,grid[0]-1.5*dx, #Park particles here
                               x_n[0]))),
           jnp.where(x_n[0]>box_size_x/2, #set right BC
                     jnp.where(part_BC_right==0,(x_n[0]+box_size_x/2)%(box_size_x)-box_size_x/2,
                     jnp.where(part_BC_right==1,box_size_x-x_n[0],
                     jnp.where(part_BC_right==2,grid[-1]+3*dx, #Park particles here
                               x_n[0]))),
                     x_n[0]))

    v_n = jnp.where(x_n[0]<-box_size_x/2, #set left BC
                    jnp.where(part_BC_left==0,v_n,
                    jnp.where(part_BC_left==1,v_n*jnp.array([-1,1,1]),
                    jnp.where(part_BC_left==2,jnp.array([0,0,0]),
                              v_n))),
          jnp.where(x_n[0]>box_size_x/2, #set right BC
                    jnp.where(part_BC_right==0,v_n,
                    jnp.where(part_BC_right==1,v_n*jnp.array([-1,1,1]),
                    jnp.where(part_BC_right==2,jnp.array([0,0,0]),
                              v_n))),
                    v_n))
    
    q = jnp.where(((x_n[0]<-box_size_x/2)&(part_BC_left==2))
                  |((x_n[0]>box_size_x/2)&(part_BC_right==2)),0,q)
    q_m = jnp.where(((x_n[0]<-box_size_x/2)&(part_BC_left==2))
                  |((x_n[0]>box_size_x/2)&(part_BC_right==2)),0,q_m)
    
    return jnp.array([x_n0,x_n1,x_n2]),v_n,q,q_m


def set_BCs_no_v(x_n,dx,grid,box_size_x,box_size_y,box_size_z,part_BC_left,part_BC_right):
    #Half-steps only used in jy, jz calculations, which only need particle 
    #positions for easy calculation
    #Periodic in y and z
    x_n1 = (x_n[1]+box_size_y/2)%(box_size_y)-box_size_y/2
    x_n2 = (x_n[2]+box_size_z/2)%(box_size_z)-box_size_z/2
    
    x_n0 = jnp.where(x_n[0]<-box_size_x/2, #set left BC
                     jnp.where(part_BC_left==0,(x_n[0]+box_size_x/2)%(box_size_x)-box_size_x/2,
                     jnp.where(part_BC_left==1,-box_size_x-x_n[0],
                     jnp.where(part_BC_left==2,grid[0]-1.5*dx,
                                x_n[0]))),
           jnp.where(x_n[0]>box_size_x/2, #set right BC
                     jnp.where(part_BC_right==0,(x_n[0]+box_size_x/2)%(box_size_x)-box_size_x/2,
                     jnp.where(part_BC_right==1,box_size_x-x_n[0],
                     jnp.where(part_BC_right==2,grid[-1]+3*dx,
                                  x_n[0]))),
                     x_n[0]))
    
    return jnp.array([x_n0,x_n1,x_n2])

@jit
def set_BCs_all(xs_n,vs_n,qs,ms,q_ms,dx,grid,box_size_x,box_size_y,box_size_z,part_BC_left,part_BC_right):   
    xs_n, vs_n ,qs, q_ms= vmap(lambda x_n,v_n,q,q_m:
                                set_BCs(x_n,v_n,q,q_m,dx,grid,box_size_x,box_size_y,box_size_z,part_BC_left,part_BC_right)
                                )(xs_n,vs_n,qs,q_ms)
    return xs_n,vs_n,qs,ms,q_ms

@jit
def set_BCs_all_midsteps(xs_n,qs,dx,grid,box_size_x,box_size_y,box_size_z,part_BC_left,part_BC_right):
    #Half-steps only used in jy, jz calculations, which only need particle 
    #positions for easy calculation
    xs_n = vmap(lambda x_n:
                set_BCs_no_v(x_n,dx,grid,box_size_x,box_size_y,box_size_z,part_BC_left,part_BC_right)
                )(xs_n)
    return xs_n

def remove_particles(xs_nplushalf,xs_n,xs_nminushalf,vs_n,qs,ms,q_ms,box_size_x,part_BC_left,part_BC_right):
    #Removing particles makes JAX recompile, making code very slow
    indices_to_remove = jnp.nonzero(((xs_nplushalf[:,0]<-box_size_x/2)&(part_BC_left==2))
                                    |((xs_nplushalf[:,0]>box_size_x/2)&(part_BC_right==2)))[0]
    
    xs_nplushalf = jnp.delete(xs_nplushalf,indices_to_remove,axis=0)
    xs_n = jnp.delete(xs_n,indices_to_remove,axis=0)
    xs_nminushalf = jnp.delete(xs_nminushalf,indices_to_remove,axis=0)
    vs_n = jnp.delete(vs_n,indices_to_remove,axis=0)
    qs = jnp.delete(qs,indices_to_remove,axis=0)
    ms = jnp.delete(ms,indices_to_remove,axis=0)
    q_ms = jnp.delete(q_ms,indices_to_remove,axis=0)
    
    return xs_nplushalf,xs_nminushalf,vs_n,qs,ms,q_ms,xs_n
    