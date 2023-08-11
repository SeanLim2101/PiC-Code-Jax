#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:10:05 2023

@author: seanlim
"""

from jax import jit,vmap
import jax.numpy as jnp
import jax
#0 is periodic
#1 is reflective
#2 is destructive

@jit
def field_ghost_cells(BC_left,BC_right,field):
    field_ghost_cell_L = jnp.where(BC_left==0,field[-1],
                         jnp.where(BC_left==1,field[0],
                         jnp.where(BC_left==2,jnp.array([0,0,0]),
                                   jnp.array([0,0,0]))))
    field_ghost_cell_R = jnp.where(BC_right==0,field[0],
                         jnp.where(BC_right==1,field[-1],
                         jnp.where(BC_right==2,jnp.array([0,0,0]),
                                   jnp.array([0,0,0]))))
    return field_ghost_cell_L, field_ghost_cell_R

@jit
def chargedens_BCs(BC_left,BC_right,x,dx,grid,q):
    rem_charge_left = (q/dx)*jnp.where(abs(x-grid[0])<=dx/2,0.5*(0.5+(grid[0]-x)/dx)**2,0)
    rem_charge_right = (q/dx)*jnp.where(abs(grid[-1]-x)<=dx/2,0.5*(0.5+(x-grid[-1])/dx)**2,0)
    charge_for_L = jnp.where(BC_left==0,rem_charge_right,
                   jnp.where(BC_left==1,rem_charge_left,
                   jnp.where(BC_left==2,0,
                             0)))
    charge_for_R = jnp.where(BC_right==0,rem_charge_left,
                   jnp.where(BC_right==1,rem_charge_right,
                   jnp.where(BC_right==2,0,
                             0)))
    return charge_for_L, charge_for_R

@jit
def set_BCs(x_n,v_n,box_size_x,box_size_y,box_size_z,BC_left,BC_right):
    #Periodic in y and z
    x_n1 = (x_n[1]+box_size_y/2)%(box_size_y)-box_size_y/2
    x_n2 = (x_n[2]+box_size_z/2)%(box_size_z)-box_size_z/2
            
    x_n0 = jnp.where(x_n[0]<-box_size_x/2, #set left BC
                     jnp.where(BC_left==0,(x_n[0]+box_size_x/2)%(box_size_x)-box_size_x/2,
                     jnp.where(BC_left==1,-box_size_x-x_n[0],
                                x_n[0])),
           jnp.where(x_n[0]>box_size_x/2, #set right BC
                     jnp.where(BC_right==0,(x_n[0]+box_size_x/2)%(box_size_x)-box_size_x/2,
                     jnp.where(BC_right==1,box_size_x-x_n[0],
                                  x_n[0])),
                     x_n[0]))

    v_n = jnp.where(x_n[0]<-box_size_x/2, #set left BC
                    jnp.where(BC_left==0,v_n,
                    jnp.where(BC_left==1,v_n*jnp.array([-1,1,1]),
                              v_n)),
          jnp.where(x_n[0]>box_size_x/2, #set right BC
                    jnp.where(BC_right==0,v_n,
                    jnp.where(BC_right==1,v_n*jnp.array([-1,1,1]),
                              v_n)),
                    v_n))
    return jnp.array([x_n0,x_n1,x_n2]),v_n

@jit
def set_BCs_no_v(x_n,box_size_x,box_size_y,box_size_z,BC_left,BC_right):
    #set just x_n BCs since half step vs are not required
    #Periodic in y and z
    x_n1 = (x_n[1]+box_size_y/2)%(box_size_y)-box_size_y/2
    x_n2 = (x_n[2]+box_size_z/2)%(box_size_z)-box_size_z/2
    
    x_n0 = jnp.where(x_n[0]<-box_size_x/2, #set left BC
                     jnp.where(BC_left==0,(x_n[0]+box_size_x/2)%(box_size_x)-box_size_x/2,
                     jnp.where(BC_left==1,-box_size_x-x_n[0],
                                x_n[0])),
           jnp.where(x_n[0]>box_size_x/2, #set right BC
                     jnp.where(BC_right==0,(x_n[0]+box_size_x/2)%(box_size_x)-box_size_x/2,
                     jnp.where(BC_right==1,box_size_x-x_n[0],
                                  x_n[0])),
                     x_n[0]))
    
    return jnp.array([x_n0,x_n1,x_n2])

@jit
def set_BCs_all(xs_n,vs_n,qs,ms,q_ms,box_size_x,box_size_y,box_size_z,BC_left,BC_right):
    #indices_to_remove = jnp.where(((xs_n[:,0]<-box_size_x/2)&BC_left==2)
    #                              |((xs_n[:,0]>box_size_x/2)&BC_right==2))
    # particle_params = (xs_n,vs_n,qs,ms,q_ms)
    
    # def remove_particles(i,particle_params):
    #     xs_n = particle_params[0]
    #     vs_n = particle_params[1]
    #     qs = particle_params[2]
    #     ms = particle_params[3]
    #     q_ms = particle_params[4]
    #     xs_n = jnp.delete(xs_n,indices_to_remove[i])
    #     vs_n = jnp.delete(vs_n,indices_to_remove[i])
    #     qs = jnp.delete(qs,indices_to_remove[i])
    #     ms = jnp.delete(ms,indices_to_remove[i])
    #     q_ms = jnp.delete(q_ms,indices_to_remove[i])
    #     return (xs_n,vs_n,qs,ms,q_ms)
    # particle_params = jax.lax.fori_loop(0,len(indices_to_remove),remove_particles,particle_params)
    # xs_n,vs_n,qs,ms,q_ms = particle_params[0],particle_params[1],particle_params[2],particle_params[3],particle_params[4]
    
    #xs_n = jnp.delete(xs_n,indices_to_remove,axis=0)
    #vs_n = jnp.delete(vs_n,indices_to_remove,axis=0)
    #qs = jnp.delete(qs,indices_to_remove,axis=0)
    #ms = jnp.delete(ms,indices_to_remove,axis=0)
    #q_ms = jnp.delete(q_ms,indices_to_remove,axis=0)
    
    #For non-destructive BCs
    xs_n, vs_n = vmap(lambda x_n,v_n:
                                set_BCs(x_n,v_n,box_size_x,box_size_y,box_size_z,BC_left,BC_right)
                                )(xs_n,vs_n)
    return xs_n,vs_n,qs,ms,q_ms

@jit
def set_BCs_all_midsteps(xs_n,qs,box_size_x,box_size_y,box_size_z,BC_left,BC_right):
    #indices_to_remove = jnp.nonzero(((xs_n[:,0]<-box_size_x/2)&BC_left==2)
    #                                |((xs_n[:,0]>box_size_x/2)&BC_right==2))
    # particle_params = (xs_n,qs)
    # def remove_particles(i,particle_params):
    #     xs_n = particle_params[0]
    #     qs = particle_params[1]
    #     xs_n = jnp.delete(xs_n,indices_to_remove[i])
    #     qs = jnp.delete(qs,indices_to_remove[i])
    #     return (xs_n,qs)
    # particle_params = jax.lax.fori_loop(0,len(indices_to_remove),remove_particles,particle_params)
    # xs_n,q_midsteps = particle_params[0],particle_params[1]
    
    #xs_n = jnp.delete(xs_n,indices_to_remove,axis=0)
    
    #For non-destructive BCs
    xs_n = vmap(lambda x_n:
                set_BCs_no_v(x_n,box_size_x,box_size_y,box_size_z,BC_left,BC_right)
                )(xs_n)
    return xs_n
    
    
    
    