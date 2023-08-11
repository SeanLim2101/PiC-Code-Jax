#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 12:32:40 2023

@author: seanlim
"""
from jax import vmap,jit
import jax
import jax.numpy as jnp
import csv
from particle_mover import get_fields_at_x,boris_step
from EM_solver import field_update1,field_update2
from particles_to_grid import find_j,find_chargedens_grid
from diagnostics import get_system_ke,get_E_energy,get_B_energy,Ts_in_cells
from boundary_conditions import set_BCs_all,set_BCs_all_midsteps

#Cycle
@jit
def one_cycle(params,dx,dt,grid,grid_start,staggered_grid,
              ext_E,ext_B,box_size_x,box_size_y,box_size_z,BC_left,BC_right):
    xs_nplushalf = params[0]
    vs_n = params[1]
    E_fields = params[2]
    B_fields = params[3]
    xs_nminushalf = params[4]
    xs_n = params[5]
    qs = params[6]
    ms = params[7]
    q_ms = params[8]
    
    #find j from x_n-1/2 and x_n+1/2 first
    j = jax.block_until_ready(find_j(xs_nminushalf,xs_n,xs_nplushalf,vs_n,qs,dx,dt,grid,grid_start,BC_left,BC_right))
    #1/2 step E&B field update
    E_fields, B_fields = jax.block_until_ready(field_update1(E_fields,B_fields,dx,dt/2,j,BC_left,BC_right))
    
    #Find E&B fields and boris step
    total_E = E_fields+ext_E
    total_B = B_fields+ext_B
    E_fields_at_x = jax.block_until_ready(vmap(lambda x_n: 
                                               get_fields_at_x(x_n,total_E,dx,staggered_grid,grid_start+dx/2,BC_left,BC_right)
                                               )(xs_nplushalf))
    B_fields_at_x = jax.block_until_ready(vmap(lambda x_n: 
                                               get_fields_at_x(x_n,total_B,dx,grid,grid_start,BC_left,BC_right)
                                               )(xs_nplushalf))
    xs_nplus3_2,vs_nplus1 = jax.block_until_ready(boris_step(dt,xs_nplushalf,vs_n,q_ms,
                                                             E_fields_at_x,B_fields_at_x))
    
    #Implement BCs
    xs_nplus3_2,vs_nplus1,qs,ms,q_ms = jax.block_until_ready(set_BCs_all(xs_nplus3_2,vs_nplus1,qs,ms,q_ms,box_size_x,box_size_y,box_size_z,BC_left,BC_right))
    
    xs_nplus1 = jax.block_until_ready(set_BCs_all_midsteps(xs_nplus3_2-(dt/2)*vs_nplus1,qs,box_size_x,box_size_y,box_size_z,BC_left,BC_right))
    
    #find j from x_n3/2 and x_n1/2
    j = jax.block_until_ready(find_j(xs_nplushalf,xs_nplus1,xs_nplus3_2,vs_nplus1,qs,dx,dt,grid,grid_start,BC_left,BC_right))
    #1/2 step E&B field update
    E_fields, B_fields = jax.block_until_ready(field_update2(E_fields,B_fields,dx,dt/2,j,BC_left,BC_right))
    
    params = (xs_nplus3_2,vs_nplus1,E_fields,B_fields,xs_nplushalf,xs_nplus1,qs,ms,q_ms)
    return params

@jit
def n_cycles(n,xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,xs_n,
             qs,ms,q_ms,dx,dt,grid,grid_start,staggered_grid,
             ext_E,ext_B,box_size_x,box_size_y,box_size_z,
             BC_left,BC_right):
    params_i = (xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,xs_n,qs,ms,q_ms)
    onecycle_fixed = lambda j,params:one_cycle(params,
                                               dx,dt,grid,grid_start,staggered_grid,
                                               ext_E,ext_B,box_size_x,box_size_y,box_size_z,
                                               BC_left,BC_right)
    params_iplusn = jax.lax.fori_loop(0,n,onecycle_fixed,params_i)
    return params_iplusn[0],params_iplusn[1],params_iplusn[2],params_iplusn[3],params_iplusn[4],params_iplusn[5],params_iplusn[6],params_iplusn[7],params_iplusn[8]

def simulation(steps_per_snapshot,total_steps,ICs,ext_fields,dx,dt,BC_left,BC_right,write_to_file = False):
    #Unpack ICs
    box_size_x=ICs[0][0]
    box_size_y=ICs[0][1]
    box_size_z=ICs[0][2]
    
    xs_n = ICs[1][0]
    vs_n = ICs[1][1]
    qs=ICs[1][2]
    ms=ICs[1][3]
    q_ms=ICs[1][4]
    
    no_pseudo_species1 = ICs[1][5]
    species1_index = [0,no_pseudo_species1]
    species2_index = [no_pseudo_species1+1,no_pseudo_species1*2]
    weight = ICs[1][6]
    
    E_fields = ICs[2][0]
    B_fields = ICs[2][1]
    ext_E = ext_fields[0]
    ext_B = ext_fields[1]
    grid = jnp.arange(-box_size_x/2+dx/2,box_size_x/2+dx/2,dx)
    grid_start = grid[0]-dx/2
    staggered_grid = grid + dx/2
    
    xs_nplushalf,vs_n,qs,ms,q_ms = jax.block_until_ready(set_BCs_all(xs_n+(dt/2)*vs_n,vs_n,qs,ms,q_ms,box_size_x,box_size_y,box_size_z,BC_left,BC_right))
    xs_nminushalf = jax.block_until_ready(set_BCs_all_midsteps(xs_n-(dt/2)*vs_n,qs,box_size_x,box_size_y,box_size_z,BC_left,BC_right))
    
    if write_to_file == True:
        
        cell_headers = []
        for i in range(len(grid)):
            cell_headers.append('cell '+str(i))
        
        datafile_names = ['time.csv','kinetic_energy.csv',
                          'E_x.csv','E_y.csv','E_z.csv','E_energy_densities.csv',
                          'B_x.csv','B_y.csv','B_z.csv','B_energy_densities.csv',
                          'chargedens.csv',
                          'species1_no_densities.csv','species2_no_densities.csv',
                          'species1_temp.csv','species2_temp.csv']
        datafile_headers = [['t/s'],['kinetic energy/J'],
                            cell_headers,cell_headers,cell_headers,cell_headers,
                            cell_headers,cell_headers,cell_headers,cell_headers,
                            cell_headers,
                            cell_headers,cell_headers,
                            cell_headers,cell_headers]
        
        for i,file in enumerate(datafile_names):
            with open(file,'w') as f:
                writer = csv.writer(f)
                writer.writerow(datafile_headers[i])
                
    elif write_to_file == False:
        t = []
        ke_over_time = []
        xs_over_time = []
        vs_over_time = []
        #js_over_time = []
        E_fields_over_time = []
        E_field_energy = []
        B_fields_over_time = []
        B_field_energy = []
        chargedens_over_time = []
        Ts_over_time = []
    
    current_t = 0
    steps_taken = 0
    while steps_taken<total_steps:  
        #Perform n cycles
        xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,xs_n,qs,ms,q_ms = n_cycles(steps_per_snapshot,
                                                                                     xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,xs_n,
                                                                                     qs,ms,q_ms,dx,dt,grid,grid_start,staggered_grid,
                                                                                     ext_E,ext_B,box_size_x,box_size_y,box_size_z,
                                                                                     BC_left,BC_right)
        
        steps_taken += steps_per_snapshot
        current_t += steps_per_snapshot*dt
        
        #Get data
        chargedens_n = jax.block_until_ready(find_chargedens_grid(xs_n,qs,dx,grid,BC_left,BC_right))
        n1_t = jnp.histogram(xs_n[species1_index[0]:species1_index[1],0],
                             bins=jnp.linspace(-box_size_x/2,box_size_x/2,len(grid)+1))[0]
        n2_t = jnp.histogram(xs_n[species2_index[0]:species2_index[1],0],
                             bins=jnp.linspace(-box_size_x/2,box_size_x/2,len(grid)+1))[0]
        species1_temps = jax.block_until_ready(Ts_in_cells(xs_n,vs_n,ms,weight,
                                                           species1_index[0],species1_index[1],dx,grid,grid_start))
        species2_temps = jax.block_until_ready(Ts_in_cells(xs_n,vs_n,ms,weight,
                                                           species2_index[0],species2_index[1],dx,grid,grid_start))
        #j = jax.block_until_ready(find_j(xs_nminushalf,xs_n,xs_nplushalf,vs_n,qs,dx,dt,grid,grid_start,BC_left,BC_right))
        
        datas = [[current_t],[jax.block_until_ready(get_system_ke(vs_n,ms))],
                 E_fields[:,0],E_fields[:,1],E_fields[:,2],jax.block_until_ready(get_E_energy(E_fields,dx)),
                 B_fields[:,0],B_fields[:,1],B_fields[:,2],jax.block_until_ready(get_B_energy(B_fields,dx)),
                 chargedens_n,
                 n1_t,n2_t,
                 species1_temps,species2_temps]
        
        if write_to_file == True:
            for i,data in enumerate(datas):
                with open(datafile_names[i],'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)

        elif write_to_file == False:
            t.append(datas[0][0])
            ke_over_time.append(datas[1][0])
            xs_over_time.append(xs_nplushalf)
            vs_over_time.append(vs_n)
            E_fields_over_time.append(E_fields)
            E_field_energy.append(datas[5])
            B_fields_over_time.append(B_fields)
            B_field_energy.append(datas[9])
            chargedens_over_time.append(datas[10])  
            #js_over_time.append(datas[11])   
            Ts_over_time.append((datas[12],datas[13]))
    
    if write_to_file == False:
        return {'Time':t,
                'Kinetic Energy':ke_over_time,
                'Positions':xs_over_time,
                'Velocities':vs_over_time,
                'E-fields':E_fields_over_time,
                'E-field Energy':E_field_energy,
                'B-fields':B_fields_over_time,
                'B-field Energy':B_field_energy,
                'Charge Densities':chargedens_over_time,
                #'Current Densities':js_over_time,
                'Temperature':Ts_over_time,
                }
