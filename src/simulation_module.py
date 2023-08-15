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
from diagnostics import get_system_ke,get_E_energy,get_B_energy,Ts_in_cells,histogram_velocities
from boundary_conditions import set_BCs_all,set_BCs_all_midsteps,remove_particles

#Cycle
def one_cycle(params,dx,dt,grid,grid_start,staggered_grid,
              ext_E,ext_B,box_size_x,box_size_y,box_size_z,
              part_BC_left,part_BC_right,field_BC_left,field_BC_right,
              laser_mag,laser_k):
    xs_nplushalf = params[0]
    vs_n = params[1]
    E_fields = params[2]
    B_fields = params[3]
    xs_nminushalf = params[4]
    xs_n = params[5]
    qs = params[6]
    ms = params[7]
    q_ms = params[8]
    t = params[9]
    
    #find j from x_n-1/2 and x_n+1/2 first
    j = jax.block_until_ready(find_j(xs_nminushalf,xs_n,xs_nplushalf,vs_n,qs,dx,dt,grid,grid_start,part_BC_left,part_BC_right))
    #1/2 step E&B field update
    E_fields, B_fields = jax.block_until_ready(field_update1(E_fields,B_fields,dx,dt/2,j,field_BC_left,field_BC_right,t,laser_mag,laser_k))
    
    #Find E&B fields and boris step
    total_E = E_fields+ext_E
    total_B = B_fields+ext_B
    E_fields_at_x = jax.block_until_ready(vmap(lambda x_n: 
                                               get_fields_at_x(x_n,total_E,dx,staggered_grid,grid_start+dx/2,part_BC_left,part_BC_right)
                                               )(xs_nplushalf))
    B_fields_at_x = jax.block_until_ready(vmap(lambda x_n: 
                                               get_fields_at_x(x_n,total_B,dx,grid,grid_start,part_BC_left,part_BC_right)
                                               )(xs_nplushalf))
    xs_nplus3_2,vs_nplus1 = jax.block_until_ready(boris_step(dt,xs_nplushalf,vs_n,q_ms,
                                                             E_fields_at_x,B_fields_at_x))
    
    #Implement BCs
    xs_nplus3_2,vs_nplus1,qs,ms,q_ms = jax.block_until_ready(set_BCs_all(xs_nplus3_2,vs_nplus1,qs,ms,q_ms,dx,grid,box_size_x,box_size_y,box_size_z,part_BC_left,part_BC_right))
    
    xs_nplus1 = jax.block_until_ready(set_BCs_all_midsteps(xs_nplus3_2-(dt/2)*vs_nplus1,qs,dx,grid,box_size_x,box_size_y,box_size_z,part_BC_left,part_BC_right))
    
    #find j from x_n3/2 and x_n1/2
    j = jax.block_until_ready(find_j(xs_nplushalf,xs_nplus1,xs_nplus3_2,vs_nplus1,qs,dx,dt,grid,grid_start,part_BC_left,part_BC_right))
    #1/2 step E&B field update
    E_fields, B_fields = jax.block_until_ready(field_update2(E_fields,B_fields,dx,dt/2,j,field_BC_left,field_BC_right,t,laser_mag,laser_k))
    
    t += dt
    params = (xs_nplus3_2,vs_nplus1,E_fields,B_fields,xs_nplushalf,xs_nplus1,qs,ms,q_ms,t)
    return params

@jit
def n_cycles(n,xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,xs_n,
             qs,ms,q_ms,t,dx,dt,grid,grid_start,staggered_grid,
             ext_E,ext_B,box_size_x,box_size_y,box_size_z,
             part_BC_left,part_BC_right,field_BC_left,field_BC_right,
             laser_mag,laser_k):
    params_i = (xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,xs_n,qs,ms,q_ms,t)
    onecycle_fixed = lambda j,params:one_cycle(params,
                                               dx,dt,grid,grid_start,staggered_grid,
                                               ext_E,ext_B,box_size_x,box_size_y,box_size_z,
                                               part_BC_left,part_BC_right,field_BC_left,field_BC_right,
                                               laser_mag,laser_k)
    params_iplusn = jax.lax.fori_loop(0,n,onecycle_fixed,params_i)
    return params_iplusn[0],params_iplusn[1],params_iplusn[2],params_iplusn[3],params_iplusn[4],params_iplusn[5],params_iplusn[6],params_iplusn[7],params_iplusn[8],params_iplusn[9]

def simulation(steps_per_snapshot,total_steps,ICs,ext_fields,dx,dt,
               part_BC_left,part_BC_right,field_BC_left,field_BC_right,
               laser_mag = 0, laser_k = 0,
               write_to_file = False, path_to_file = ''):
    #Unpack ICs
    box_size_x=ICs[0][0]
    box_size_y=ICs[0][1]
    box_size_z=ICs[0][2]
    
    xs_n = ICs[1][0]
    vs_n = ICs[1][1]
    qs=ICs[1][2]
    ms=ICs[1][3]
    q_ms=ICs[1][4]
    no_each_pseudospecies = ICs[1][5]
    pseudospecies_indices = []
    for i,no_pseudospecies in enumerate(no_each_pseudospecies):
        if i == 0:
            zeroth_pseudospecies_indices = [0,no_pseudospecies]
            pseudospecies_indices.append(zeroth_pseudospecies_indices)
        if i>0:
            ith_pseudospecies_indices = [pseudospecies_indices[i-1][1]+1,pseudospecies_indices[i-1][1]+no_pseudospecies]
            pseudospecies_indices.append(ith_pseudospecies_indices)
    weight = ICs[1][6]
    
    E_fields = ICs[2][0]
    B_fields = ICs[2][1]
    ext_E = ext_fields[0]
    ext_B = ext_fields[1]
    grid = jnp.arange(-box_size_x/2+dx/2,box_size_x/2+dx/2,dx)
    grid_start = grid[0]-dx/2
    staggered_grid = grid + dx/2
    
    xs_nplushalf,vs_n,qs,ms,q_ms = jax.block_until_ready(set_BCs_all(xs_n+(dt/2)*vs_n,vs_n,qs,ms,q_ms,dx,grid,box_size_x,box_size_y,box_size_z,part_BC_left,part_BC_right))
    xs_nminushalf = jax.block_until_ready(set_BCs_all_midsteps(xs_n-(dt/2)*vs_n,qs,dx,grid,box_size_x,box_size_y,box_size_z,part_BC_left,part_BC_right))
    
    
    if write_to_file == True:
        
        cell_headers = []
        for i in range(len(grid)):
            cell_headers.append('cell '+str(i))
        
        datafile_names = ['time.csv','kinetic_energy.csv',
                          'E_x.csv','E_y.csv','E_z.csv','E_energy_densities.csv',
                          'B_x.csv','B_y.csv','B_z.csv','B_energy_densities.csv',
                          'chargedens.csv']
            
        datafile_headers = [['t/s'],['kinetic energy/J'],
                            cell_headers,cell_headers,cell_headers,cell_headers,
                            cell_headers,cell_headers,cell_headers,cell_headers,
                            cell_headers]
        
        v_rms0_species = []
        for i,ith_pseudospecies_indices in enumerate(pseudospecies_indices):
            datafile_names.append('species'+str(i)+'_no_densities.csv')
            datafile_headers.append(cell_headers)
            
            datafile_names.append('species'+str(i)+'_temp.csv')
            datafile_headers.append(cell_headers)
            
            datafile_names.append('species'+str(i)+'_v_dist.csv')
            vs_sq = vmap(jnp.dot)(vs_n[ith_pseudospecies_indices[0]:ith_pseudospecies_indices[1]],vs_n[ith_pseudospecies_indices[0]:ith_pseudospecies_indices[1]])
            v_rms = jnp.sqrt(jnp.sum(vs_sq)/(ith_pseudospecies_indices[1]-ith_pseudospecies_indices[0]))
            v_rms0_species.append(v_rms)
            datafile_headers.append(jnp.linspace(-3*v_rms,3*v_rms,30))
        
        for i,file in enumerate(datafile_names):
            with open(path_to_file+file,'w') as f:
                writer = csv.writer(f)
                writer.writerow(datafile_headers[i])
                
    elif write_to_file == False:
        ts = []
        ke_over_time = []
        xs_over_time = []
        vs_over_time = []
        E_fields_over_time = []
        E_field_energy = []
        B_fields_over_time = []
        B_field_energy = []
        chargedens_over_time = []
        Ts_over_time = []
    
    t = 0
    steps_taken = 0
    while steps_taken<total_steps:  
        #Perform n cycles
        xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,xs_n,qs,ms,q_ms,t = n_cycles(steps_per_snapshot,
                                                                                       xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,xs_n,
                                                                                       qs,ms,q_ms,t,dx,dt,grid,grid_start,staggered_grid,
                                                                                       ext_E,ext_B,box_size_x,box_size_y,box_size_z,
                                                                                       part_BC_left,part_BC_right,field_BC_left,field_BC_right,
                                                                                       laser_mag,laser_k)
        
        if part_BC_left == 2 or part_BC_right == 2: #Remove particles from simulation, but takes a while to run
            xs_nplushalf,xs_nminushalf,vs_n,qs,ms,q_ms,xs_n = remove_particles(xs_nplushalf,xs_n,xs_nminushalf,vs_n,qs,ms,q_ms,box_size_x,part_BC_left,part_BC_right)
        
        steps_taken += steps_per_snapshot
        
        #Get data
        chargedens_n = jax.block_until_ready(find_chargedens_grid(xs_n,qs,dx,grid,part_BC_left,part_BC_right))
        
        datas = [[t],[jax.block_until_ready(get_system_ke(vs_n,ms))],
                 E_fields[:,0],E_fields[:,1],E_fields[:,2],jax.block_until_ready(get_E_energy(E_fields,dx)),
                 B_fields[:,0],B_fields[:,1],B_fields[:,2],jax.block_until_ready(get_B_energy(B_fields,dx)),
                 chargedens_n]
        
        for i, indices in enumerate(pseudospecies_indices):
            ni_t = jnp.histogram(xs_n[indices[0]:indices[1],0],
                                 bins=jnp.linspace(-box_size_x/2,box_size_x/2,len(grid)+1))[0]
            species_i_temp = jax.block_until_ready(Ts_in_cells(xs_n,vs_n,ms,weight,
                                                               indices[0],indices[1],dx,grid,grid_start))
            species_i_velocities =  jax.block_until_ready(histogram_velocities(vs_n,indices[0],indices[1],v_rms0_species[i]))
            datas.append(ni_t)
            datas.append(species_i_temp)
            datas.append(species_i_velocities)
        
        if write_to_file == True:
            for i,data in enumerate(datas):
                with open(path_to_file+datafile_names[i],'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)

        elif write_to_file == False:
            ts.append(datas[0][0])
            ke_over_time.append(datas[1][0])
            xs_over_time.append(xs_n)
            vs_over_time.append(vs_n)
            E_fields_over_time.append(E_fields)
            E_field_energy.append(datas[5])
            B_fields_over_time.append(B_fields)
            B_field_energy.append(datas[9])
            chargedens_over_time.append(datas[10])
            Ts_over_time.append((datas[12],datas[13]))
    
    if write_to_file == False:
        return {'Time':ts,
                'Kinetic Energy':ke_over_time,
                'Positions':xs_over_time,
                'Velocities':vs_over_time,
                'E-fields':E_fields_over_time,
                'E-field Energy':E_field_energy,
                'B-fields':B_fields_over_time,
                'B-field Energy':B_field_energy,
                'Charge Densities':chargedens_over_time,
                'Temperature':Ts_over_time,
                }
