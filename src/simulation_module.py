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
from particle_mover import get_fields_at_x,set_BCs,boris_step
from EM_solver import field_update1,field_update2
from particles_to_grid import find_j,find_chargedens_grid
from diagnostics import get_system_ke,get_E_energy,get_B_energy,Ts_in_cells

#Cycle
@jit
def one_cycle(params,qs,q_ms,dx,dt,grid,grid_start,staggered_grid,ext_E,ext_B,box_size_x,box_size_y,box_size_z):
    xs_nplushalf = params[0]
    vs_n = params[1]
    E_fields = params[2]
    B_fields = params[3]
    xs_nminushalf = params[4]
    xs_n = params[5]
       
    #find j from x_n-1/2 and x_n+1/2 first
    j = find_j(xs_nminushalf,xs_n,xs_nplushalf,vs_n,qs,dx,dt,grid,grid_start)
    #1/2 step E&B field update
    E_fields, B_fields = field_update1(E_fields,B_fields,dx,dt/2,j)
    
    #Find E&B fields and boris step
    total_E = E_fields+ext_E
    total_B = B_fields+ext_B
    E_fields_at_x = vmap(lambda x_n: get_fields_at_x(x_n,total_E,dx,staggered_grid,grid_start+dx/2))(xs_nplushalf)
    B_fields_at_x = vmap(lambda x_n: get_fields_at_x(x_n,total_B,dx,grid,grid_start))(xs_nplushalf)
    xs_nplus3_2,vs_nplus1 = boris_step(dt,xs_nplushalf,vs_n,q_ms,E_fields_at_x,B_fields_at_x)
    
    #Implement (periodic) BCs
    xs_nplus3_2 = vmap(lambda x_n:set_BCs(x_n,box_size_x,box_size_y,box_size_z))(xs_nplus3_2)
    
    xs_nplus1 = vmap(lambda x_n:set_BCs(x_n,box_size_x,box_size_y,box_size_z))(xs_nplus3_2-(dt/2)*vs_nplus1)
    
    #find j from x_n3/2 and x_n1/2
    j = find_j(xs_nminushalf,xs_n,xs_nplushalf,vs_n,qs,dx,dt,grid,grid_start)
    #1/2 step E&B field update
    E_fields, B_fields = field_update2(E_fields,B_fields,dx,dt/2,j)
    
    params = (xs_nplus3_2,vs_nplus1,E_fields,B_fields,xs_nplushalf,xs_nplus1)
    return params

@jit
def n_cycles(n,xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,xs_n,qs,q_ms,dx,dt,grid,grid_start,staggered_grid,ext_E,ext_B,box_size_x,box_size_y,box_size_z):
    params_i = (xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,xs_n)
    onecycle_fixed = lambda j,params:one_cycle(params,qs,q_ms,dx,dt,grid,grid_start,staggered_grid,ext_E,ext_B,box_size_x,box_size_y,box_size_z)
    params_iplusn = jax.lax.fori_loop(0,n,onecycle_fixed,params_i)
    return params_iplusn[0],params_iplusn[1],params_iplusn[2],params_iplusn[3],params_iplusn[4],params_iplusn[5]

def simulation(steps_per_snapshot,total_steps,ICs,ext_fields,dx,dt,write_to_file = False):
    #Unpack ICs
    box_size_x=ICs[0][0]
    box_size_y=ICs[0][1]
    box_size_z=ICs[0][2]
    
    xs_n = ICs[1][0]
    vs_n = ICs[1][1]
    qs=ICs[1][2]
    ms=ICs[1][3]
    q_ms=ICs[1][4]
    
    no_pseudoelectrons = ICs[1][5]
    electrons_index = [0,no_pseudoelectrons]
    ions_index = [no_pseudoelectrons+1,no_pseudoelectrons*2]
    weight = ICs[1][6]
    
    E_fields = ICs[2][0]
    B_fields = ICs[2][1]
    ext_E = ext_fields[0]
    ext_B = ext_fields[1]
    grid = jnp.arange(-box_size_x/2+dx/2,box_size_x/2+dx/2,dx)
    grid_start = grid[0]-dx/2
    staggered_grid = grid + dx/2
    
    xs_nminushalf = vmap(lambda x_n:set_BCs(x_n,box_size_x,box_size_y,box_size_z))(xs_n-(dt/2)*vs_n)
    xs_nplushalf = vmap(lambda x_n:set_BCs(x_n,box_size_x,box_size_y,box_size_z))(xs_n+(dt/2)*vs_n)
    
    if write_to_file == True:
        
        cell_headers = []
        for i in range(len(grid)):
            cell_headers.append('cell '+str(i))
        
        datafile_names = ['time.csv','kinetic_energy.csv',
                          'E_x.csv','E_y.csv','E_z.csv','E_energy_densities.csv',
                          'B_x.csv','B_y.csv','B_z.csv','B_energy_densities.csv',
                          'chargedens.csv','electron_temp.csv','ion_temp.csv']
        datafile_headers = [['t/s'],['kinetic energy/J'],
                            cell_headers,cell_headers,cell_headers,cell_headers,
                            cell_headers,cell_headers,cell_headers,cell_headers,
                            cell_headers,cell_headers,cell_headers]
        
        for i,file in enumerate(datafile_names):
            with open(file,'w') as f:
                writer = csv.writer(f)
                writer.writerow(datafile_headers[i])
                
    elif write_to_file == False:
        t = []
        ke_over_time = []
        xs_over_time = []
        vs_over_time = []
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
        xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,xs_n = n_cycles(steps_per_snapshot,xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,xs_n,qs,q_ms,dx,dt,grid,grid_start,staggered_grid,ext_E,ext_B,box_size_x,box_size_y,box_size_z)    
        
        steps_taken += steps_per_snapshot
        current_t += steps_per_snapshot*dt
        
        #Get data
        chargedens_n = find_chargedens_grid(xs_n,qs,dx,grid)
        electron_temps = Ts_in_cells(xs_n,vs_n,ms,weight,electrons_index[0],electrons_index[1],dx,grid,grid_start)
        ion_temps = Ts_in_cells(xs_n,vs_n,ms,weight,ions_index[0],ions_index[1],dx,grid,grid_start)
        
        if write_to_file == True:
            datas = [[current_t],[get_system_ke(vs_n,ms)],
                     E_fields[:,0],E_fields[:,1],E_fields[:,2],get_E_energy(E_fields),
                     B_fields[:,0],B_fields[:,1],B_fields[:,2],get_B_energy(B_fields),
                     chargedens_n,electron_temps,ion_temps]
            for i,data in enumerate(datas):
                with open(datafile_names[i],'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)

               
        elif write_to_file == False:
            t.append(current_t)
            ke_over_time.append(get_system_ke(vs_n,ms))
            xs_over_time.append(xs_nplushalf)
            vs_over_time.append(vs_n)
            E_fields_over_time.append(E_fields)
            E_field_energy.append(get_E_energy(E_fields))
            B_fields_over_time.append(B_fields)
            B_field_energy.append(get_B_energy(B_fields))
            chargedens_over_time.append(chargedens_n)     
            Ts_over_time.append((electron_temps,ion_temps))
    
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
                'Temperature':Ts_over_time,
                }
