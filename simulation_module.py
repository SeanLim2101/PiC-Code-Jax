#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 12:32:40 2023

@author: seanlim
"""

from jax import vmap, jit
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

#E & B field functions
@jit
def ext_E(xs_n):
    #Returns array of E-fields experienced by particles.
    # E_xs = 5*(xs_n[:,0]/(xs_n[:,0]**2+xs_n[:,1]**2)**(3/2))
    # E_ys = 5*(xs_n[:,1]/(xs_n[:,0]**2+xs_n[:,1]**2)**(3/2))
    E_xs = jnp.zeros(len(xs_n))
    E_ys = jnp.zeros(len(xs_n))
    E_zs = jnp.zeros(len(xs_n))
    return jnp.transpose(jnp.array([E_xs,E_ys,E_zs]))

@jit
def ext_B(xs_n):
    #Returns array of B-fields experienced by particles.
    # Bmax = 1
    # L=2
    # B_xs = Bmax*(xs_n[:,0]/L)**2+1
    # B_ys = -Bmax*xs_n[:,0]*xs_n[:,1]/(L**2)
    # B_zs = -Bmax*xs_n[:,0]*xs_n[:,2]/(L**2)
    B_xs = jnp.zeros(len(xs_n))
    B_ys = jnp.zeros(len(xs_n))
    B_zs = jnp.zeros(len(xs_n))
    # B_zs = jnp.sqrt(xs_n[:,0]**2+xs_n[:,1]**2)
    return jnp.transpose(jnp.array([B_xs,B_ys,B_zs]))

@jit
def find_chargedens_1particle(x,q,grid,dx):
    grid_noBCs =  (q/dx)*jnp.where(abs(x-grid)<=dx/2,3/4-(x-grid)**2/(dx**2),
                        jnp.where((dx/2<abs(x-grid))&(abs(x-grid)<=3*dx/2),0.5*(3/2-abs(x-grid)/dx)**2,
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
        chargedens += find_chargedens_1particle(xs_n[i,0],qs[i,0],grid,dx)
        return chargedens
    chargedens = jax.lax.fori_loop(0,len(xs_n),chargedens_update,chargedens)
    return chargedens

@jit
def get_fields_at_x(x_n,fields,grid,grid_start,dx):
    fields = jnp.insert(fields,0,fields[-1],axis=0)
    fields = jnp.append(fields,jnp.array([fields[1]]),axis=0)
    x = (x_n[0]-grid_start)%(grid[-1]-grid[0]+dx)+grid_start
    i = ((x-grid_start)//dx).astype(int)
    fields_n = 0.5*fields[i]*(0.5+(grid[i]-x)/dx)**2 + fields[i+1]*(0.75-(grid[i]-x)**2/dx**2) + 0.5*fields[i+2]*(0.5-(grid[i]-x)/dx)**2
    return fields_n

@jit
def curl(field,dx,roll):
    #First, set ghost cells
    field = jnp.insert(field,0,field[-1],axis=0)
    field = jnp.append(field,jnp.array([field[1]]),axis=0)  
    #if taking i+1 - i, roll by -1 first. If taking i - i-1, no need to roll.
    field = jnp.roll(field,roll,axis=0)
    dFz_dx = (field[1:-1,2]-field[0:-2,2])/dx
    dFy_dx = (field[1:-1,1]-field[0:-2,1])/dx
    return jnp.transpose(jnp.array([jnp.zeros(len(dFz_dx)),-dFz_dx,dFy_dx]))

@jit
def field_update1(E_fields,B_fields,dx,dt_2,j):
    #First, update E
    curlB = curl(B_fields,dx,0)    
    E_fields += dt_2*((3e8**2)*curlB-(j/8.85e-12))
    #Then, update B
    curlE = curl(E_fields,dx,-1)
    B_fields -= dt_2*curlE
    return E_fields,B_fields

@jit
def field_update2(E_fields,B_fields,dx,dt_2,j):    
    #First, update B
    curlE = curl(E_fields,dx,-1)
    B_fields -= dt_2*curlE
    #Then, update E 
    curlB = curl(B_fields,dx,0)
    E_fields += dt_2*((3e8**2)*curlB-(j/8.85e-12))
    return E_fields,B_fields

@jit
def find_j(xs_nminushalf,xs_nplushalf,qs,dx,dt,grid):
    current_dens = jnp.zeros(len(grid))
    def current_update(i,j):
        x_nminushalf = xs_nminushalf[i,0]
        x_nplushalf = xs_nplushalf[i,0]
        q = qs[i,0]
        
        diff_chargedens_1particle = (find_chargedens_1particle(x_nplushalf,q,grid,dx)-find_chargedens_1particle(x_nminushalf,q,grid,dx))/dt
        j_grid_if_BC = jnp.zeros(len(grid))
        j_grid_if_BC = j_grid_if_BC.at[0].set(-(diff_chargedens_1particle[0]+diff_chargedens_1particle[-1]+diff_chargedens_1particle[-2]+diff_chargedens_1particle[-3])*dx)
        j_grid = jnp.where((diff_chargedens_1particle[0]!=0) | (diff_chargedens_1particle[-1]!=0),j_grid_if_BC,jnp.zeros(len(grid))) #Ghost cell to start from 0 in case a particle exists at cell 0
        def iterate_grid(k,j_grid):
            j_grid = j_grid.at[k+1].set(-diff_chargedens_1particle[k+1]*dx+j_grid[k])
            return j_grid
        j_grid = jax.lax.fori_loop(0,len(grid)-1,iterate_grid,j_grid)
        j += j_grid
        return j
    current_dens = jax.lax.fori_loop(0,len(xs_nminushalf),current_update,current_dens)
    return jnp.transpose(jnp.array([current_dens,jnp.zeros(len(grid)),jnp.zeros(len(grid))]))

#Particle mover functions
@jit
def rotation(dt,B,vsub,q_m):
    Rvec = vsub+0.5*dt*(q_m)*jnp.cross(vsub,B)
    Bvec = 0.5*q_m*dt*B
    vplus = (jnp.cross(Rvec,Bvec)+jnp.dot(Rvec,Bvec)*Bvec+Rvec)/(1+jnp.dot(Bvec,Bvec))
    return vplus

@jit
def set_BCs(x_n,box_size_x,box_size_y,box_size_z):
    x_n0 = (x_n[0]+box_size_x/2)%(box_size_x)-box_size_x/2
    x_n1 = (x_n[1]+box_size_y/2)%(box_size_y)-box_size_y/2
    x_n2 = (x_n[2]+box_size_z/2)%(box_size_z)-box_size_z/2
    return jnp.array([x_n0,x_n1,x_n2])

@jit
def boris_step(dt,xs_nplushalf,vs_n,q_ms,E_fields_at_x,B_fields_at_x):
    """Perform a Boris step, with timestep dt and xs_n and vs_nsubhalf being 
    matrices of particle 2D position and velocity. Returns new matrix of particle 
    2D position and velocity"""
    total_Es = E_fields_at_x + ext_E(xs_nplushalf)
    total_Bs = B_fields_at_x + ext_B(xs_nplushalf)

    vs_n_int = vs_n + (q_ms)*total_Es*dt/2
    
    vs_n_rot = vmap(lambda B_n,v_n,q_ms:rotation(dt,B_n,v_n,q_ms))(total_Bs,vs_n_int,q_ms[:,0])
    
    vs_nplus1 = vs_n_rot + (q_ms)*total_Es*dt/2
    
    xs_nplus3_2 = xs_nplushalf + dt*vs_nplus1
    
    return xs_nplus3_2,vs_nplus1

#Cycle
@jit
def one_cycle(params,qs,q_ms,dx,dt,grid,grid_start,staggered_grid,box_size_x,box_size_y,box_size_z):
    xs_nplushalf = params[0]
    vs_n = params[1]
    E_fields = params[2]
    B_fields = params[3]
    xs_nminushalf = params[4]
       
    #find j from x_n-1/2 and x_n+1/2 first
    j = find_j(xs_nminushalf,xs_nplushalf,qs,dx,dt,grid)
    #1/2 step E&B field update
    E_fields, B_fields = field_update1(E_fields,B_fields,dx,dt/2,j)
    
    #Find E&B fields and boris step
    E_fields_at_x = vmap(lambda x_n: get_fields_at_x(x_n,E_fields,staggered_grid,grid_start+dx/2,dx))(xs_nplushalf)
    B_fields_at_x = vmap(lambda x_n: get_fields_at_x(x_n,B_fields,grid,grid_start,dx))(xs_nplushalf)
    xs_nplus3_2,vs_nplus1 = boris_step(dt,xs_nplushalf,vs_n,q_ms,E_fields_at_x,B_fields_at_x)
    
    #Implement (periodic) BCs
    xs_nplus3_2 = vmap(lambda x_n:set_BCs(x_n,box_size_x,box_size_y,box_size_z))(xs_nplus3_2)
    
    #find j from x_n and x_n+1
    j = find_j(xs_nplushalf,xs_nplus3_2,qs,dx,dt,grid)
    #1/2 step E&B field update
    E_fields, B_fields = field_update2(E_fields,B_fields,dx,dt/2,j)
    
    params = (xs_nplus3_2,vs_nplus1,E_fields,B_fields,xs_nplushalf)
    return params

@jit
def n_cycles(n,xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,qs,q_ms,dx,dt,grid,grid_start,staggered_grid,box_size_x,box_size_y,box_size_z):
    params_i = (xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf)
    onecycle_fixed = lambda j,params:one_cycle(params,qs,q_ms,dx,dt,grid,grid_start,staggered_grid,box_size_x,box_size_y,box_size_z)
    params_iplusn = jax.lax.fori_loop(0,n,onecycle_fixed,params_i)
    return params_iplusn[0],params_iplusn[1],params_iplusn[2],params_iplusn[3],params_iplusn[4]

#Diagnostics
@jit
def get_system_ke(vs,ms):
    kes = 0.5*ms*vmap(jnp.dot)(vs,vs)
    return jnp.sum(kes)

@jit
def get_system_momentum(vs,ms):
    return jnp.sum(ms*vs,axis=0)

def simulation(steps_per_snapshot,total_steps,ICs,dx,dt,colors,animate=True,trace=False):
    #Unpack ICs
    box_size_x=ICs[0][0]
    box_size_y=ICs[0][1]
    box_size_z=ICs[0][2]
    
    xs_nplushalf = ICs[1][0]
    vs_n = ICs[1][1]
    qs=ICs[1][2]
    ms=ICs[1][3]
    q_ms=ICs[1][4]
    
    E_fields = ICs[2][0]
    B_fields = ICs[2][1]
    
    grid = jnp.arange(-box_size_x/2+dx/2,box_size_x/2+dx/2,dx)
    grid_start = grid[0]-dx/2
    staggered_grid = grid + dx/2
    
    xs_nminushalf = vmap(lambda x_n:set_BCs(x_n,box_size_x,box_size_y,box_size_z))(xs_nplushalf-dt*vs_n)
    
    t=[]
    ke_over_time=[]
    momentum_over_time=[]
    xs_over_time=[]
    vs_over_time=[]
    E_fields_over_time=[]
    B_fields_over_time=[]
    chargedens_over_time=[]
    current_over_time=[]
    
    if animate:
        ax = plt.axes(projection='3d',xlim=(grid[0]-dx/2,grid[-1]+dx/2),ylim=(-box_size_y/2,box_size_y/2),zlim=(-box_size_z/2,box_size_z/2))
    current_t = 0
    steps_taken = 0
    while steps_taken<total_steps:  
        #Check charge dens
        chargedens_n = find_chargedens_grid(xs_nplushalf,qs,dx,grid)
        j = find_j(xs_nminushalf,xs_nplushalf,qs,dx,dt,grid)
        
        #Get data
        t.append(current_t)
        ke_over_time.append(get_system_ke(vs_n,ms))
        momentum_over_time.append(get_system_momentum(vs_n,ms))
        xs_over_time.append(xs_nplushalf)
        vs_over_time.append(vs_n)
        E_fields_over_time.append(E_fields)
        B_fields_over_time.append(B_fields)
        chargedens_over_time.append(chargedens_n)
        current_over_time.append(j)
        
        #Perform n cycles
        xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf = n_cycles(steps_per_snapshot,xs_nplushalf,vs_n,E_fields,B_fields,xs_nminushalf,qs,q_ms,dx,dt,grid,grid_start,staggered_grid,box_size_x,box_size_y,box_size_z)    

        if animate:         
            #ax.view_init(90,-90)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_aspect('equal')
            points = ax.scatter3D(xs_nplushalf[:,0],xs_nplushalf[:,1],xs_nplushalf[:,2],c=colors[:,0])
            normalised_Es = E_fields/1e8
            normalised_Bs = B_fields
            fields1 = ax.quiver(grid,jnp.zeros(len(grid)),jnp.zeros(len(grid)),normalised_Es[:,0],normalised_Es[:,1],normalised_Es[:,2],color='red',label='Normalised E-field')
            fields2 = ax.quiver(staggered_grid,jnp.zeros(len(grid)),jnp.zeros(len(grid)),normalised_Bs[:,0],normalised_Bs[:,1],normalised_Bs[:,2],color='blue',label='Normalised B-field')
            ax.legend()
            plt.pause(0.1)
            if trace==False:
                points.remove()
                fields1.remove()
                fields2.remove()
        steps_taken += steps_per_snapshot
        current_t += steps_per_snapshot*dt
    if animate:
        plt.close()
    Data = {'Time':t,
            'Kinetic Energy':ke_over_time,
            'Momentum':momentum_over_time,
            'Position':xs_over_time,
            'Velocities':vs_over_time,
            'E-fields':E_fields_over_time,
            'B-fields':B_fields_over_time,
            'Charge Densities':chargedens_over_time,
            'Current Densities':current_over_time, 
            }
    return Data
