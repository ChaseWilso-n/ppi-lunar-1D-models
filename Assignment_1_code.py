# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:21:40 2025

@author: chase
"""

import numpy as np
import matplotlib.pyplot as plt
G = 6.67384e-11 # +/- 0.00080e-11 [m^3.kg^-1.s^-2]

def euler_step(x_n0, d_dr, dr): return x_n0 + d_dr * dr

def dM_dr(rho,r): return 4*np.pi*rho*r*r

def g_r(M,r): return G*M/r/r

def dp_dr(rho,g): return -1*rho*g

def dI_dr(rho,r): return 8/3*np.pi*rho*r*r*r*r

### Observed Quantities
M_Moon = 7.34630e+22 # +/- 0.00088e+22 [kg]
M_Moon_unc = 0.00088e+22/M_Moon
R_Moon = 1737151.0 # +/- 1.0 [m]
R_Moon_unc = 1.0/R_Moon # [m]
rho_Moon = 3345.56 # +/- 0.40 [kg.m^-3]
I_Moon_norm = 0.393112 # +/- 0.000012
I_Moon = I_Moon_norm*M_Moon*R_Moon*R_Moon
I_Moon_unc = np.sqrt((0.000012/I_Moon_norm)**2 + M_Moon_unc**2)
GM_Moon = 4902.80007e+9 # +/- 0.00014 [km^3.s^-2]
GM_Moon_unc = 0.00014e+9/GM_Moon
g_Moon_surface = GM_Moon/R_Moon/R_Moon
g_Moon_surface_unc = np.sqrt(M_Moon_unc**2 + R_Moon_unc**2 + R_Moon_unc**2)
p_Moon_core = 3/8/np.pi*GM_Moon*M_Moon/(R_Moon**4)
p_Moon_core_unc = np.sqrt(GM_Moon_unc**2 + M_Moon_unc**2 + 4*(R_Moon_unc**2))

##### Upward Integral for Mass #####
### Numerical Solution
dr = 25 # [m]
r_array = np.arange(0, R_Moon, dr)
M_array = np.zeros_like(r_array)
M_array[0] = 0. # Initial condition

for i,r in enumerate(r_array):
    if i!=0:
        M_array[i] = euler_step(M_array[i-1],dM_dr(rho_Moon,r),dr)
    else: pass

print('-'*60)
print(f'Numerical Solution: Moon Mass = {M_array[-1]} kg')
print(f'Observed: Moon Mass = {M_Moon} kg')
print(f'Observed Uncertainty = {M_Moon_unc*100} %')
print(f'Error = {(M_array[-1]-M_Moon)/M_Moon*100} %')

##### Gravity Function #####
### Numerical Solution
g_array = np.zeros_like(r_array)
g_array[0] = 0. # Initial condition

for i,r in enumerate(r_array):
    if i!=0:
        g_array[i] = g_r(M_array[i],r)
    else: pass

print('-'*60)
print(f'Numerical Solution: Moon Surface Gravity = {g_array[-1]} m/s^2')
print(f'Analytical: Moon Surface Gravity = {g_Moon_surface} m/s^2')
print(f'Analytical Uncertainty = {g_Moon_surface_unc*100} %')
print(f'Error = {(g_array[-1]-g_Moon_surface)/g_Moon_surface*100} %')

##### Downward Integral for Pressure #####
### Numerical Solution
p_array = np.zeros_like(r_array)
p_array[-1] = 0. # Initial condition

for i in range(len(r_array)-2,-1,-1):
    p_array[i] = euler_step(p_array[i+1],dp_dr(rho_Moon,g_array[i+1]),-dr)

print('-'*60)
print(f'Numerical Solution: Moon Core Pressure = {p_array[0]/1e+9} GPa')
print(f'Analytical: Moon Core Pressure = {p_Moon_core/1e+9} GPa')
print(f'Analytical Uncertainty = {p_Moon_core_unc*100} %')
print(f'Error = {(p_array[0]-p_Moon_core)/p_Moon_core*100} %')

##### Plot Results #####
fig,axes=plt.subplots(1,3,figsize=(10,6),sharey=False)
fig.suptitle('Homogeneous Density Moon')
### Mass
axes[0].plot(M_array/1e+22,r_array/1e+3,color='Blue')
axes[0].set_xlabel('Integrated Mass [10$^{22}$ kg]')
axes[0].set_ylabel('Radius [km]')
axes[0].grid(True)
### Gravity
axes[1].plot(g_array,r_array/1e+3,color='Black')
axes[1].set_xlabel('Gravity [m/s$^2$]')
axes[1].set_ylabel('Radius [km]')
axes[1].grid(True)
### Pressure
axes[2].plot(p_array/1e+9,r_array/1e+3,color='Red')
axes[2].set_xlabel('Pressure [GPa]')
axes[2].set_ylabel('Radius [km]')
axes[2].grid(True)

plt.tight_layout()
plt.show()

##### Check Moment of Inertia #####
I_array = np.zeros_like(r_array)
I_array[0] = 0. # Initial condition

for i,r in enumerate(r_array):
    if i!=0:
        I_array[i] = euler_step(I_array[i-1],dI_dr(rho_Moon,r),dr)
    else: pass

print('-'*60)
print(f'Numerical Solution: Moon Moment of Inertia = {I_array[-1]} kg m^2')
print(f'Observed: Moon Moment of Inertia = {I_Moon} kg m^2')
print(f'Observed Uncertainty = {I_Moon_unc*100} %')
print(f'Error = {(I_array[-1]-I_Moon)/I_Moon*100} %')















##### Model M1 #####

def shell_mass(rho,R_outer,R_inner): return 4/3*np.pi*rho*(R_outer**3-R_inner**3)

def shell_MoI(rho,R_outer,R_inner): return 8/15*np.pi*rho*(R_outer**5-R_inner**5)

def four_layer_moon_rho_I(rho_crust,rho_outer_core,rho_inner_core,thickness_crust,R_outer_core,R_inner_core, verbose=False):
    ### Radius of layer interfaces
    R_crust = R_Moon
    R_mantle = R_crust - thickness_crust
    ### Known Layer Properties
    M_crust = shell_mass(rho_crust,R_crust,R_mantle)
    M_outer_core = shell_mass(rho_outer_core,R_outer_core,R_inner_core)
    M_inner_core = shell_mass(rho_inner_core,R_inner_core,0)
    ### Mantle Properties
    M_mantle = M_Moon - M_crust - M_outer_core - M_inner_core
    V_mantle = 4/3*np.pi * (R_mantle**3 - R_outer_core**3)
    rho_mantle = M_mantle/V_mantle
    
    I_inner_core = shell_MoI(rho_inner_core,R_inner_core,0)
    I_outer_core = shell_MoI(rho_outer_core,R_outer_core,R_inner_core)
    I_mantle = shell_MoI(rho_mantle,R_mantle,R_outer_core)
    I_crust = shell_MoI(rho_crust,R_crust,R_mantle)
    
    if verbose:
        print(f'Outer Core Mass = {M_outer_core} kg')
        print(f'Inner Core Mass = {M_inner_core} kg')
        print(f'Mantle Mass = {M_mantle} kg')
        print(f'Outer Core MoI = {I_outer_core} kg m^2')
        print(f'Inner Core MoI = {I_inner_core} kg m^2')
        print(f'Mantle MoI = {I_mantle} kg m^2')
    
    I_total = I_crust + I_mantle + I_inner_core + I_outer_core
    
    return R_mantle, rho_mantle, I_total

def find_R_inner_core(rho_crust,rho_outer_core,rho_inner_core,thickness_crust,R_outer_core,max_iter=1000,tol=1e+26):
    ### Bisection search for R_inner_core
    a = 1.
    R_mantle_a, rho_mantle_a, I_total_a = four_layer_moon_rho_I(rho_crust,rho_outer_core,rho_inner_core,thickness_crust,R_outer_core,a)
    fa = I_total_a - I_Moon
    b = R_outer_core - 1.
    R_mantle_b, rho_mantle_b, I_total_b = four_layer_moon_rho_I(rho_crust,rho_outer_core,rho_inner_core,thickness_crust,R_outer_core,b)
    fb = I_total_b - I_Moon
    
    for iteration in range(max_iter):
        c = 0.5*(a+b)
        R_mantle_c, rho_mantle_c, I_total_c = four_layer_moon_rho_I(rho_crust,rho_outer_core,rho_inner_core,thickness_crust,R_outer_core,c)
        fc = I_total_c - I_Moon
        if abs(fc) < tol:
            R_inner_core = c
            break
        if fa*fb > 0:
            raise RuntimeError('Bisection interval does not bracket a root.')
        if fa*fc < 0:
            b,fb = c,fc
        else:
            a,fa = c,fc
    else: raise RuntimeError('Bisection Search Did Not Converge for R_inner_core.')
    R_mantle, rho_mantle, I_total = four_layer_moon_rho_I(rho_crust,rho_outer_core,rho_inner_core,thickness_crust,R_outer_core,R_inner_core,verbose=True)
    return R_inner_core, R_mantle, rho_mantle, I_total

def compute_1D_profiles(R_crust,rho_crust,R_mantle,rho_mantle,R_outer_core,rho_outer_core,R_inner_core,rho_inner_core,dr,verbose=False):
    ### Construct density profile
    r_array = np.arange(0,R_crust,dr)
    rho_array = np.zeros_like(r_array)
    
    for i,r in enumerate(r_array):
        if r < R_inner_core:
            rho_array[i] = rho_inner_core
        elif r < R_outer_core:
            rho_array[i] = rho_outer_core
        elif r < R_mantle:
            rho_array[i] = rho_mantle
        else:
            rho_array[i] = rho_crust
    
    ##### Upward Integral for Mass #####
    M_array = np.zeros_like(r_array)
    M_array[0] = 0. # Initial condition
    
    for i,r in enumerate(r_array):
        if i!=0:
            M_array[i] = euler_step(M_array[i-1],dM_dr(rho_array[i],r),dr)
        else: pass
    
    if verbose:
        print('-'*60)
        print(f'Numerical Solution: Moon Mass = {M_array[-1]} kg')
        print(f'Observed: Moon Mass = {M_Moon} kg')
        print(f'Observed Uncertainty = {M_Moon_unc*100} %')
        print(f'Error = {(M_array[-1]-M_Moon)/M_Moon*100} %')
    
    ##### Gravity Function #####
    g_array = np.zeros_like(r_array)
    g_array[0] = 0. # Initial condition
    
    for i,r in enumerate(r_array):
        if i!=0:
            g_array[i] = g_r(M_array[i],r)
        else: pass
    
    if verbose:
        print('-'*60)
        print(f'Numerical Solution: Moon Surface Gravity = {g_array[-1]} m/s^2')
        print(f'Analytical: Moon Surface Gravity = {g_Moon_surface} m/s^2')
        print(f'Analytical Uncertainty = {g_Moon_surface_unc*100} %')
        print(f'Error = {(g_array[-1]-g_Moon_surface)/g_Moon_surface*100} %')
    
    ##### Downward Integral for Pressure #####
    p_array = np.zeros_like(r_array)
    p_array[-1] = 0. # Initial condition
    
    for i in range(len(r_array)-2,-1,-1):
        p_array[i] = euler_step(p_array[i+1],dp_dr(rho_array[i],g_array[i+1]),-dr)
    
    if verbose:
        print('-'*60)
        print(f'Numerical Solution: Moon Core Pressure = {p_array[0]/1e+9} GPa')
        print(f'Analytical (Homogeneous Moon): Moon Core Pressure = {p_Moon_core/1e+9} GPa')
        print(f'Analytical Uncertainty = {p_Moon_core_unc*100} %')
        print(f'Error = {(p_array[0]-p_Moon_core)/p_Moon_core*100} %')
    
    return r_array, rho_array, p_array, g_array

### Constant Crust Properties
rho_crust = 2760 # [kg.m^-3]
thickness_crust = 40000 # [m]
### Constant Core Properties
rho_inner_core = 8000 # [kg.m^-3]
rho_outer_core = 5171 # [kg.m^-3]
### Spatial Step for 1D Profiles
dr = 25 # [m]

### Max Model
print('-'*60)
print('Maximum Model:')
R_outer_core_max = 465600 # [m]
R_inner_core_max,R_mantle_max,rho_mantle_max,I_total_max = find_R_inner_core(rho_crust,rho_outer_core,rho_inner_core,thickness_crust,R_outer_core_max)
print(f'Inner Core Radius = {R_inner_core_max/1e+3} km')
print(f'Outer Core Radius = {R_outer_core_max/1e+3} km')
print(f'Mantle Density = {rho_mantle_max} kg/m^3')
print(f'Moment of Inertia = {I_total_max/R_Moon/R_Moon/M_Moon}')
r_array_max, rho_array_max, p_array_max, g_array_max = compute_1D_profiles(R_Moon,rho_crust,R_mantle_max,rho_mantle_max,R_outer_core_max,rho_outer_core,R_inner_core_max,rho_inner_core,dr)

### Min Model
print('-'*60)
print('Minimum Model:')
R_outer_core_min = 336500 # [m]
R_inner_core_min,R_mantle_min,rho_mantle_min,I_total_min = find_R_inner_core(rho_crust,rho_outer_core,rho_inner_core,thickness_crust,R_outer_core_min)
print(f'Inner Core Radius = {R_inner_core_min/1e+3} km')
print(f'Outer Core Radius = {R_outer_core_min/1e+3} km')
print(f'Mantle Density = {rho_mantle_min} kg/m^3')
print(f'Moment of Inertia = {I_total_min/R_Moon/R_Moon/M_Moon}')
r_array_min, rho_array_min, p_array_min, g_array_min = compute_1D_profiles(R_Moon,rho_crust,R_mantle_min,rho_mantle_min,R_outer_core_min,rho_outer_core,R_inner_core_min,rho_inner_core,dr)

### Mean Model
print('-'*60)
print('Mean Model:')
R_outer_core_mean = 0.5*(R_outer_core_max+R_outer_core_min) # [m]
R_inner_core_mean,R_mantle_mean,rho_mantle_mean,I_total_mean = find_R_inner_core(rho_crust,rho_outer_core,rho_inner_core,thickness_crust,R_outer_core_mean)
print(f'Inner Core Radius = {R_inner_core_mean/1e+3} km')
print(f'Outer Core Radius = {R_outer_core_mean/1e+3} km')
print(f'Mantle Density = {rho_mantle_mean} kg/m^3')
print(f'Moment of Inertia = {I_total_mean/R_Moon/R_Moon/M_Moon}')
r_array_mean, rho_array_mean, p_array_mean, g_array_mean = compute_1D_profiles(R_Moon,rho_crust,R_mantle_mean,rho_mantle_mean,R_outer_core_mean,rho_outer_core,R_inner_core_mean,rho_inner_core,dr)

### Plot Results
cases = [
    ("Max (R_oc = 465.6 km)",  dict(R_crust=R_Moon, rho_crust=rho_crust, R_mantle=R_Moon-thickness_crust, rho_mantle=rho_mantle_max,
                  R_outer_core=R_outer_core_max, rho_outer_core=rho_outer_core,
                  R_inner_core=R_inner_core_max, rho_inner_core=rho_inner_core)),
    ("Mean (R_oc = 401.05 km)", dict(R_crust=R_Moon, rho_crust=rho_crust, R_mantle=R_Moon-thickness_crust, rho_mantle=rho_mantle_mean,
                  R_outer_core=R_outer_core_mean, rho_outer_core=rho_outer_core,
                  R_inner_core=R_inner_core_mean, rho_inner_core=rho_inner_core)),
    ("Min (R_oc = 336.5 km",  dict(R_crust=R_Moon, rho_crust=rho_crust, R_mantle=R_Moon-thickness_crust, rho_mantle=rho_mantle_min,
                  R_outer_core=R_outer_core_min, rho_outer_core=rho_outer_core,
                  R_inner_core=R_inner_core_min, rho_inner_core=rho_inner_core)),
]

fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
# fig.suptitle('Four-Layer Homogeneous Moon 1D Profiles (Max/Mean/Min $R_{oc}$)')

linestyles = ["--", "-", ":"]

for (ls, (label, prm)) in zip(linestyles, cases):
    r, rho, p, g = compute_1D_profiles(dr=dr, **prm)

    axes[0].plot(rho, r/1e3, linestyle=ls, label=label)
    axes[1].plot(g,   r/1e3, linestyle=ls, label=label)
    axes[2].plot(p/1e9, r/1e3, linestyle=ls, label=label)

axes[0].set_xlabel('Density [kg/m$^3$]')
axes[0].set_ylabel('Radius [km]')
axes[0].grid(True)

axes[1].set_xlabel('Gravity [m/s$^2$]')
axes[1].grid(True)

axes[2].set_xlabel('Pressure [GPa]')
axes[2].grid(True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.show()



# Save as Model M1
r_array_M1 = r_array_mean.copy()
rho_array_M1 = rho_array_mean.copy()
p_array_M1 = p_array_mean.copy()
g_array_M1 = g_array_mean.copy()















##### Model M2 #####
def layer_rayleigh_number(dT_tot,R_inner,R_outer,rho,alpha,k,Cp,eta):
    ### Layer thickness
    D = R_outer - R_inner
    ### Layer dT
    dT = dT_tot*D/R_Moon
    ### Average Gravity in Layer
    mask = (r_array >= R_inner) & (r_array <= R_outer)
    g = g_array[mask].mean()
    ### Compute Rayleigh Number
    kappa = k/rho/Cp
    Ra = rho*alpha*g*dT*D**3/kappa/eta
    return Ra

def compute_Ra_layers(T_surface,T_core,
                      R_inner_core,R_outer_core,R_mantle,R_crust,
                      rho_inner_core,rho_outer_core,rho_mantle,rho_crust,
                      alpha_inner_core,alpha_outer_core,alpha_mantle,alpha_crust,
                      k_inner_core,k_outer_core,k_mantle,k_crust,
                      Cp_inner_core,Cp_outer_core,Cp_mantle,Cp_crust,
                      eta_inner_core,eta_outer_core,eta_mantle,eta_crust):
    
    dT_tot = T_core - T_surface
    results = {}
    ### Inner Core
    Ra_inner_core = layer_rayleigh_number(dT_tot,0,R_inner_core,rho_inner_core,alpha_inner_core,k_inner_core,Cp_inner_core,eta_inner_core)
    results["inner_core"]={"Ra":float(Ra_inner_core),"regime":"convective" if Ra_inner_core > 1e+3 else "conductive"}
    ### Outer Core
    Ra_outer_core = layer_rayleigh_number(dT_tot,R_inner_core,R_outer_core,rho_outer_core,alpha_outer_core,k_outer_core,Cp_outer_core,eta_outer_core)
    results["outer_core"]={"Ra":float(Ra_outer_core),"regime":"convective" if Ra_outer_core > 1e+3 else "conductive"}
    ### Mantle
    Ra_mantle = layer_rayleigh_number(dT_tot,R_outer_core,R_mantle,rho_mantle,alpha_mantle,k_mantle,Cp_mantle,eta_mantle)
    results["mantle"]={"Ra":float(Ra_mantle),"regime":"convective" if Ra_mantle > 1e+3 else "conductive"}
    ### Crust
    Ra_crust = layer_rayleigh_number(dT_tot,R_mantle,R_crust,rho_crust,alpha_crust,k_crust,Cp_crust,eta_crust)
    results["crust"]={"Ra":float(Ra_crust),"regime":"convective" if Ra_crust > 1e+3 else "conductive"}

    return results

def integrate_adiabat(r_array,g_array,R_inner,R_outer,T_inner,alpha,Cp):
    T = np.full_like(r_array,np.nan)
    mask = (r_array >= R_inner) & (r_array <= R_outer)
    idx = np.where(mask)[0]
    T[idx[0]] = T_inner
    for i in idx[:-1]:
        dr = r_array[i+1] - r_array[i]
        dTdr = -1.*alpha*g_array[i]*T[i]/Cp
        T[i+1] = T[i] + dTdr*dr
    return T
    
def integrate_conduction(r_array,R_inner,R_outer,T_inner,T_outer):
    T = np.full_like(r_array,np.nan)
    mask = (r_array >= R_inner) & (r_array <= R_outer)
    x = (r_array[mask] - R_inner)/(R_outer - R_inner)
    T[mask] = T_inner + x*(T_outer-T_inner)
    return T

def thermal_profile_1D(T_core,R_inner_core,R_outer_core,R_mantle,R_crust,
                       alpha_inner_core,alpha_outer_core,alpha_mantle,alpha_crust,
                       Cp_inner_core,Cp_outer_core,Cp_mantle,Cp_crust):
    T_array = np.zeros_like(r_array)
    
    # Inner core
    mask_inner_core = (r_array <= R_inner_core)
    T_array[mask_inner_core] = T_core
    T_inner_core_outer = T_core
    
    # Outer core
    T_outer_core = integrate_adiabat(r_array,g_array,R_inner_core,R_outer_core,
                                     T_inner_core_outer,alpha_outer_core,Cp_outer_core)
    valid_outer_core = ~np.isnan(T_outer_core)
    T_array[valid_outer_core] = T_outer_core[valid_outer_core]
    idx_outer_core = np.where((r_array >= R_inner_core) & (r_array <= R_outer_core))[0]
    T_outer_core_outer = T_outer_core[idx_outer_core[-1]]  
    
    # Mantle
    T_mantle = integrate_adiabat(r_array,g_array,R_outer_core,R_mantle,
                                 T_outer_core_outer,alpha_mantle,Cp_mantle)
    valid_mantle = ~np.isnan(T_mantle)
    T_array[valid_mantle] = T_mantle[valid_mantle]
    idx_mantle = np.where((r_array >= R_outer_core) & (r_array <= R_mantle))[0]
    T_mantle_outer = T_mantle[idx_mantle[-1]]
    
    # Crust
    T_crust = integrate_conduction(r_array,R_mantle,R_crust,T_mantle_outer,T_surface)
    valid_crust = ~np.isnan(T_crust)
    T_array[valid_crust] = T_crust[valid_crust]
    
    return T_array

def density_from_temp_pressure(rho_0,alpha,K,T,p,T_0=0,p_0=0):
    rho = rho_0 * (1- alpha*(T-T_0) + ((p-p_0)/K))
    return rho

def build_layer_params(r_array,
                       R_inner_core,R_outer_core,R_mantle,R_crust,
                       rho0_inner,rho0_outer,rho0_mantle,rho0_crust,
                       alpha_inner,alpha_outer,alpha_mantle,alpha_crust,
                       K_inner,K_outer,K_mantle,K_crust):
    rho0   = np.zeros_like(r_array)
    alphaT = np.zeros_like(r_array)
    K      = np.zeros_like(r_array)

    mask_inner  = (r_array <= R_inner_core)
    mask_outer  = (r_array > R_inner_core) & (r_array <= R_outer_core)
    mask_mantle = (r_array > R_outer_core) & (r_array <= R_mantle)
    mask_crust  = (r_array > R_mantle)

    rho0[mask_inner]   = rho0_inner
    rho0[mask_outer]   = rho0_outer
    rho0[mask_mantle]  = rho0_mantle
    rho0[mask_crust]   = rho0_crust

    alphaT[mask_inner]   = alpha_inner
    alphaT[mask_outer]   = alpha_outer
    alphaT[mask_mantle]  = alpha_mantle
    alphaT[mask_crust]   = alpha_crust

    K[mask_inner]   = K_inner
    K[mask_outer]   = K_outer
    K[mask_mantle]  = K_mantle
    K[mask_crust]   = K_crust

    return rho0, alphaT, K

def iterate_structure_linear_density(r_array,T_array,rho0_array,alpha_array,K_array,n_iter=5,tol=1e-4):
    dr = r_array[1] - r_array[0]
    N  = len(r_array)

    # Initial Guess
    rho = rho0_array.copy()

    for it in range(n_iter):
        # Upward for Mass
        M = np.zeros_like(r_array)
        for i in range(1, N):
            r_mid  = 0.5*(r_array[i] + r_array[i-1])
            rho_mid = 0.5*(rho[i] + rho[i-1])
            dM = 4.0*np.pi * r_mid**2 * rho_mid * dr
            M[i] = M[i-1] + dM

        # Gravity
        g = np.zeros_like(r_array)
        g[1:] = G * M[1:] / (r_array[1:]**2)  # g(0)=0

        # Downward for Pressure
        P = np.zeros_like(r_array)
        P[-1] = 0.0  # surface boundary condition
        for i in range(N-2, -1, -1):
            rho_mid = 0.5*(rho[i] + rho[i+1])
            g_mid   = 0.5*(g[i] + g[i+1])
            dP = -rho_mid * g_mid * dr
            P[i] = P[i+1] - dP   # stepping inward

        # Update Density
        rho_new = density_from_temp_pressure(rho0_array, alpha_array, K_array, T_array, P)

        rel_change = np.max(np.abs(rho_new - rho) / rho0_array)
        rho = rho_new
        if rel_change < tol:
            break

    return rho, M, g, P

def moment_of_inertia(r_array,rho_array):
    I = 0.
    for i in range(1, len(r_array)):
        rho_mid = 0.5*(rho_array[i] + rho_array[i-1])
        dI = shell_MoI(rho_mid,r_array[i],r_array[i-1])
        I += dI
    return I

### Set constants
T_surface = 246 # [K]
R_inner_core = R_inner_core_mean
R_outer_core = R_outer_core_mean
R_mantle = R_mantle_mean
R_crust = R_Moon

rho_mantle = rho_mantle_mean

alpha_inner_core = 1e-5 # [K^-1]
alpha_outer_core = 1e-4 # [K^-1]
alpha_mantle = 2e-5#4e-5 # [K^-1]
alpha_crust = 4e-5 # [K^-1]

k_inner_core = 40. # [W.m^-1.K^-1]
k_outer_core = 40. # [W.m^-1.K^-1]
k_mantle = 3.#4. # [W.m^-1.K^-1]
k_crust = 2. # [W.m^-1.K^-1]

Cp_inner_core = 850. # [J.kg^-1.K^-1]
Cp_outer_core = 850. # [J.kg^-1.K^-1]
Cp_mantle = 1000.#1400. # [J.kg^-1.K^-1]
Cp_crust = 850. # [J.kg^-1.K^-1]

eta_inner_core = 1e+23 # [Pa.s]
eta_outer_core = 1e-2 # [Pa.s]
eta_mantle = 1e+21 # [Pa.s]
eta_crust = 1e+23 # [Pa.s]

K_inner_core = 1.7e+11 # [Pa]
K_outer_core = 1.4e+11 # [Pa]
K_mantle = 1.3e+11 # [Pa]
K_crust = 7.5e+10 # [Pa]

### Max Model
T_core_max = 1900 # [K]
results_max = compute_Ra_layers(T_surface,T_core_max,
                      R_inner_core,R_outer_core,R_mantle,R_crust,
                      rho_inner_core,rho_outer_core,rho_mantle,rho_crust,
                      alpha_inner_core,alpha_outer_core,alpha_mantle,alpha_crust,
                      k_inner_core,k_outer_core,k_mantle,k_crust,
                      Cp_inner_core,Cp_outer_core,Cp_mantle,Cp_crust,
                      eta_inner_core,eta_outer_core,eta_mantle,eta_crust)
print('-'*60)
print('Max Model:')
for layer, info in results_max.items():
    print(f"{layer:11s} Ra = {info['Ra']:.3e}  -> {info['regime']}")
T_array_max = thermal_profile_1D(T_core_max,R_inner_core,R_outer_core,R_mantle,R_crust,
                                 alpha_inner_core,alpha_outer_core,alpha_mantle,alpha_crust,
                                 Cp_inner_core,Cp_outer_core,Cp_mantle,Cp_crust)

### Min Model
T_core_min = 1300 # [K]
results_min = compute_Ra_layers(T_surface,T_core_min,
                      R_inner_core,R_outer_core,R_mantle,R_crust,
                      rho_inner_core,rho_outer_core,rho_mantle,rho_crust,
                      alpha_inner_core,alpha_outer_core,alpha_mantle,alpha_crust,
                      k_inner_core,k_outer_core,k_mantle,k_crust,
                      Cp_inner_core,Cp_outer_core,Cp_mantle,Cp_crust,
                      eta_inner_core,eta_outer_core,eta_mantle,eta_crust)
print('-'*60)
print('Min Model:')
for layer, info in results_min.items():
    print(f"{layer:11s} Ra = {info['Ra']:.3e}  -> {info['regime']}")
T_array_min = thermal_profile_1D(T_core_min,R_inner_core,R_outer_core,R_mantle,R_crust,
                                 alpha_inner_core,alpha_outer_core,alpha_mantle,alpha_crust,
                                 Cp_inner_core,Cp_outer_core,Cp_mantle,Cp_crust)

### Mean Model
T_core_mean = 0.5*(T_core_max+T_core_min) # [K]
results_mean = compute_Ra_layers(T_surface,T_core_mean,
                      R_inner_core,R_outer_core,R_mantle,R_crust,
                      rho_inner_core,rho_outer_core,rho_mantle,rho_crust,
                      alpha_inner_core,alpha_outer_core,alpha_mantle,alpha_crust,
                      k_inner_core,k_outer_core,k_mantle,k_crust,
                      Cp_inner_core,Cp_outer_core,Cp_mantle,Cp_crust,
                      eta_inner_core,eta_outer_core,eta_mantle,eta_crust)
print('-'*60)
print('Mean Model:')
for layer, info in results_mean.items():
    print(f"{layer:11s} Ra = {info['Ra']:.3e}  -> {info['regime']}")
T_array_mean = thermal_profile_1D(T_core_mean,R_inner_core,R_outer_core,R_mantle,R_crust,
                                 alpha_inner_core,alpha_outer_core,alpha_mantle,alpha_crust,
                                 Cp_inner_core,Cp_outer_core,Cp_mantle,Cp_crust)
    
##### Plot Results #####
fig,axes=plt.subplots(1,3,figsize=(10,6),sharey=False)
fig.suptitle('Four-Layer Homogeneous Density Moon')
### Max
axes[0].plot(T_array_max,r_array/1e+3)
axes[0].set_title('Max Core, T = 1900 K')
axes[0].set_xlabel('Temperature [K]')
axes[0].set_xlim(0,2000)
axes[0].set_ylabel('Radius [km]')
axes[0].grid(True)
### Mean
axes[1].plot(T_array_mean,r_array/1e+3)
axes[1].set_title('Mean Core, T = 1600 K')
axes[1].set_xlabel('Temperature [K]')
axes[1].set_xlim(0,2000)
axes[1].set_ylabel('Radius [km]')
axes[1].grid(True)
### Min
axes[2].plot(T_array_min,r_array/1e+3)
axes[2].set_title('Min Core, T = 1300 K')
axes[2].set_xlabel('Temperature [K]')
axes[2].set_xlim(0,2000)
axes[2].set_ylabel('Radius [km]')
axes[2].grid(True)

plt.tight_layout()
plt.show()

### Layer Parameters
rho0_array, alpha_array, K_array = build_layer_params(
    r_array,
    R_inner_core, R_outer_core, R_mantle, R_Moon,
    rho_inner_core, rho_outer_core, rho_mantle, rho_crust,
    alpha_inner_core, alpha_outer_core, alpha_mantle, alpha_crust,
    K_inner_core, K_outer_core, K_mantle, K_crust)

### Max Model
rho_max, M_max, g_max, p_max = iterate_structure_linear_density(
    r_array, T_array_max,
    rho0_array, alpha_array, K_array,
    n_iter=5, tol=1e-4)
I_max = moment_of_inertia(r_array, rho_max)
dT_max = (T_array_max[-1]-T_array_max[-2])/dr
q_max = dT_max * -k_crust

print('-'*60)
print("Max Model:")
print(f"Mass model = {M_max[-1]:.3e} kg")
print(f"Mass obs   = {M_Moon:.3e} kg")
print(f"ΔM/M_obs   = {(M_max[-1]-M_Moon)/M_Moon*100:.2f} %")
print(f"I_model    = {I_max:.3e} kg m^2")
print(f"I_obs      = {I_Moon:.3e} kg m^2")
print(f"ΔI/I_obs   = {(I_max-I_Moon)/I_Moon*100:.2f} %")
print(f"Surface Q  = {q_max*1000:.2f} mW/m^2")


### Mean Model
rho_mean, M_mean, g_mean, p_mean = iterate_structure_linear_density(
    r_array, T_array_mean,
    rho0_array, alpha_array, K_array,
    n_iter=5, tol=1e-4)
I_mean = moment_of_inertia(r_array, rho_mean)
dT_mean = (T_array_mean[-1]-T_array_mean[-2])/dr
q_mean = dT_mean * -k_crust

print('-'*60)
print("Mean Model:")
print(f"Mass model = {M_mean[-1]:.3e} kg")
print(f"Mass obs   = {M_Moon:.3e} kg")
print(f"ΔM/M_obs   = {(M_mean[-1]-M_Moon)/M_Moon*100:.2f} %")
print(f"I_model    = {I_mean:.3e} kg m^2")
print(f"I_obs      = {I_Moon:.3e} kg m^2")
print(f"ΔI/I_obs   = {(I_mean-I_Moon)/I_Moon*100:.2f} %")
print(f"Surface Q  = {q_mean*1000:.2f} mW/m^2")


### Min Model
rho_min, M_min, g_min, p_min = iterate_structure_linear_density(
    r_array, T_array_min,
    rho0_array, alpha_array, K_array,
    n_iter=5, tol=1e-4)
I_min = moment_of_inertia(r_array, rho_min)
dT_min = (T_array_min[-1]-T_array_min[-2])/dr
q_min = dT_min * -k_crust

print('-'*60)
print("Min Model:")
print(f"Mass model = {M_min[-1]:.3e} kg")
print(f"Mass obs   = {M_Moon:.3e} kg")
print(f"ΔM/M_obs   = {(M_min[-1]-M_Moon)/M_Moon*100:.2f} %")
print(f"I_model    = {I_min:.3e} kg m^2")
print(f"I_obs      = {I_Moon:.3e} kg m^2")
print(f"ΔI/I_obs   = {(I_min-I_Moon)/I_Moon*100:.2f} %")
print(f"Surface Q  = {q_min*1000:.2f} mW/m^2")


### Plot Results
fig,axs = plt.subplots(1, 4, figsize=(12, 6), sharey=True)

axs[0].plot(rho_max, r_array/1e3, linestyle='--', label="Max (T_core=1900 K)")
axs[0].plot(rho_mean, r_array/1e3, linestyle='-', label="Mean (T_core=1600 K)")
axs[0].plot(rho_min, r_array/1e3, linestyle=':', label="Min (T_core=1300 K)")
axs[1].plot(g_max, r_array/1e3, linestyle='--')
axs[1].plot(g_mean, r_array/1e3, linestyle='-')
axs[1].plot(g_min, r_array/1e3, linestyle=':')
axs[2].plot(p_max/1e9, r_array/1e3, linestyle='--')
axs[2].plot(p_mean/1e9, r_array/1e3, linestyle='-')
axs[2].plot(p_min/1e9, r_array/1e3, linestyle=':')
axs[3].plot(T_array_max, r_array/1e3, linestyle='--')
axs[3].plot(T_array_mean, r_array/1e3, linestyle='-')
axs[3].plot(T_array_min, r_array/1e3, linestyle=':')

axs[0].set_xlabel("Density [kg/m³]")
axs[1].set_xlabel("Gravity [m/s²]")
axs[2].set_xlabel("Pressure [GPa]")
axs[3].set_xlabel("Temperature [K]")

axs[0].set_ylabel("Radius [km]")
for ax in axs:
    ax.grid(True)
    
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.show()

# Save as Model M2
r_array_M2 = r_array_mean.copy()
rho_array_M2 = rho_mean.copy()
p_array_M2 = p_mean.copy()
g_array_M2 = g_mean.copy()
T_array_M2 = T_array_mean.copy()















##### Model M3 #####
def get_temperature_at_radius(T_array, r_array, R_target):
    idx = np.argmin(np.abs(r_array - R_target))
    return T_array[idx]

R_Moon_obs = R_Moon
M_Moon_obs = M_Moon
I_factor_obs = I_Moon_norm

# # Use Mean Values from Model M1
R_ic = R_inner_core_mean
R_oc = R_outer_core_mean
R_ma = R_mantle_mean
R_surface = R_Moon_obs

import numpy as np
import matplotlib.pyplot as plt
from burnman import Mineral, PerplexMaterial, Layer, Planet, Composition, minerals
from burnman.tools.chemistry import formula_mass

# Compositions from midpoints of Hirose et al. (2021), ignoring carbon and hydrogen
inner_core_composition = Composition({'Fe': 94.4, 'Ni': 5., 'Si': 0.55, 'O': 0.05}, 'weight')
outer_core_composition = Composition({'Fe': 90., 'Ni': 5., 'Si': 2., 'O': 3.}, 'weight')


for c in [inner_core_composition, outer_core_composition]:
    c.renormalize('atomic', 'total', 1.)

inner_core_elemental_composition = dict(inner_core_composition.atomic_composition)
outer_core_elemental_composition = dict(outer_core_composition.atomic_composition)
inner_core_molar_mass = formula_mass(inner_core_elemental_composition)
outer_core_molar_mass = formula_mass(outer_core_elemental_composition)

icb_radius = R_ic
inner_core = Layer('inner core', radii=np.linspace(0., icb_radius, 21))

hcp_iron = minerals.SE_2015.hcp_iron()
params = hcp_iron.params

params['name'] = 'modified solid iron'
params['formula'] = inner_core_elemental_composition
params['molar_mass'] = inner_core_molar_mass
delta_V = 2.0e-7

inner_core_material = Mineral(params=params,
                              property_modifiers=[['linear',
                                                   {'delta_E': 0.,
                                                    'delta_S': 0.,
                                                    'delta_V': delta_V}]])

# check that the new inner core material does what we expect:
hcp_iron.set_state(200.e9, 4000.)
inner_core_material.set_state(200.e9, 4000.)
assert np.abs(delta_V - (inner_core_material.V - hcp_iron.V)) < 1.e-12

inner_core.set_material(inner_core_material)

inner_core.set_temperature_mode('adiabatic')

cmb_radius = R_oc
outer_core = Layer('outer core', radii=np.linspace(icb_radius, cmb_radius, 21))

liq_iron = minerals.SE_2015.liquid_iron()
params = liq_iron.params

params['name'] = 'modified liquid iron'
params['formula'] = outer_core_elemental_composition
params['molar_mass'] = outer_core_molar_mass
delta_V = -2.3e-7
outer_core_material = Mineral(params=params,
                              property_modifiers=[['linear',
                                                   {'delta_E': 0.,
                                                    'delta_S': 0.,
                                                    'delta_V': delta_V}]])

# check that the new inner core material does what we expect:
liq_iron.set_state(200.e9, 4000.)
outer_core_material.set_state(200.e9, 4000.)
assert np.abs(delta_V - (outer_core_material.V - liq_iron.V)) < 1.e-12

outer_core.set_material(outer_core_material)

outer_core.set_temperature_mode('adiabatic')

from burnman import BoundaryLayerPerturbation

lab_radius = R_ma # top of convecting mantle = base of crust
lab_temperature = get_temperature_at_radius(T_array_mean,r_array,lab_radius)

convecting_mantle_radii = np.linspace(cmb_radius, lab_radius, 101)
convecting_mantle = Layer('convecting mantle', radii=convecting_mantle_radii)

# Import a low resolution PerpleX data table.
fname = r"C:\Users\chase\burnman-go\burnman\tutorial\data\pyrolite_perplex_table_lo_res.dat"
pyrolite = PerplexMaterial(fname, name='pyrolite')
convecting_mantle.set_material(pyrolite)

T_cmb_mean = get_temperature_at_radius(T_array_mean,r_array,cmb_radius)
dT_mantle_mean = T_cmb_mean - lab_temperature

# Here we add a thermal boundary layer perturbation.
tbl_perturbation = BoundaryLayerPerturbation(radius_bottom=cmb_radius,
                                             radius_top=lab_radius,
                                             rayleigh_number=results_mean["mantle"]["Ra"],
                                             temperature_change=dT_mantle_mean,
                                             boundary_layer_ratio=60./900.)

# Onto this perturbation, we add a linear superadiabaticity term.
dT_sup_Moon = 0.
dT_superadiabatic = dT_sup_Moon*(convecting_mantle_radii - convecting_mantle_radii[-1])/(convecting_mantle_radii[0] - convecting_mantle_radii[-1])

convecting_mantle_tbl = (tbl_perturbation.temperature(convecting_mantle_radii)
                         + dT_superadiabatic)

convecting_mantle.set_temperature_mode('perturbed-adiabatic',
                                       temperatures=convecting_mantle_tbl)

planet_radius = R_surface
surface_temperature = T_surface
andesine = minerals.SLB_2011.plagioclase(molar_fractions=[0.4, 0.6])
crust = Layer('crust', radii=np.linspace(lab_radius, planet_radius, 11))
crust.set_material(andesine)
crust.set_temperature_mode('user-defined',
                           np.linspace(lab_temperature,
                                       surface_temperature, 11))

planet_Moon = Planet('Moon',
                    [inner_core, outer_core,
                     convecting_mantle,
                     crust], verbose=True)
planet_Moon.make()

obs_mass = M_Moon_obs
obs_moment_of_inertia_factor = I_factor_obs
dM = (planet_Moon.mass - obs_mass)/obs_mass * 100.
dI = (planet_Moon.moment_of_inertia_factor - obs_moment_of_inertia_factor)/obs_moment_of_inertia_factor * 100.

print(f'Mass = {planet_Moon.mass:.3e} (Obs = {obs_mass:.3e})')
print(f'Mass misfit: {dM:+.2f} %')
print(f'Moment of inertia factor= {planet_Moon.moment_of_inertia_factor:.4f} '
      f'(Obs = {obs_moment_of_inertia_factor:.4f})')
print(f'MoI factor misfit: {dI:+.2f} %')

print('Layer mass fractions:')
for layer in planet_Moon.layers:
    print(f'{layer.name}: {layer.mass / planet_Moon.mass:.3f}')
    
fig = plt.figure(figsize=(8, 5))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]


bounds = np.array([[layer.radii[0]/1.e3, layer.radii[-1]/1.e3]
                   for layer in planet_Moon.layers])
maxy = [
    1.1 * np.max(planet_Moon.density / 1.e3),   #
    1.1 * np.max(planet_Moon.pressure / 1.e9),  
    1.1 * np.max(planet_Moon.gravity),          
    1.1 * np.max(planet_Moon.temperature)       
]

for bound in bounds:
    for i in range(4):
        ax[i].fill_betweenx([0., maxy[i]],
                            [bound[0], bound[0]],
                            [bound[1], bound[1]], alpha=0.2)

ax[0].plot(planet_Moon.radii / 1.e3, planet_Moon.density / 1.e3)
ax[0].set_ylabel('Density ($10^3$ kg/m$^3$)')

# Make a subplot showing the calculated pressure profile
ax[1].plot(planet_Moon.radii / 1.e3, planet_Moon.pressure / 1.e9)
ax[1].set_ylabel('Pressure (GPa)')

# Make a subplot showing the calculated gravity profile
ax[2].plot(planet_Moon.radii / 1.e3, planet_Moon.gravity)
ax[2].set_ylabel('Gravity (m/s$^2)$')
ax[2].set_xlabel('Radius (km)')

# Make a subplot showing the calculated temperature profile
ax[3].plot(planet_Moon.radii / 1.e3, planet_Moon.temperature)
ax[3].set_ylabel('Temperature (K)')
ax[3].set_xlabel('Radius (km)')
ax[3].set_ylim(0.,)

for i in range(2):
    ax[i].set_xticklabels([])
for i in range(4):
    ax[i].set_xlim(0., max(planet_Moon.radii) / 1.e3)
    ax[i].set_ylim(0., maxy[i])

fig.set_layout_engine('tight')
plt.show()

# Save as Model M3
r_array_M3 = planet_Moon.radii.copy()
rho_array_M3 = planet_Moon.density.copy()
p_array_M3 = planet_Moon.pressure.copy()
g_array_M3 = planet_Moon.gravity.copy()
T_array_M3 = planet_Moon.temperature.copy()















##### Comparison of Models #####
fig = plt.figure(figsize=(8, 5))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]


bounds = np.array([[layer.radii[0]/1.e3, layer.radii[-1]/1.e3]
                   for layer in planet_Moon.layers])
maxy = [
    1.1 * np.max(planet_Moon.density / 1.e3),   #
    1.1 * np.max(planet_Moon.pressure / 1.e9),  
    1.1 * np.max(planet_Moon.gravity),          
    1.1 * np.max(planet_Moon.temperature)       
]

for bound in bounds:
    for i in range(4):
        ax[i].fill_betweenx([0., maxy[i]],
                            [bound[0], bound[0]],
                            [bound[1], bound[1]], alpha=0.2)

ax[0].plot(r_array_M1 / 1.e3, rho_array_M1 / 1.e3, label = 'Model M1', color = 'Blue')
ax[0].plot(r_array_M2 / 1.e3, rho_array_M2 / 1.e3, label = 'Model M2', color = 'Red')
ax[0].plot(r_array_M3 / 1.e3, rho_array_M3 / 1.e3, label = 'Model M3', color = 'Black')
ax[0].set_ylabel('Density ($10^3$ kg/m$^3$)')
ax[0].legend()

# Make a subplot showing the calculated pressure profile
ax[1].plot(r_array_M1 / 1.e3, p_array_M1 / 1.e9, color = 'Blue')
ax[1].plot(r_array_M2 / 1.e3, p_array_M2 / 1.e9, color = 'Red')
ax[1].plot(r_array_M3 / 1.e3, p_array_M3 / 1.e9, color = 'Black')
ax[1].set_ylabel('Pressure (GPa)')

# Make a subplot showing the calculated gravity profile
ax[2].plot(r_array_M1 / 1.e3, g_array_M1, color = 'Blue')
ax[2].plot(r_array_M2 / 1.e3, g_array_M2, color = 'Red')
ax[2].plot(r_array_M3 / 1.e3, g_array_M3, color = 'Black')
ax[2].set_ylabel('Gravity (m/s$^2)$')
ax[2].set_xlabel('Radius (km)')

# Make a subplot showing the calculated temperature profile
# ax[3].plot(r_array_M1 / 1.e3, T_array_M1, color = 'Blue')
ax[3].plot(r_array_M2 / 1.e3, T_array_M2, color = 'Red')
ax[3].plot(r_array_M3 / 1.e3, T_array_M3, color = 'Black')
ax[3].set_ylabel('Temperature (K)')
ax[3].set_xlabel('Radius (km)')
ax[3].set_ylim(0.,)

for i in range(2):
    ax[i].set_xticklabels([])
for i in range(4):
    ax[i].set_xlim(0., max(planet_Moon.radii) / 1.e3)
    ax[i].set_ylim(0., maxy[i])

fig.set_layout_engine('tight')
plt.show()