#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:20:27 2024

@author: marcello.andolina
"""


import functions1 as fu
import numpy as np
from scipy.special import factorial, gamma, genlaguerre
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import rc, rcParams
import time
from tqdm import tqdm

hbar=6.654/(2*np.pi)*10**-34
e=1.60218**-19
mass=7.1*10**-26





# # Parameters for the Morse potential
De1 = 0.474  # Depth of the potential well
Re1 = 4.23  # Equilibrium bond length
#we1 =1.31 ## Width of the potential well
we1 = 0.05
De2 = 0.846  # Depth of the potential well
Re2 = 4.2  # Equilibrium bond length
#we2 =1.763# Width of the potential well
we2 = 0.05
Esp=1.212
lambd1 = fu.lambd(De1,we1)
lambd2 = fu.lambd(De2,we2)
nuM1 = int(np.floor(lambd1 +0.5))
nuM2 = int(np.floor(lambd2 + 0.5))
# print(nuM1)
# print(nuM2)
Reg=1

lambd1 = fu.lambd(De1,we1)
lambd2 = fu.lambd(De2,we2)

# Calculate energy eigenvalues
energy_eigenvalues = fu.omega_nu(lambd1) * De1
energy_eigenvalues1 = fu.omega_nu(lambd2) * De2+Esp

dim1=len(energy_eigenvalues)
dim2=len(energy_eigenvalues1)
Amunu=fu.Afc(lambd1, Re1, we1, lambd2, Re2, we2)
GammaMatrix=fu.gammaMatrix(dim1,dim2,Reg,De1, Re1, we1, De2, Re2, we2,Esp)

t=1
Reg=1
Omega_0=1
omega_L=1.2


states = [f'{i}s' for i in range(dim1)] + [f'{i}p' for i in range(dim2)]
rho_initial = np.zeros((dim1+dim2,dim1+dim2), dtype=complex)
rho_initial[0, 0] = 0
rho_initial[1, 1] = 0
rho_initial[2, 2] = 0
rho_initial[dim1-1, dim1-1] = 1



tau = 200
t_eval = np.linspace(0, tau,500)
N_t=len(t_eval)

Reg=1

# Checking that the sum of all populations is one over time
populations_sum = []


start_time = time.time()
#solution0 = fu.solve_dyn_vect(rho_initial, t_eval, Reg, De1, Re1, we1, De2, Re2, we2, Esp,Omega_0,omega_L, 3)
solution0 = fu.solve_dyn_vect(rho_initial, t_eval, Reg, De1, Re1, we1, De2, Re2, we2, Esp,Omega_0,omega_L,1)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Computation Time: {elapsed_time:.4f} seconds")




N=dim1+dim2
# Calculate populations and trace
populations_sum = np.zeros(N_t)
populations0 = solution0.y[: N, :]
# Calculate populations and trace


# Plot populations and trace
plt.figure(figsize=(10, 6))

for i in range(N):
    state_population = populations0[i, :]
    plt.plot(t_eval, state_population, label=f'Population in state {states[i]}')
    populations_sum += state_population

plt.plot(t_eval, populations_sum, 'k--', label='Sum of populations', linewidth=2)

plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Population Dynamics of Electronic States Over Time')
plt.legend()
plt.grid(True)
plt.savefig('./Figs/Test.pdf', dpi=300)
plt.show()




