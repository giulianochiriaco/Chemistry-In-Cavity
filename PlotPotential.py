#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:35:17 2024

@author: marcello.andolina
"""

import numpy as np
import functions1 as fu
from scipy.special import factorial, gamma, genlaguerre
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, rcParams
from matplotlib.colors import LogNorm




# Example usage of the defined functions


# Parameters for the Morse potential
De1 = 0.474  # Depth of the potential well
Re1 = 4.23  # Equilibrium bond length
#we1 = 0.05 # Width of the potential well
we1 = 1.31 # Width of the potential well


De2 = 0.846  # Depth of the potential well
Re2 = 4.201 # Equilibrium bond length
#we2 = 0.05
we2 = 1.763# Width of the potential well
# Position vector R
R = np.linspace(2.9, 11, 500)
E_sp=1.212
# Calculate Morse potential
V1= fu.Vmorse(R, De1, Re1, we1)
V2 = fu.Vmorse(R, De2, Re2, we2)+1.212


# Calculate lambda for energy eigenvalues
lambd1 = fu.lambd(De1,we1)
lambd2 = fu.lambd(De2,we2)
# print(lambd1)
# print(lambd2)

# Calculate energy eigenvalues
energy_eigenvalues = fu.omega_nu(lambd1) * De1
energy_eigenvalues1 = fu.omega_nu(lambd2) * De2+E_sp



fig, ax = plt.subplots(frameon=True)
ax.figure.set_size_inches(w=6*1.3,h=6)
rc('axes', linewidth=1.2)
rc('lines', linewidth=1.2)
rc('text',usetex=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)  
plt.plot(R, V1,color='b' , linewidth=2,label='Morse Potential')
plt.plot(R, V2, color='r' , linewidth=2,label='Morse Potential')
# Plotting
plt.xlabel(r'$R [{\rm \AA}]$',fontsize=20)
plt.ylabel(r' $E$ [{\rm eV}]',fontsize=20)
xM=10.4
xm=2.9
plt.xlim([xm,xM])
plt.ylim([-2,2])
# Plot energy eigenvalues as horizontal lines
for energy in energy_eigenvalues:
    x_start=np.where(V1<energy)[0][0]
    x_end=np.where(V1<energy)[0][-1]
    plt.axhline(y=energy, xmin=(R[x_start]-xm)/(xM-xm), xmax=(R[x_end]-xm)/(xM-xm), color='b', linestyle='--',linewidth=0.15, label=f'Eigenvalue: {energy:.2f}')
    #plt.axhline(y=energy, xmin=x_start/500, xmax=x_end/500, color='r', linestyle='--',linewidth=0.5, label=f'Eigenvalue: {energy:.2f}')
for energy in energy_eigenvalues1:
    x_start=np.where(V2<energy)[0][0]
    x_end=np.where(V2<energy)[0][-1]
    plt.axhline(y=energy, xmin=(R[x_start]-xm)/(xM-xm), xmax=(R[x_end]-xm)/(xM-xm), color='r', linestyle='--', linewidth=0.15, label=f'Eigenvalue: {energy:.2f}')


#plt.legend(loc='best', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('./Figs/Potential.pdf', dpi=300)



# Compute the Franck-Condon factors matrix
Amunu =fu.Afc(lambd1, Re1, we1, lambd2, Re2, we2)
nuM1 = int(np.floor(lambd1 +0.5))
nuM2 = int(np.floor(lambd2 + 0.5))
Rv = np.linspace(0.000001, 10 * max(we1, we2) + max(Re1, Re2), 10000)  # creates a grid for numerical integration
 
 # Generate all psiR for nu1 and nu2
psi1 = np.array([fu.psiR(Rv, lambd1, nu, Re1, we1,1) for nu in range(nuM1 )])
psi2 = np.array([fu.psiR(Rv, lambd2, nu, Re2, we2,1) for nu in range(nuM2 )])
# Plot the first 5 wavefunctions for both potentials
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

rc('text', usetex=True) 
# Plot wavefunctions for potential 1
for nu in range(min(5, nuM1)):
    ax[0].plot(Rv, psi1[nu], label=f'nu={nu}')
ax[0].set_title('Wavefunctions for Potential 1')
ax[0].set_xlabel('R')
ax[0].set_ylabel('psi')
ax[0].legend()
ax[0].grid(True)
xM = 5
xm = 3.5
ax[0].set_xlim([xm, xM])

# Plot wavefunctions for potential 2
for nu in range(min(5, nuM2)):
    ax[1].plot(Rv, psi2[nu], label=f'nu={nu}')
ax[1].set_title('Wavefunctions for Potential 2')
ax[1].set_xlabel('R')
ax[1].set_ylabel('psi')
ax[1].legend()
ax[1].grid(True)
xM = 5
xm = 3.5
ax[1].set_xlim([xm, xM])

plt.tight_layout()
plt.savefig('./Figs/wavefunctions.pdf', dpi=300)
print(f"plot")
# Plot the Franck-Condon factors matrix
plt.figure(figsize=(8, 6))
rc('axes', linewidth=1.2)
rc('lines', linewidth=1.2)
rc('text', usetex=True) 
 
plt.imshow(np.abs(Amunu), cmap='viridis',aspect='auto', vmin=0, vmax=1)
#plt.imshow(np.abs(Amunu), cmap='viridis', aspect='auto', norm=LogNorm(vmin=10**(-9), vmax=np.abs(Amunu).max()))

plt.colorbar(label=r'$|A_{\mu,\nu}|$' )
plt.xlabel(r'$\nu$')
plt.ylabel(r'$\mu$')
plt.tight_layout()
plt.savefig('./Figs/FCF.pdf', dpi=300)
plt.show()
