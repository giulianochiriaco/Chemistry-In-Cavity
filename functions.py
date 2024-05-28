#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:18:34 2024

@author: marcello.andolina
"""


import numpy as np
from scipy.special import factorial, gamma, genlaguerre
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp,odeint
from matplotlib import rc, rcParams
import time
from tqdm import tqdm

def Vmorse(R, De, Re, we):
    """
    Define the Morse potential with depth De, equilibrium position Re, 
    and typical width we as a function of position R (input vector).
    """
    return De * (np.exp(-2 * (R - Re) / we) - 2 * np.exp(-(R - Re) / we))

def omega_nu(lambd):
    """
    Gives the vector of the energies of the bound vibrational states.
    Increasing order
    """
    nuv = np.arange(int(lambd + 0.5)) #different from the notes of Giuliano
    return -(lambd - nuv - 0.5)**2 / lambd**2

def psiR(R, lambd, nu, Re, we, norm=1):
    """
    Gives the eigenfunction of the bound state at mode nu as function 
    of radial coordinate R (vector as input) for a Morse potential 
    given by Re and we. lambd is sqrt(2*m*De)*we/hbar.
    """
   
    zv = 2 * lambd * np.exp(-(R - Re) / we)
   
    psi= psiz(zv, lambd, nu)

    
    # Compute the norm
    norm_factor = np.sqrt(np.sum(np.conjugate(psi) * psi))
        
    # Normalize the wavefunction
    return psi / norm_factor if norm else psi




def psiz(z, lambd, nu):
    """
    Gives the NON-normalized wavefunction of mode nu as function of 
    the new coordinate z.
    """
    z0= 2 * lambd 
    

    term1=np.exp(-z / 2)/np.exp(-z0 / 2)
    term2=genlaguerre(nu, 2 * lambd - 2 * nu - 1)(z)/genlaguerre(nu, 2 * lambd - 2 * nu - 1)(z0)
    term3=(z/z0)**(lambd - nu - 0.5)
   # N = np.sqrt((2 * lambd - 2 * nu - 1) * factorial(nu) / gamma(2 * lambd - nu))
   # Replace invalid values with small numbers or zero
    term1 = np.where(np.isfinite(term1), term1, 0)
    term2 = np.where(np.isfinite(term2), term2, 0)
    term3 = np.where(np.isfinite(term3), term3, 0)
    
    psi=term1*term2*term3
    return psi
    
   

# def Afc(lambd1, Re1, we1, lambd2, Re2, we2):
#     """
#     Gives a matrix (table) with the Franck-Condon factors between the vibrational levels
#     of two different Morse potentials. The Morse potentials are characterized by different
#     lambda (i.e., De), Re and we. Returns a matrix array with the Franck-Condon factors 
#     between all the vibrational levels.
#     """
#     nuM1 = int(np.floor(lambd1 - 0.5))
#     nuM2 = int(np.floor(lambd2 - 0.5))
#     Amunu = np.zeros((nuM1 + 1, nuM2 + 1))
#     Rv = np.linspace(0, 50 * max(we1, we2) + max(Re1, Re2), 10000) #creates a grid for numerical integration
#     for nu1 in range(nuM1 + 1):
#         psi1 = psiR(Rv, lambd1, nu1, Re1, we1) #creates the wavefunction \psi_nu1
#         N1 = np.sum(psi1 * np.conjugate(psi1)) #calculate the norm of \psi_nu1
#         for nu2 in range(nuM2 + 1):
#             psi2 = psiR(Rv, lambd2, nu2, Re2, we2) #creates the wavefunction \psi_nu2
#             N2 = np.sum(psi2 * np.conjugate(psi2)) #calculate the norm of \psi_nu2
#             Amunu[nu1, nu2] = np.sum(psi1 * psi2) / np.sqrt(N1 * N2) #calculate the scalar product of \psi_nu1 and \psi_nu2
#     return Amunu


def Afc(lambd1, Re1, we1, lambd2, Re2, we2):
    """
    Gives a matrix (table) with the Franck-Condon factors between the vibrational levels
    of two different Morse potentials. The Morse potentials are characterized by different
    lambda (i.e., De), Re, and we. Returns a matrix array with the Franck-Condon factors 
    between all the vibrational levels. 
    """
    nuM1 = int(np.floor(lambd1 +0.5))
    nuM2 = int(np.floor(lambd2 + 0.5))
    Rv = np.linspace(0, 50 * max(we1, we2) + max(Re1, Re2), 10000)  # creates a grid for numerical integration
    
    # Generate all psiR for nu1 and nu2
    psi1 = np.array([psiR(Rv, lambd1, nu, Re1, we1) for nu in range(nuM1 )])
    psi2 = np.array([psiR(Rv, lambd2, nu, Re2, we2) for nu in range(nuM2 )])
   
    
    # Compute the norms
   # N1 = np.sum(psi1 * np.conjugate(psi1), axis=1)
   # N2 = np.sum(psi2 * np.conjugate(psi2), axis=1)
    
    # Compute the scalar products
    scalar_products = np.einsum('ik,jk->ij', psi1, np.conjugate(psi2))
    
    # Compute Amunu
    Amunu = scalar_products# / np.sqrt(np.outer(N1, N2))
    

    
    return Amunu




def Ec_filter(Et,tv,r,wc):
    "Given an input vector of electric field Et, time vector tv, a cavity with reflectivity r and frequency wc"
    "returns the electric field inside the cavity as function of t, by summing the electric fields at earlier times reflected by the mirrors"
    Nt = len(tv)
    "Cut-off number of reflected fields summed"
    Nr = -int(20/np.log(r)) 
    dt = tv[1]-tv[0]
    "how many indices in the array one has to go back to find the first reflected field"
    dn = np.pi/(dt*wc)  
    out = np.zeros(Nt,dtype=complex)
    for n in np.arange(Nr):
        Eshift = np.zeros(Nt,dtype=complex)
        Dn = int(dn*n)
        if Dn<Nt:
            "shifted electric field"
            Eshift[Dn:] = Et[:Nt-Dn] 
            out = out +(-r)**n*Eshift
    return out
    
def Ec_Gaussian(tv,wL,tau,wc,r):
    "Given an input time vector tv, a cavity with reflectivity r and frequency wc"
    "a Gaussian pulse of main frequency wL and width tau"
    "returns the electric field inside the cavity as function of t, by summing the electric fields at earlier times reflected by the mirrors"
    Nr = -int(20/np.log(r))
    out = np.zeros(len(tv),dtype=complex)
    for n in range(Nr):
        out = out + (-r)**n*np.exp(-1j*wL*(tv-n*np.pi/wc))*np.exp(-(tv-n*np.pi/wc)**2/tau**2)
    return out

def gammaMatrix_Old(dim1,dim2,Reg):
    Ga = np.zeros((dim1, dim2))
    # Creates the A_munu matrix
    #
    # given that Reg is in Angstrom and \hbar\omega is in eV G0 in eV reads
    Ga0=1
    #Ga = np.ones((dim1, dim2))
    Ga=Ga0*np.eye(dim1, dim2)+0.5*np.ones((dim1, dim2))
    
    return Ga


def gammaMatrix(dim1,dim2,Reg,De1, Re1, we1, De2, Re2, we2,E_sp):
    Ga = np.zeros((dim1, dim2)) 
    # Creates the A_munu matrix
    lambd1 = lambd(De1,we1)
    lambd2 = lambd(De2,we2)
    Amunu=Afc(lambd1, Re1, we1, lambd2, Re2, we2)
    energy_eigenvalues = omega_nu(lambd1) * De1
    energy_eigenvalues1 = omega_nu(lambd2) * De2+E_sp
   
    # given that Reg is in Angstrom and \hbar\omega is in eV G0 in eV reads
    #OLD for loop computation of Gamma
    # for i in range (dim1):
    #     for j in range (dim2):
    #         Ga[i,j]=Amunu[i,j]*( energy_eigenvalues1[j]-  energy_eigenvalues[i])**3
    # Vectorized computation of Ga matrix
    energy_diff = energy_eigenvalues1[ np.newaxis,:] - energy_eigenvalues[:,np.newaxis]
    Ga += Amunu**2 * (energy_diff ** 3)
    
    return Ga
    



def E0(t):
    E0=1
    # 
    return E0


def omegaMuNu(t,dim1,dim2,Reg):
    Om = np.zeros((dim1, dim2), dtype=complex)
    # Om=Deg*E0(t)
    # Ensuring Hermitian by mirroring real components
    return Om


# def system_equations(t, y, dim1,dim2,Reg,lambd1, Re1, we1, lambd2, Re2, we2,Gamma):
#     """
#     Calculate the time derivative of rho
#     """
#     #I arrange the state in the following way. the states n-s are the first dim states, the states n-p the second dim states.
#     rho = y.reshape((dim1+dim2, dim1+dim2))
#     # Initialize the rate of change matrix with zeros
#     drho_dt = np.zeros_like(rho)
#     # Get the current omega matrix for this time step (insert omega(t, dim1,dim2))
#     Omega_t = omegaMuNu(t,dim1,dim2,Reg)
#     # Get the current gamma matrix
 

#     # Loop over each dimension to update population dynamics

    
#     for mu in range(dim1): 
#         for nu in range(dim2):
#           # Population update for s-states
#             drho_dt[mu, mu] += Gamma[mu, nu] * rho[dim1+nu, dim1+nu]-1j*(Omega_t[mu, nu]*rho[nu+dim1 , mu]-np.conj(Omega_t[mu, nu])*rho[mu , nu+dim1])
#             # Population update for p-states
#             drho_dt[nu + dim1, nu+ dim1] += - Gamma[mu, nu] * rho[nu + dim1, nu + dim1]-1j*(np.conj(Omega_t[mu, nu])*rho[mu , nu+dim1]-Omega_t[mu, nu]*rho[nu+dim1 , mu])
           
#             drho_dt[nu+dim1, mu]=-0.5*np.sum(Gamma[:, nu]*rho[nu+dim1, mu])-1j*(np.conj(Omega_t[mu, nu] )*rho[mu , mu]-np.conj(Omega_t[mu, nu])*rho[nu+dim1 , nu+dim1])
#             # The coherence terms must be the conjugate of their counterparts
#             drho_dt[ mu,nu+dim1]= np.conj(drho_dt[nu+dim1, mu])
         
    
#     return drho_dt.flatten()


def system_equations_opt_vect(t, y, dim1, dim2, Reg, De1, Re1, we1, De2, Re2, we2,Gamma):
    """
    Calculate the time derivative of rho
    """
    N=dim1+dim2
    rho=vector_to_rho(y, N)
    
    drho_dt = np.zeros_like(rho, dtype=complex)
    Omega_t = omegaMuNu(t, dim1, dim2, Reg)
 
    populations_s=np.zeros(dim1)
    populations_p=np.zeros(dim2)

    # Update population dynamics for s-states
    Gamma_rho_p_diag = np.einsum('ij,j->ij', Gamma, np.diag(rho[dim1:, dim1:]))
    populations_s=  Gamma_rho_p_diag.sum(axis=1)
    
    # Update population dynamics for p-states
    populations_p= -Gamma_rho_p_diag.sum(axis=0)
   
    # Set the first dim1 diagonal terms for s-states
    drho_dt[np.arange(dim1), np.arange(dim1)] = populations_s
   
    # Set the following dim2 diagonal terms for p-states
    drho_dt[np.arange(dim1, dim1 + dim2), np.arange(dim1, dim1 + dim2)] = populations_p
   
    # Update coherence terms
    Gamma_v=Gamma.sum(axis=0) #I calculate \sum_\mu^\prime \Gamma_\mu^\prime,\nu
    coherences = -0.5 * np.einsum('j,jk->jk', Gamma_v, rho[dim1:, :dim1])
    drho_dt[dim1:, :dim1] += coherences
    drho_dt[:dim1, dim1:] += np.conj(coherences.T)
    
    # Adding Hamiltonian dynamics
    # Coherent evolution term: -i [Omega, rho]
    Omega=np.zeros_like(rho)
    Omega[:dim1, dim1:] = Omega_t
    Omega[dim1:, :dim1] = np.conj(Omega_t.T)
    commutator = np.dot(Omega, rho) - np.dot(rho, Omega)
   
    drho_dt += -1j * commutator
    
    drho_dt_vector=rho_to_vector(drho_dt)
    return  drho_dt_vector




def solve_dyn_vect(rho_initial, t, dim1, dim2, Reg, De1, Re1, we1, De2, Re2, we2, Esp, method_code):
    # Map numerical values to integration methods
    method_mapping = {
        1: 'RK45',
        2: 'RK23',
        3: 'DOP853',
        4: 'Radau',
        5: 'BDF',
        6: 'LSODA',
    }
    
    method = method_mapping.get(method_code, 'DOP853')  # Default to 'DOP853' if code is invalid

    tau = t[-1]
    rho_initial_vector = rho_to_vector(rho_initial)
    Gamma = gammaMatrix(dim1, dim2, Reg, De1, Re1, we1, De2, Re2, we2, Esp)
    # Initialize tqdm progress bar
  

    solution = solve_ivp(system_equations_opt_vect, [0, tau], rho_initial_vector,
                        args=(dim1, dim2, Reg, De1, Re1, we1, De2, Re2, we2, Gamma),
                        method=method, t_eval=t)
 
    return solution

def system_equations_opt(t, y, dim1, dim2, Reg, De1, Re1, we1, De2, Re2, we2,Gamma):
    """
    Calculate the time derivative of rho
    """
    
    rho = y.reshape((dim1 + dim2, dim1 + dim2))
    drho_dt = np.zeros_like(rho)
    Omega_t = omegaMuNu(t, dim1, dim2, Reg)
 
    populations_s=np.zeros(dim1)
    populations_p=np.zeros(dim2)

    # Update population dynamics for s-states
    Gamma_rho_p_diag = np.einsum('ij,j->ij', Gamma, np.diag(rho[dim1:, dim1:]))
    populations_s=  Gamma_rho_p_diag.sum(axis=1)
    
    # Update population dynamics for p-states
    populations_p= -Gamma_rho_p_diag.sum(axis=0)
   
    # Set the first dim1 diagonal terms for s-states
    drho_dt[np.arange(dim1), np.arange(dim1)] = populations_s
   
    # Set the following dim2 diagonal terms for p-states
    drho_dt[np.arange(dim1, dim1 + dim2), np.arange(dim1, dim1 + dim2)] = populations_p
   
    # Update coherence terms
    Gamma_v=Gamma.sum(axis=0) #I calculate \sum_\mu^\prime \Gamma_\mu^\prime,\nu
    coherences = -0.5 * np.einsum('j,jk->jk', Gamma_v, rho[dim1:, :dim1])
    drho_dt[dim1:, :dim1] += coherences
    drho_dt[:dim1, dim1:] += np.conj(coherences.T)
    
    # Adding Hamiltonian dynamics
    # Coherent evolution term: -i [Omega, rho]
    Omega=np.zeros_like(rho)
    Omega[:dim1, dim1:] = Omega_t
    Omega[dim1:, :dim1] = np.conj(Omega_t.T)
    commutator = np.dot(Omega, rho) - np.dot(rho, Omega)
    drho_dt += -1j * commutator

    return drho_dt.flatten()



def rho_to_vector(rho):
    N = rho.shape[0]
    vect = np.zeros(N**2)
    
    # Diagonal elements
    vect[:N] =np.real( np.diag(rho))
    
    # Off-diagonal elements
    off_diag_indices = np.tril_indices(N, -1)
    real_parts = np.real(rho[off_diag_indices])
    imag_parts = np.imag(rho[off_diag_indices])
    
    vect[N + 2*np.arange(len(real_parts))] = real_parts
    vect[N + 2*np.arange(len(imag_parts)) + 1] = imag_parts
    
    return vect

def vector_to_rho(vect, N):
    rho = np.zeros((N, N), dtype=complex)
    
    # Diagonal elements
    np.fill_diagonal(rho, vect[:N])
    
    # Off-diagonal elements
    off_diag_indices = np.tril_indices(N, -1)
    rho[off_diag_indices] = vect[N + 2*np.arange(len(off_diag_indices[0]))] + 1j * vect[N + 2*np.arange(len(off_diag_indices[0])) + 1]
    rho.T[off_diag_indices] = np.conj(rho[off_diag_indices])
    
    return rho
 


# def solve_dyn(rho_initial,t, dim1,dim2,Deg,De1, Re1, we1, De2, Re2, we2,Esp):
#     tau = t[-1]
#     Gamma=gammaMatrix(dim1,dim2,Reg,De1, Re1, we1, De2, Re2, we2,Esp)
#     solution = solve_ivp(system_equations_opt, [0, tau], rho_initial.flatten(), args=(dim1,dim2,Deg,De1, Re1, we1, De2, Re2, we2,Gamma), method='DOP853', t_eval=t)
#     #after few test the method DOP853 is the onyl one giving correct results
#     return solution


def lambd(De,we):
    hbar=6.654/(2*np.pi)*10**-34
    e=1.6*10**-19
    mass=7.1*10**-26
    lambd=np.sqrt(2 * De*e*mass) * we*10**-10/hbar #2*De/0.0073  # Simplified expression with units adjusted
    return lambd




hbar=6.654/(2*np.pi)*10**-34
e=1.6*10**-19
mass=7.1*10**-26
