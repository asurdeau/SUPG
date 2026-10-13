# Modules généraux
import numpy as np
import yaml
import math
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from numba import njit
import scipy as sp
from matplotlib.backends.backend_pdf import PdfPages

# Modules perso
import config
from config import XVEL, YVEL, PRES
import grid_mod
import spatial_operators
import time_operators


# CONSTANT IN SPACE DATA

# LINEAR SYSTEM OF DECOUPLED EQUATIONS
def getTest1Coefficients(basisChoice, spacePolynOrder, lamb):
    evolCoeffs = np.zeros((20, 3, spacePolynOrder, spacePolynOrder)) 
    massCoeffs = np.zeros((20, 3, spacePolynOrder, spacePolynOrder))

    Id = np.array([np.zeros((spacePolynOrder, spacePolynOrder)), np.eye(spacePolynOrder), np.zeros((spacePolynOrder, spacePolynOrder))])
    
    listCoeffs = [0, 8, 16]
    for i in listCoeffs :
        evolCoeffs[i] = lamb * Id
        evolCoeffs[i+1] = Id
        massCoeffs[i] = Id
        massCoeffs[i+1] = Id

    return evolCoeffs, massCoeffs


def getTest1Solution(q, lamb, time, X, Y):
    if np.size(time) > 1:
        time = time.reshape(time.shape + (1,) * q.ndim)
    return q.copy() * np.exp(- lamb * time)




# LINEAR SYSTEM OF COUPLED EQUATIONS : ROTATION AROUND omega
def getTest2Coefficients(basisChoice, spacePolynOrder, omega):
    a, b, c = omega
    evolCoeffs = np.zeros((20, 3, spacePolynOrder, spacePolynOrder)) 
    massCoeffs = np.zeros((20, 3, spacePolynOrder, spacePolynOrder))

    Id = np.array([np.zeros((spacePolynOrder, spacePolynOrder)), np.eye(spacePolynOrder), np.zeros((spacePolynOrder, spacePolynOrder))])

    # mass : identity only
    listCoeffs = [0, 1, 8, 9, 16, 17]
    for i in listCoeffs :
        massCoeffs[i] = Id
    
    # evolultion matrice : the one that corresponds to the vector product with omega [[0, -c, b], [c, 0, -a], [-b, a, 0]]
    evolCoeffs[2] = - c * Id
    evolCoeffs[3] = Id
    evolCoeffs[4] = b * Id
    evolCoeffs[5] = Id

    evolCoeffs[6] = c * Id
    evolCoeffs[7] = Id
    evolCoeffs[10] = -a * Id
    evolCoeffs[11] = Id

    evolCoeffs[12] = - b * Id
    evolCoeffs[13] = Id
    evolCoeffs[14] = a * Id
    evolCoeffs[15] = Id

    
    return evolCoeffs, massCoeffs


def getTest2Solution(q, omega, time, X, Y):
    time = np.asarray(time)
    norm_omega = np.linalg.norm(omega)

    theta = norm_omega * time
    # on ajoute q.ndim axes de taille 1 à la fin de theta
    # pour qu'il se broadcaste contre les dimensions (i,k,j,l,var) de q
    theta = theta.reshape(theta.shape + (1,) * q.ndim)

    q_dot_omega = np.dot(q, omega)
    q_parallel = (q_dot_omega / norm_omega**2)[..., np.newaxis] * omega
    q_perp = q - q_parallel
    omega_cross_q = np.cross(omega, q)

    return q_parallel + q_perp * np.cos(theta) - omega_cross_q * np.sin(theta) / norm_omega




# NON CONSTANT IN SPACE DATA

# LINEAR SYSTEM OF DECOUPLED EQUATIONS BUT WITH MASS MATRICES
def getTest3Coefficients(basisChoice, spacePolynOrder, dx, dy, alpha, h, lamb):
    evolCoeffs = np.zeros((20, 3, spacePolynOrder, spacePolynOrder)) 
    massCoeffs = np.zeros((20, 3, spacePolynOrder, spacePolynOrder))

    Id = np.array([np.zeros((spacePolynOrder, spacePolynOrder)), np.eye(spacePolynOrder), np.zeros((spacePolynOrder, spacePolynOrder))])
    mass = spatial_operators.get_arbitrar_order_SUPG_weighted_mass_operator_coeffs(basisChoice, spacePolynOrder, dx, dy, alpha, h)[0]

    listCoeffs = [0, 8, 16]
    for i in listCoeffs :
        evolCoeffs[i] = lamb * mass
        evolCoeffs[i+1] = mass
        massCoeffs[i] = mass
        massCoeffs[i+1] = mass

    return evolCoeffs, massCoeffs


def getTest3Solution(q, lamb, time, X, Y):
    if np.size(time) > 1:
        time = time.reshape(time.shape + (1,) * q.ndim)

    return q.copy() * np.sin(2. * np.pi * X) * np.cos(2. * np.pi * Y) * np.exp(- lamb * time)




# LINEAR SYSTEM OF COUPLED EQUATIONS : ROTATION AROUND omega
def getTest4Coefficients(basisChoice, spacePolynOrder, dx, dy, alpha, h, omega):
    a, b, c = omega
    evolCoeffs = np.zeros((20, 3, spacePolynOrder, spacePolynOrder)) 
    massCoeffs = np.zeros((20, 3, spacePolynOrder, spacePolynOrder))

    
    Id = np.array([np.zeros((spacePolynOrder, spacePolynOrder)), np.eye(spacePolynOrder), np.zeros((spacePolynOrder, spacePolynOrder))])
    mass = spatial_operators.get_arbitrar_order_SUPG_weighted_mass_operator_coeffs(basisChoice, spacePolynOrder, dx, dy, alpha, h)[0]

    # mass : identity only
    listCoeffs = [0, 1, 8, 9, 16, 17]
    for i in listCoeffs :
        massCoeffs[i] = mass
    
    # evolultion matrice : the one that corresponds to the vector product with omega [[0, -c, b], [c, 0, -a], [-b, a, 0]]
    evolCoeffs[2] = - c * mass
    evolCoeffs[3] = mass
    evolCoeffs[4] = b * mass
    evolCoeffs[5] = mass

    evolCoeffs[6] = c * mass
    evolCoeffs[7] = mass
    evolCoeffs[10] = -a * mass
    evolCoeffs[11] = mass

    evolCoeffs[12] = - b * mass
    evolCoeffs[13] = mass
    evolCoeffs[14] = a * mass
    evolCoeffs[15] = mass

    
    return evolCoeffs, massCoeffs


def getTest4Solution(q, omega, time, X, Y):
    time = np.asarray(time)
    norm_omega = np.linalg.norm(omega)

    theta = norm_omega * time
    # on ajoute q.ndim axes de taille 1 à la fin de theta
    # pour qu'il se broadcaste contre les dimensions (i,k,j,l,var) de q
    theta = theta.reshape(theta.shape + (1,) * q.ndim)

    q_dot_omega = np.dot(q, omega)
    q_parallel = (q_dot_omega / norm_omega**2)[..., np.newaxis] * omega
    q_perp = q - q_parallel
    omega_cross_q = np.cross(omega, q)

    return np.sin(2. * np.pi * X) * np.cos(2. * np.pi * Y) * (q_parallel + q_perp * np.cos(theta) - omega_cross_q * np.sin(theta) / norm_omega)





# DECOUPLED LINEAR ADVECTIONS

# 1D x-advection
def getTest5Coefficients(basisChoice, spacePolynOrder, dx, dy, alpha, h, c):
    evolCoeffs = np.zeros((20, 3, spacePolynOrder, spacePolynOrder)) 
    massCoeffs = np.zeros((20, 3, spacePolynOrder, spacePolynOrder))

    Id = np.array([np.zeros((spacePolynOrder, spacePolynOrder)), np.eye(spacePolynOrder), np.zeros((spacePolynOrder, spacePolynOrder))])
    coeffs = spatial_operators.get_arbitrar_order_SUPG_weighted_evolution_operator_coeffs(basisChoice, spacePolynOrder, dx, dy, alpha, h)

    mass = coeffs[1].copy()
    right_der = coeffs[4].copy()
    left_der_stab = coeffs[2].copy()
    second_der_stab = coeffs[0].copy()

    listCoeffs = [0, 8, 16]
    for i in listCoeffs :
        evolCoeffs[i] = c * (right_der + np.sign(c) * second_der_stab)
        evolCoeffs[i+1] = Id
        massCoeffs[i] = mass + np.sign(c) * left_der_stab
        massCoeffs[i+1] = Id

    return evolCoeffs, massCoeffs


def getTest5Solution(q, c, time, X, Y):
    if np.size(time) > 1:
        time = time.reshape(time.shape + (1,) * q.ndim)

    return q.copy() * np.sin(2. * np.pi * (X - c * time)) * np.cos(2. * np.pi * Y)



# 2D ADVECTION : WE NEED TO DEFINE MASS AND EVOLUTION OPERATORS EVALUATION DIFFERENTLY, SAME AS FOR DeC UPDATE,
# BUT WE TRY TO USE THE SAME CODE AS MUCH AS POSSIBLE SO q WILL HAVE TO BE SCALAR !!
# BECAUSE FOR ADVECTION, THE EVOLUTION OPERATOR IS THE SUM OF THE TERMS OF THE FIRST TWO ROWS OF THE ACOUSTIC EVOLUTION OPERATOR
# AND THE MASS OPERATOR IS THE SUM OF THE TERMS OF THE LAST ROW OF THE ACOUSTIC MASS OPERATOR

@njit
def get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator_ADVECTION(q, iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs) :
    q_translated = np.concatenate([q, q, q], axis=-1) 

    Advection_evolved_q = np.full_like(q, 0.)

    Acoustic_evolved_q_translated = spatial_operators.get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q_translated, iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs)

    for i in range(iMin, iMax+1):
      for j in range(jMin, jMax+1):
          for k in range(spacePolynOrder):
            for l in range(spacePolynOrder):
                Advection_evolved_q[i, j, k, l] = Acoustic_evolved_q_translated[i, j, k, l, XVEL] + Acoustic_evolved_q_translated[i, j, k, l, YVEL]

    return Advection_evolved_q


@njit
def get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator_ADVECTION(q, iMin, iMax, jMin, jMax, spacePolynOrder, massCoeffs) :
    q_translated = np.concatenate([q, q, q], axis=-1) 

    Acoustic_massed_q_translated = spatial_operators.get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator(q_translated, iMin, iMax, jMin, jMax, spacePolynOrder, massCoeffs)

    Advection_massed_q = Acoustic_massed_q_translated[:, :, :, :, PRES]

    return Advection_massed_q



@njit
def getDeCUpdate_ADVECTION(q, dt, spacePolynOrder, nbSubTimeSteps, nbCorrections, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs):
    # WATCH OUT : WE HAVE q^{(p), 0} = q AND q^{(0), m} = q

    P = nbCorrections
    M = nbSubTimeSteps
    # INITIALISATION DES VARIABLES : TOUS LES q 
    shape = np.shape(q)
    q_current_stage = np.zeros(shape + (M+1,))
    first_stage_time_weights_sums = np.zeros(M+1)
    weighted_evolved_q_m_current_stage = np.zeros(shape + (M+1,))
    weighted_sum_of_weighted_evolved_q_m_current_stage = np.zeros(shape)
    weighted_massed_current_stage_differences = np.zeros(shape + (M+1,))
    last_weighted_massed_current_stage_differences = np.zeros(shape)
    q_next_stage = np.zeros(shape + (M+1,))

    # TIME WEIGHTS theta^m_r
    theta = time_operators.getGLoBasisFunctionsSubIntegrals(M+1)

    # INITIAL STAGE : q^{(0), m} = q_n : so we do nothing
    # PLUS : We have q^{(P), 0} = q_n
    q_current_stage[:, :, :, :, :, 0] = q    # We have q^{(P), 0} = q_n
    weighted_evolved_q_m_current_stage[:, :, :, :, :, 0] = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator_ADVECTION(q_current_stage[:, :, :, :, :, 0], iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs)

    if nbSubTimeSteps == 1 :
      q_next_stage[:, :, :, :, :, M] = q - dt * weighted_evolved_q_m_current_stage[:, :, :, :, :, 0]
      q_next_stage[:, :, :, :, :, M] = time_operators.periodize(q_next_stage[:, :, :, :, :, M], iMin, iMax, jMin, jMax, nGhost)
      return q_next_stage[:, :, :, :, :, M]
    
    # FIRST STAGE : EASY TO COMPUTE : STILL NO NEED OF THE MASS
    for m in range(1, M+1):
      first_stage_time_weights_sums[m] =  np.sum(theta[:, m])

    for i in range(iMin, iMax+1):
      for j in range(jMin, jMax+1):
        for k in range(spacePolynOrder):
          for l in range(spacePolynOrder):
            for var in [XVEL, YVEL, PRES]:
              for m in range(1, M+1):
                q_next_stage[i, j, k, l, var, m] = q[i, j, k, l, var] - dt * first_stage_time_weights_sums[m] * weighted_evolved_q_m_current_stage[i, j, k, l, var, 0]

    q_next_stage[:, :, :, :, :, 1:] = time_operators.periodize(q_next_stage[:, :, :, :, :, 1:], iMin, iMax, jMin, jMax, nGhost)
    q_current_stage[:, :, :, :, :, 1:] = q_next_stage[:, :, :, :, :, 1:].copy()


    # INTERMEDIATE STAGES
    for p in range(2, P):
      for m in range(1, M+1):
        weighted_evolved_q_m_current_stage[:, :, :, :, :, m] = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator_ADVECTION(q_current_stage[:, :, :, :, :, m], iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs)
        weighted_massed_current_stage_differences[:, :, :, :, :, m] = get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator_ADVECTION(q_current_stage[:, :, :, :, :, m] - q, iMin, iMax, jMin, jMax, spacePolynOrder, massCoeffs)

      for i in range(iMin, iMax+1):
        for j in range(jMin, jMax+1):
          for k in range(spacePolynOrder):
            for l in range(spacePolynOrder):
              for var in [XVEL, YVEL, PRES]:
                for m in range(1, M+1):
                  weighted_sum_of_weighted_evolved_q_m_current_stage[i, j, k, l, var] = theta[0, m] * weighted_evolved_q_m_current_stage[i, j, k, l, var, 0]
                  for r in range(1, M+1):
                    weighted_sum_of_weighted_evolved_q_m_current_stage[i, j, k, l, var] += theta[r, m] * weighted_evolved_q_m_current_stage[i, j, k, l, var, r]

                  q_next_stage[i, j, k, l, var, m] = q_current_stage[i, j, k, l, var, m] \
                                                      - weighted_massed_current_stage_differences[i, j, k, l, var, m] \
                                                      - dt * weighted_sum_of_weighted_evolved_q_m_current_stage[i, j, k, l, var]
      
      q_next_stage[:, :, :, :, :, 1:] = time_operators.periodize(q_next_stage[:, :, :, :, :, 1:], iMin, iMax, jMin, jMax, nGhost)
      q_current_stage[:, :, :, :, :, 1:] = q_next_stage[:, :, :, :, :, 1:].copy()
  


    # LAST STAGE P : ONLY COMPUTE q^{(P), M}
    for m in range(1, M+1):
      weighted_evolved_q_m_current_stage[:, :, :, :, :, m] = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator_ADVECTION(q_current_stage[:, :, :, :, :, m], iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs)

    last_weighted_massed_current_stage_differences = get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator_ADVECTION(q_current_stage[:, :, :, :, :, M] - q, iMin, iMax, jMin, jMax, spacePolynOrder, massCoeffs)

    for i in range(iMin, iMax+1):
      for j in range(jMin, jMax+1):
        for k in range(spacePolynOrder):
          for l in range(spacePolynOrder):
            for var in [XVEL, YVEL, PRES]:
              weighted_sum_of_weighted_evolved_q_m_current_stage[i, j, k, l, var] = theta[0, M] * weighted_evolved_q_m_current_stage[i, j, k, l, var, 0]
              for r in range(1, M+1):
                weighted_sum_of_weighted_evolved_q_m_current_stage[i, j, k, l, var] += theta[r, M] * weighted_evolved_q_m_current_stage[i, j, k, l, var, r]

              q_next_stage[i, j, k, l, var, M] = q_current_stage[i, j, k, l, var, M] \
                                            - last_weighted_massed_current_stage_differences[i, j, k, l, var] \
                                            - dt * weighted_sum_of_weighted_evolved_q_m_current_stage[i, j, k, l, var]


    q_next_stage[:, :, :, :, :, M] = time_operators.periodize(q_next_stage[:, :, :, :, :, M], iMin, iMax, jMin, jMax, nGhost)

    return q_next_stage[:, :, :, :, :, M]





# Here, we only multiply the derivatives by the corresponding components of the velocity
def getTest6Coefficients(basisChoice, spacePolynOrder, dx, dy, alpha, h, c):
    evolCoeffs = np.zeros((20, 3, spacePolynOrder, spacePolynOrder)) 
    massCoeffs = np.zeros((20, 3, spacePolynOrder, spacePolynOrder))
    
    evolCoeffs = spatial_operators.get_arbitrar_order_SUPG_weighted_evolution_operator_coeffs(basisChoice, spacePolynOrder, dx, dy, alpha, h)
    massCoeffs = spatial_operators.get_arbitrar_order_SUPG_weighted_evolution_operator_coeffs(basisChoice, spacePolynOrder, dx, dy, alpha, h)


    cx, cy = c[0], c[1]

    listCoeffs_xder_evolCoeffs = [0, 2, 4, 6, 12, 16]   # list of the x_der operators in evolCoeffs
    listCoeffs_yder_evolCoeffs = [3, 7, 9, 11, 15, 19]  # list of the y_der operators in evolCoeffs
    listCoeffs_xder_massCoeffs = [4, 12]                # list of the x_der operators in massCoeffs
    listCoeffs_yder_massCoeffs = [11, 15]               # list of the y_der operators in massCoeffs

    for i in listCoeffs_xder_evolCoeffs :
        evolCoeffs[i] = cx * evolCoeffs[i]

    for i in listCoeffs_yder_evolCoeffs :
        evolCoeffs[i] = cy * evolCoeffs[i]

    for i in listCoeffs_xder_massCoeffs :
        massCoeffs[i] = cx * massCoeffs[i]

    for i in listCoeffs_yder_massCoeffs :
        massCoeffs[i] = cy * massCoeffs[i]

    
    return evolCoeffs, massCoeffs


def getTest6Solution(q, c, time, X, Y):
    if np.size(time) > 1:
        time = time.reshape(time.shape + (1,) * q.ndim)

    return q.copy() * np.sin(2. * np.pi * (X - c[0] * time)) * np.cos(2. * np.pi * (Y - c[1] * time))




def get_flattened_quantity(q, xCellNumber, yCellNumber, spacePolynOrder):

    q_flattened = np.zeros((xCellNumber * spacePolynOrder, yCellNumber * spacePolynOrder, 3))

    for k in range(spacePolynOrder):
        for l in range(spacePolynOrder):
            q_flattened[k::spacePolynOrder, l::spacePolynOrder] = q[:, :, k, l]

    return q_flattened



####################################################################################################################

#                                                  CONVERGENCE TEST                                                #

####################################################################################################################








# TIME EVOLUTION FOR ONE VALUE OF dt
# if (True) :
if (False) :
    # PARAMETERS
    ### mesh
    xL, xR, yL, yR = 0., 1., 0., 1.
    Nx, Ny = 20, 20
    dx, dy = (xR - xL) / Nx, (yR - yL) / Ny
    h = max(dx, dy)
    nGhost = 2
    iMin, iMax, jMin, jMax = nGhost, Nx + nGhost - 1, nGhost, Ny + nGhost - 1

    ### time 
    finalTime = 1.
    dt_apriori = 0.01

    ### stabilisation parameter
    alpha = 0.5

    ### basis and order
    spacePolynOrder = 1
    nbSubTimeSteps = 1
    nbCorrections = 1
    basisChoice = "basic"

    ### test choice
    testChoice = 5

    ### plots options
    nbSnapshots = 11
    timeBetweenSnapshots = finalTime / (nbSnapshots-1.)
    ##### 2D mesh for plots
    xCoord_2D = np.linspace(xL - dx * nGhost, xR + dx * nGhost, (Nx + 2*nGhost)*spacePolynOrder, endpoint=False)
    yCoord_2D = np.linspace(yL - dy * nGhost, yR + dy * nGhost, (Ny + 2*nGhost)*spacePolynOrder, endpoint=False)
    X_2D, Y_2D = np.meshgrid(xCoord_2D, yCoord_2D, indexing='ij')

    ### initial data
    norm_q0_min, norm_q0_max = 1., 3.
    lamb_min, lamb_max = 0.5, 1.
    c_min, c_max = 0.5, 1.

    

    # INITIAL DATA DRAW
    norm_q0 = (norm_q0_max - norm_q0_min) * np.random.rand() + norm_q0_min
    drawn_variable_q0 = np.random.randn(3)
    u0, v0, p0 = norm_q0 * abs(drawn_variable_q0) / np.linalg.norm(drawn_variable_q0, ord=2)
    q_0 = np.full((Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder, spacePolynOrder, 3), [u0, v0, p0])
    print("Valeurs de départ : ", q_0[0, 0, 0, 0, XVEL], q_0[0, 0, 0, 0, YVEL], q_0[0, 0, 0, 0, PRES])

    ### Test choice 
    if testChoice == 3:
        # TIRAGE UNIFORME SUR [-LAMB_MAX, -LAMB_MIN] U [LAMB_MIN, LAMB_MAX]
        lamb = (2. * np.random.binomial(1, 0.5) - 1.) * ((lamb_max - lamb_min) * np.random.rand() + lamb_min)
        
        params = lamb
        getTestCoefficients = getTest3Coefficients
        getTestSolution = getTest3Solution


    elif testChoice == 4:
        # Tirage d'un vecteur de norme 2pi uniforme sur la sphère
        # tirage d'un vecteur gaussien
        drawn_variable = np.random.randn(3)
        # renormalisation
        omega = 2. * np.pi * drawn_variable / np.linalg.norm(drawn_variable, ord=2)

        params = omega
        getTestCoefficients = getTest4Coefficients
        getTestSolution = getTest4Solution


    elif testChoice == 5:
        c = (2. * np.random.binomial(1, 0.5) - 1.) * ((c_max - c_min) * np.random.rand() + c_min)

        params = c
        getTestCoefficients = getTest5Coefficients
        getTestSolution = getTest5Solution

    ### "4D" mesh for initial data
    xCoord = np.linspace(xL - dx * nGhost, xR + dx * nGhost, Nx + 2*nGhost, endpoint=False)
    yCoord = np.linspace(yL - dy * nGhost, yR + dy * nGhost, Ny + 2*nGhost, endpoint=False)
    X_FO, Y_FO = np.meshgrid(xCoord, yCoord, indexing='ij')
    X, Y = np.zeros((2, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder, spacePolynOrder, 3)) 
    for k in range(spacePolynOrder):
        for l in range(spacePolynOrder):
            for var in [XVEL, YVEL, PRES]:
                X[:, :, k, l, var] = X_FO + dx * (1. * k) / spacePolynOrder
                Y[:, :, k, l, var] = Y_FO + dy * (1. * l) / spacePolynOrder

    q_approx = getTestSolution(q_0, params, 0., X, Y)
    q_ex = getTestSolution(q_0, params, 0., X, Y)
    
    # Operators coefficients depending on the choice
    evolCoeffs, massCoeffs = getTestCoefficients(basisChoice, spacePolynOrder, params)



    # OBSERVABLES INITIALISATION
    ### Array building
    q_approx_table = np.zeros(((Nx + 2*nGhost) * spacePolynOrder, (Ny + 2*nGhost) * spacePolynOrder, 3, nbSnapshots), order="F")
    q_ex_table = np.zeros(((Nx + 2*nGhost) * spacePolynOrder, (Ny + 2*nGhost) * spacePolynOrder, 3, nbSnapshots), order="F")
    q_ecart_table = - np.ones(((Nx + 2*nGhost) * spacePolynOrder, (Ny + 2*nGhost) * spacePolynOrder, 3, nbSnapshots), order="F")
    timesTable = np.zeros(nbSnapshots)
    ### First snapshot
    k = 0
    q_ecart = abs(q_approx - q_ex)
    q_approx_table[:, :, :, k] = get_flattened_quantity(q_approx, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder)
    q_ex_table[:, :, :, k] = get_flattened_quantity(q_ex, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder)
    q_ecart_table[:, :, :, k] = get_flattened_quantity(q_ecart, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder)
    timesTable[k] = 0.
    nbSnapshotsDone += 1
    k = 1
    

    print("Début des calculs : ")
    currentTime = 0.
    doSnapshotNow = False
    nbSnapshotsDone = 0
    while ( finalTime > currentTime + 1.e-12  ):

        # dt adaptation
        if (currentTime + dt_apriori > finalTime) :
            dt = finalTime - currentTime
            currentTime = finalTime
            print("dernière itération")
            doSnapshotNow = False 

        elif (currentTime + dt_apriori > nbSnapshotsDone * timeBetweenSnapshots) :
            dt = nbSnapshotsDone * timeBetweenSnapshots - currentTime
            currentTime = nbSnapshotsDone * timeBetweenSnapshots

            doSnapshotNow = True
        else :
            dt = dt_apriori
            currentTime += dt_apriori


        # update
        q_approx = time_operators.getDeCUpdate(q_approx, dt, spacePolynOrder, nbSubTimeSteps, nbCorrections, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs)
        

        # snapshot if asked 
        if doSnapshotNow :
            print(currentTime)
            timesTable[k] = currentTime
            q_ex = getTestSolution(q_0, params, currentTime, X, Y)
            q_ecart = abs(q_approx - q_ex)
            q_approx_table[:, :, :, k] = get_flattened_quantity(q_approx, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder)
            q_ex_table[:, :, :, k] = get_flattened_quantity(q_ex, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder)
            q_ecart_table[:, :, :, k] = get_flattened_quantity(q_ecart, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder)
            doSnapshotNow = False
            nbSnapshotsDone += 1
            k += 1

    # Final snapshots
    timesTable[k] = currentTime
    q_ex = getTestSolution(q_0, params, currentTime, X, Y)
    q_ecart = abs(q_approx - q_ex)
    q_approx_table[:, :, :, k] = get_flattened_quantity(q_approx, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder)
    q_ex_table[:, :, :, k] = get_flattened_quantity(q_ex, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder)
    q_ecart_table[:, :, :, k] = get_flattened_quantity(q_ecart, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder)
    k += 1

    # Plots
    def getPlotTitle(q0, params, testChoice, dt, time):
        if testChoice == 1 :
            plotTitle = f"Solutions de $\\frac{{d}}{{dt}}q + \\lambda q = 0$ à t = {time}" \
                    f"\n avec $q_0 \\simeq$ [{round(q0[0], 3)}, {round(q0[1], 3)}, {round(q0[2], 3)}], $\\lambda$ = {round(params, 3)} et dt = {round(dt, 4)}"
        elif testChoice == 2 :
            plotTitle = f"Solutions de $\\frac{{d}}{{dt}}q + \\omega \\times q = 0$ à t = {time}" \
                    f"\n avec $q_0 \\simeq$ [{round(q0[0], 3)}, {round(q0[1], 3)}, {round(q0[2], 3)}], $\\omega$ = [{round(params[0], 3)}, {round(params[1], 3)}, {round(params[2], 3)}] et dt = {round(dt, 4)}"
        elif testChoice == 3 :
            plotTitle = f"Solutions de $(M_x \\otimes M_y)\\frac{{d}}{{dt}}q + \\lambda (M_x \\otimes M_y) q = 0$ à t = {time}" \
                    f"\n avec $q_0 \\simeq$ [{round(q0[0], 3)}, {round(q0[1], 3)}, {round(q0[2], 3)}], $\\lambda$ = {round(params, 3)} et dt = {round(dt, 4)}"
        elif testChoice == 4 :
                    plotTitle = f"Solutions de $(M_x \\otimes M_y)\\frac{{d}}{{dt}}q + (M_x \\otimes M_y) \\omega \\times q = 0$ à t = {time}" \
                            f"\n avec $q_0 \\simeq$ [{round(q0[0], 3)}, {round(q0[1], 3)}, {round(q0[2], 3)}], $\\omega$ = [{round(params[0], 3)}, {round(params[1], 3)}, {round(params[2], 3)}] et dt = {round(dt, 4)}"
        elif testChoice == 5 :
                    plotTitle = f"Solutions de $(M_x \\otimes M_y)\\frac{{d}}{{dt}}q + c ((D_x + \\alpha h D^x_x) \\otimes M_y) q = 0$ à t = {time}" \
                            f"\n avec $q_0 \\simeq$ [{round(q0[0], 3)}, {round(q0[1], 3)}, {round(q0[2], 3)}], $c$ = {round(params, 3)} et dt = {round(dt, 4)}"

        return plotTitle
    
    for var, varName in [(XVEL, "U"), (YVEL, "V"), (PRES, "P")] :
        with PdfPages(f"Plots/Combined/Test_Evol_DeC_{spacePolynOrder}_dt_{round(dt, 4)}_h_{round(1./Nx, 4)}_{varName}") as pdf:
            for k in range(nbSnapshots):
                plotTitle = getPlotTitle([u0, v0, p0], params, testChoice, dt, timesTable[k])

                fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
                fig.suptitle(plotTitle, fontsize=16)
                ax = axes[0]
                ax.set_title(f"approchée")
                pcm1 = ax.pcolormesh(X_2D, Y_2D, q_approx_table[:, :, var, k])
                plt.colorbar(pcm1, ax=ax)

                ax = axes[1]
                ax.set_title(f"exacte")
                pcm2 = ax.pcolormesh(X_2D, Y_2D, q_ex_table[:, :, var, k])
                plt.colorbar(pcm2, ax=ax)

                ax = axes[2]
                ax.set_title(f"ecart")
                pcm3 = ax.pcolormesh(X_2D, Y_2D, q_ecart_table[:, :, var, k])
                plt.colorbar(pcm3, ax=ax)
                pdf.savefig()
                plt.close()




####################################################################################################################

#                                                  CONVERGENCE TEST                                                #

####################################################################################################################





# CONVERGENCE TEST : END TIME ERRORS WITH VARRYING Nx AND Ny
if (True) :
# if (False) :
    # PARAMETERS
    ### meshes 
    xL, xR, yL, yR = 0., 1., 0., 1.
    nList = np.array(10**(np.linspace(1, 2, 5)), dtype='int')
    NxList = nList
    NyList = nList
    dxList = (xR - xL) / NxList
    dyList = (yR - yL) / NyList
    hList = np.zeros(len(nList))
    nGhost = 2

    ### time
    finalTime = 1.

    ### stabilisation parameters
    alpha = 0.5
    CFL = 0.2

    ### basis and order
    spacePolynOrder = 3
    nbSubTimeSteps = 4
    nbCorrections = 4
    basisChoice = "basic"

    ### test choice
    testChoice = 5

    ### initial data
    norm_q0_min, norm_q0_max = 1., 3.
    lamb_min, lamb_max = 0.5, 1.
    c_min, c_max = 0.5, 1.

    ### plot options
    plotsLocation = "Plots/Combined/"    


    # INITIAL DATA DRAW
    norm_q0 = (norm_q0_max - norm_q0_min) * np.random.rand() + norm_q0_min
    drawn_variable_q0 = np.random.randn(3)
    u0, v0, p0 = norm_q0 * abs(drawn_variable_q0) / np.linalg.norm(drawn_variable_q0, ord=2)
    print("Valeurs de départ : ", u0, v0, p0)


    ### Test choice
    if testChoice == 1:
        # TIRAGE UNIFORME SUR [-LAMB_MAX, -LAMB_MIN] U [LAMB_MIN, LAMB_MAX]
        params = (2. * np.random.binomial(1, 0.5) - 1.) * ((lamb_max - lamb_min) * np.random.rand() + lamb_min)
        getTestCoefficients = getTest1Coefficients
        getTestSolution = getTest1Solution

    elif testChoice == 2:
        # Tirage d'un vecteur de norme 2pi uniforme sur la sphère
        # tirage d'un vecteur gaussien
        drawn_variable = np.random.randn(3)
        # renormalisation
        params = 2. * np.pi * drawn_variable / np.linalg.norm(drawn_variable, ord=2)
        getTestCoefficients = getTest2Coefficients
        getTestSolution = getTest2Solution

    elif testChoice == 3:
        # TIRAGE UNIFORME SUR [-LAMB_MAX, -LAMB_MIN] U [LAMB_MIN, LAMB_MAX]
        params = (2. * np.random.binomial(1, 0.5) - 1.) * ((lamb_max - lamb_min) * np.random.rand() + lamb_min)
        getTestCoefficients = getTest3Coefficients
        getTestSolution = getTest3Solution

    elif testChoice == 4:
        # Tirage d'un vecteur de norme 2pi uniforme sur la sphère
        # tirage d'un vecteur gaussien
        drawn_variable = np.random.randn(3)
        # renormalisation
        params = 2. * np.pi * drawn_variable / np.linalg.norm(drawn_variable, ord=2)
        getTestCoefficients = getTest4Coefficients
        getTestSolution = getTest4Solution

    elif testChoice == 5:
        params = (2. * np.random.binomial(1, 0.5) - 1.) * ((c_max - c_min) * np.random.rand() + c_min)
        getTestCoefficients = getTest5Coefficients
        getTestSolution = getTest5Solution

    elif testChoice == 6:
        params = (2. * np.random.binomial(1, 0.5, 2) - 1.) * ((c_max - c_min) * np.random.rand(2) + c_min)
        getTestCoefficients = getTest6Coefficients
        getTestSolution = getTest6Solution


    # OBSERVABLES INITIALISATION
    ### Array creation
    u_finalTime_sup_errors, v_finalTime_sup_errors, p_finalTime_sup_errors = - np.ones((3, len(nList)))
    u_finalTime_L2_errors, v_finalTime_L2_errors, p_finalTime_L2_errors = - np.ones((3, len(nList)))

    simuIndex = 0

    for i in range(len(nList)) :

        Nx = NxList[i]
        Ny = NyList[i]
        dx = dxList[i]
        dy = dyList[i]
        h = max(dx, dy)
        hList[i] = h

        # "First guess for dt"
        dt = min(dx, dy)

        # Adaptations of stabilisation parameters and dt
        if (testChoice == 5) :

            c = params

            print("adaptation de alpha et CFL selon l'ordre : ")
            if spacePolynOrder == 1 :
                alpha = 0.588
                CFL = 1.701
            if spacePolynOrder == 2 :
                alpha = 0.229
                CFL = 0.492
            if spacePolynOrder == 3 :
                alpha = 0.089
                CFL = 0.492
            
            print("Adpatation de dx selon la condition CFL dt < dx / (2 |c|)")
            dt = CFL * min(dx, dy) / (2. * abs(c))
            print("Vérification de la condition CFL : ", dt < dx / (2 * abs(c)))

        if (testChoice == 6) :
            print("Adpatation de dx selon la condition CFL dt < dx / (2 |c|)")
            dt = CFL * dx * dy / (abs(c[0]) * dy + abs(c[1]) * dx)

        print("dt : ", dt)


        # "4d" mesh creation for solutions
        evolCoeffs, massCoeffs = getTestCoefficients(basisChoice, spacePolynOrder, dx, dy, alpha, h, params)
        iMin, iMax, jMin, jMax = nGhost, Nx + nGhost - 1, nGhost, Ny + nGhost - 1
        xCoord = np.linspace(xL - dx * nGhost, xR + dx * nGhost, Nx + 2*nGhost, endpoint=False)
        yCoord = np.linspace(yL - dy * nGhost, yR + dy * nGhost, Ny + 2*nGhost, endpoint=False)
        X_FO, Y_FO = np.meshgrid(xCoord, yCoord, indexing='ij')
        X, Y = np.zeros((2, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder, spacePolynOrder, 3)) 
        for k in range(spacePolynOrder):
            for l in range(spacePolynOrder):
                for var in [XVEL, YVEL, PRES]:
                    X[:, :, k, l, var] = X_FO + dx * (1. * k) / spacePolynOrder
                    Y[:, :, k, l, var] = Y_FO + dy * (1. * l) / spacePolynOrder


        # INITIAL DATA
        q_0 = np.full((Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder, spacePolynOrder, 3), [u0, v0, p0])
        q_approx = getTestSolution(q_0, params, 0., X, Y)
        q_exact = getTestSolution(q_0, params, finalTime, X, Y)

        # EXTRACTION OF THE ONLY SCALAR VALUE IF SCALAR TEST :
        if testChoice == 6 :
            q_approx = getTestSolution(q_0, params, 0., X, Y)[:, :, :, :, 0]
            q_exact = getTestSolution(q_0, params, finalTime, X, Y)[:, :, :, :, 0]



        # TIME LOOP
        print("Début des calculs pour n = ", nList[i], " : ", flush=True)
        currentTime = 0.
        taux_progression = 0.2
        while ( abs(finalTime - currentTime) > 1.e-14  ):

            # dt adaptation
            if currentTime + dt > finalTime :
                dt = finalTime - currentTime
                currentTime = finalTime
                print(" -  Temps final dépassé, on raccourcit le pas de temps \n \n", flush=True)
            else : 
                currentTime += dt
                if ( abs(currentTime + dt - finalTime * taux_progression) < 0.5 * dt ):
                    print(" -  Progression du calcul : " + str( 100 * round(taux_progression, 3) )+"%", flush=True)
                    taux_progression += 0.2


            # update
            if testChoice == 6 :
                q_approx = getDeCUpdate_ADVECTION(q_approx, dt, spacePolynOrder, nbSubTimeSteps, nbCorrections, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs)
            else :
                q_approx = time_operators.getDeCUpdate(q_approx, dt, spacePolynOrder, nbSubTimeSteps, nbCorrections, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs)


        # Computation of observation
        if testChoice == 6 :
            u_finalTime_sup_errors[simuIndex] = np.max(abs(q_approx - q_exact))
            u_finalTime_L2_errors[simuIndex] = np.sqrt( dx / spacePolynOrder * dy / spacePolynOrder * np.sum((q_approx[:, :, :, :] - q_exact[:, :, :, :])**2) )
        else : 
            u_finalTime_sup_errors[simuIndex] = np.max(abs(q_approx[:, :, :, :, XVEL] - q_exact[:, :, :, :, XVEL]))
            v_finalTime_sup_errors[simuIndex] = np.max(abs(q_approx[:, :, :, :, YVEL] - q_exact[:, :, :, :, YVEL]))
            p_finalTime_sup_errors[simuIndex] = np.max(abs(q_approx[:, :, :, :, PRES] - q_exact[:, :, :, :, PRES]))

            u_finalTime_L2_errors[simuIndex] = np.sqrt( dx / spacePolynOrder * dy / spacePolynOrder * np.sum((q_approx[:, :, :, :, XVEL] - q_exact[:, :, :, :, XVEL])**2) )
            v_finalTime_L2_errors[simuIndex] = np.sqrt( dx / spacePolynOrder * dy / spacePolynOrder * np.sum((q_approx[:, :, :, :, YVEL] - q_exact[:, :, :, :, YVEL])**2) )
            p_finalTime_L2_errors[simuIndex] = np.sqrt( dx / spacePolynOrder * dy / spacePolynOrder * np.sum((q_approx[:, :, :, :, PRES] - q_exact[:, :, :, :, PRES])**2) )


        simuIndex += 1


    # PLOTTING RESULTS
    ### Titles
    if testChoice == 1 :
        resultTitle = f"q' + lambda q = 0, avec lambda ~ {round(params, 3)}"
        plotTitle = f"à t = {finalTime} en fonction de h pour $\\frac{{d}}{{dt}}q + \\lambda q = 0$" \
                    f"\n $\\lambda$ = {round(params, 3)}, $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}]"
        
    elif testChoice == 2 :
        resultTitle = f"q' + omega x q = 0, avec omega ~ [{round(params[0], 3)}, {round(params[1], 3)}, {round(params[2], 3)}]"
        plotTitle = f"à t = {finalTime} en fonction de h pour $\\frac{{d}}{{dt}}q + \\omega \\times q = 0$" \
                    f"\n $\\omega \\simeq$ [{round(params[0], 3)}, {round(params[1], 3)}, {round(params[2], 3)}]" \
                    f", $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}]"
        
    elif testChoice == 3 :
        resultTitle = f"(M_x (x) M_y) q' + lambda (M_x (x) M_y) q = 0, avec lambda ~ {round(params, 3)}"
        plotTitle = f"à t = {finalTime} en fonction de h pour $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}]" \
                    f"\n $(M_x \\otimes M_y) \\frac{{d}}{{dt}}q + \\lambda (M_x \\otimes M_y) q = 0$ avec $\\lambda$ = {round(params, 3)}"
        
    elif testChoice == 4 :
        resultTitle = f"(M_x (x) M_y) q' + (M_x (x) M_y) omega x q = 0, avec omega ~ [{round(params[0], 3)}, {round(params[1], 3)}, {round(params[2], 3)}]"
        plotTitle = f"à t = {finalTime} en fonction de h pour $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}]" \
                    f"\n $(M_x \\otimes M_y) \\frac{{d}}{{dt}}q + (M_x \\otimes M_y) \\omega \times q = 0$ avec $\\omega \\simeq$ [{round(params[0], 3)}, {round(params[1], 3)}, {round(params[2], 3)}]"
        
    elif testChoice == 5 :
        resultTitle = f"(M_x (x) M_y) q' + c (D_x (x) M_y) q = 0, avec c ~ {round(params, 3)}"
        plotTitle = f"à t = {finalTime} en fonction de h pour $\\partial_t q + c \\partial_x q = 0$" \
                    f"\n c = {round(params, 3)}, $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}], $\\alpha$={alpha:6.3e}, CFL ={round(CFL, 3)}"
        
    elif testChoice == 6 :
        resultTitle = f"(M_x (x) M_y + stab) q' + [c_x (D_x (x) M_y) + c_y (M_x (x) D_y) + stab] q = 0, avec c ~ {round(params, 3)}"
        plotTitle = f"à t = {finalTime} en fonction de h pour $\\partial_t q + c \\cdot \\nabla q = 0$" \
                    f"\n c = {round(params, 3)}, $q_0 \\simeq$ {round(u0, 3)}"

    ### Exact values in text file
    if testChoice == 6 :
        with open(f"{plotsLocation}Test_{testChoice}_Conv_DeC_P{spacePolynOrder}M{nbSubTimeSteps}C{nbCorrections}.txt", 'a') as f:
            f.write(f"=== Test de DeC (P{spacePolynOrder}, M{nbSubTimeSteps}, C{nbCorrections}) sur l'EDO {resultTitle} et q0 ~ {round(u0, 3)}  ===" + "\n" * 2)
            f.write(f"{'h':>12} | {'erreurs sup':>14} | {'ordre sup':>14} | {'erreurs L2':>14} | {'ordre L2':>14} \n")
            f.write("-" * 13 + ("+" + "-" * 16) * 4 + "" + "\n")
            f.write(f"{hList[0]:>12.3e} | {u_finalTime_sup_errors[0]:>14.3e} | {" - ":>14} | {u_finalTime_L2_errors[0]:>14.3e} | {" - ":>14} | \n")
            for k in range(1, len(hList)):
                ordre_sup_u =  (np.log(u_finalTime_sup_errors[k]) - np.log(u_finalTime_sup_errors[k-1])) / (np.log(hList[k]) - np.log(hList[k-1]))
                ordre_L2_u =  (np.log(u_finalTime_L2_errors[k]) - np.log(u_finalTime_L2_errors[k-1])) / (np.log(hList[k]) - np.log(hList[k-1]))
                f.write(f"{hList[k]:>12.3e} | {u_finalTime_sup_errors[k]:>14.3e} | {round(ordre_sup_u, 3):>14} | {u_finalTime_L2_errors[k]:>14.3e} | {round(ordre_L2_u, 3):>14} \n")
            f.write("\n" * 3)

    else :
        with open(f"{plotsLocation}Test_{testChoice}_Conv_DeC_{spacePolynOrder}.txt", 'a') as f:
            f.write(f"=== Test de DeC (P{spacePolynOrder}, M{nbSubTimeSteps}, C{nbCorrections}) sur l'EDO {resultTitle} et q0 ~ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}]  === \n\n")
            f.write(f"Autres paramètres : alpha = {alpha:6.3e}, CFL = {round(CFL, 3)} \n\n")
            f.write(f"--- Erreurs en norme sup : \n")
            f.write(f"{'h':>12} | {'u':>14} | {'ordre':>14} | {'v':>14} | {'ordre':>14} | {'p':>14} | {'ordre':>14}\n")
            f.write("-" * 13 + ("+" + "-" * 16) * 6 + "" + "\n")
            f.write(f"{hList[0]:>12.3e} | {u_finalTime_sup_errors[0]:>14.3e} | {" - ":>14} | {v_finalTime_sup_errors[0]:>14.3e} | {" - ":>14} | {p_finalTime_sup_errors[0]:>14.3e} | {" - ":>14} \n")
            for k in range(1, len(hList)):
                ordre_sup_u =  (np.log(u_finalTime_sup_errors[k]) - np.log(u_finalTime_sup_errors[k-1])) / (np.log(hList[k]) - np.log(hList[k-1]))
                ordre_sup_v =  (np.log(v_finalTime_sup_errors[k]) - np.log(v_finalTime_sup_errors[k-1])) / (np.log(hList[k]) - np.log(hList[k-1]))
                ordre_sup_p =  (np.log(p_finalTime_sup_errors[k]) - np.log(p_finalTime_sup_errors[k-1])) / (np.log(hList[k]) - np.log(hList[k-1]))
                # f.write(f"{hList[k]:>12.3e} | {u_finalTime_sup_errors[k]:>14.3e} | {ordre_u:>14.3e} | {v_finalTime_sup_errors[k]:>14.3e} | {ordre_v:>14.3e} | {p_finalTime_sup_errors[k]:>14.3e} | {ordre_p:>14.3e} \n")
                f.write(f"{hList[k]:>12.3e} | {u_finalTime_sup_errors[k]:>14.3e} | {round(ordre_sup_u, 3):>14} | {v_finalTime_sup_errors[k]:>14.3e} | {round(ordre_sup_v, 3):>14} | {p_finalTime_sup_errors[k]:>14.3e} | {round(ordre_sup_p, 3):>14} \n")

            f.write(f"\n--- Erreurs en norme L2 : \n")
            f.write(f"{'h':>12} | {'u':>14} | {'ordre':>14} | {'v':>14} | {'ordre':>14} | {'p':>14} | {'ordre':>14}\n")
            f.write("-" * 13 + ("+" + "-" * 16) * 6 + "" + "\n")
            f.write(f"{hList[0]:>12.3e} | {u_finalTime_L2_errors[0]:>14.3e} | {" - ":>14} | {v_finalTime_L2_errors[0]:>14.3e} | {" - ":>14} | {p_finalTime_L2_errors[0]:>14.3e} | {" - ":>14} \n")
            for k in range(1, len(hList)):
                ordre_L2_u =  (np.log(u_finalTime_L2_errors[k]) - np.log(u_finalTime_L2_errors[k-1])) / (np.log(hList[k]) - np.log(hList[k-1]))
                ordre_L2_v =  (np.log(v_finalTime_L2_errors[k]) - np.log(v_finalTime_L2_errors[k-1])) / (np.log(hList[k]) - np.log(hList[k-1]))
                ordre_L2_p =  (np.log(p_finalTime_L2_errors[k]) - np.log(p_finalTime_L2_errors[k-1])) / (np.log(hList[k]) - np.log(hList[k-1]))
                # f.write(f"{hList[k]:>12.3e} | {u_finalTime_L2_errors[k]:>14.3e} | {ordre_u:>14.3e} | {v_finalTime_L2_errors[k]:>14.3e} | {ordre_v:>14.3e} | {p_finalTime_L2_errors[k]:>14.3e} | {ordre_p:>14.3e} \n")
                f.write(f"{hList[k]:>12.3e} | {u_finalTime_L2_errors[k]:>14.3e} | {round(ordre_L2_u, 3):>14} | {v_finalTime_L2_errors[k]:>14.3e} | {round(ordre_L2_v, 3):>14} | {p_finalTime_L2_errors[k]:>14.3e} | {round(ordre_L2_p, 3):>14} \n")
            f.write("\n" * 3)

    # Plots
    with PdfPages(f"{plotsLocation}Test_{testChoice}_Conv_DeC_P{spacePolynOrder}M{nbSubTimeSteps}C{nbCorrections}") as pdf:
        # Evolutions des erreurs sup :
        plt.figure()
        plt.title(f"Erreurs sup {plotTitle}")
        plt.plot(hList, u_finalTime_sup_errors, label="u", marker="o")
        plt.plot(hList, v_finalTime_sup_errors, label="v", marker="s")
        plt.plot(hList, p_finalTime_sup_errors, label="p", marker="^")
        plt.grid(alpha=0.5, ls="--")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("h")
        plt.ylabel("$||q^N - q(T_f)||_{{\\infty}}$")
        plt.legend()
        pdf.savefig()
        plt.close()

        # Evolutions des erreurs L2 :
        plt.figure()
        plt.title(f"Erreurs L2 {plotTitle}")
        plt.plot(hList, u_finalTime_L2_errors, label="u", marker="o")
        plt.plot(hList, v_finalTime_L2_errors, label="v", marker="s")
        plt.plot(hList, p_finalTime_L2_errors, label="p", marker="^")
        plt.grid(alpha=0.5, ls="--")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("h")
        plt.ylabel("$||q^N - q(T_f)||_{{L^2}}$")
        plt.legend()
        pdf.savefig()
        plt.close()

