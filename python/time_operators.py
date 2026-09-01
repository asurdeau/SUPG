# Modules généraux
import numpy as np
from numba import njit
import time

# Modules perso
import config
from config import XVEL, YVEL, PRES
import grid_mod
from spatial_operators import getApproxDivFlux, get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator, get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator
import spatial_operators


####################################################################################################################

#                                                 FIRST ORDER UPDATE                                               #

####################################################################################################################



def getForwardEulerUpdate(q, dt, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, simulationChoice, operators, operatorsCoeff, \
                          divFluxTimeTable, updateTimeTable, currentIteration):
    
    # CALCUL DE DIV.FLUXES
    t0 = time.time()
    divF = getApproxDivFlux(q, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, operators, operatorsCoeff)
    t1 = time.time()

    divFluxTimeTable[currentIteration] = t1 - t0
        
    # MISES À JOUR
    t2 = time.time()
    q[iMin:iMax+1, jMin:jMax+1] += - dt * divF[iMin:iMax+1, jMin:jMax+1]
    t3 = time.time()

    updateTimeTable[currentIteration] = t3 - t2

    ### Conditions de bords
    if (simulationChoice == 6 or simulationChoice == 7):     # vortex simulation : Dirichlet BC
        gridOp.dirichlet(q)
    else :                                                   # else (default) : periodic BC 
        gridOp.periodize(q) 

    return q



def getForwardEulerUpdateAndPhysicalTerms(q, dt, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, simulationChoice, operators, operatorsCoeff, \
                                          divFluxTimeTable, updateTimeTable, currentIteration):
    
    # CALCUL DE DIV.FLUXES
    t0 = time.time()
    divF, physicalTerms = getApproxDivFlux(q, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, operators, operatorsCoeff)
    t1 = time.time()

    divFluxTimeTable[currentIteration] = t1 - t0
        
    # MISES À JOUR
    t2 = time.time()
    q[iMin:iMax+1, jMin:jMax+1] += - dt * divF[iMin:iMax+1, jMin:jMax+1]
    t3 = time.time()

    updateTimeTable[currentIteration] = t3 - t2

    ### Conditions de bords
    if (simulationChoice == 6 or simulationChoice == 7):     # vortex simulation : Dirichlet BC
        gridOp.dirichlet(q)
    else :                                                   # else (default) : periodic BC 
        gridOp.periodize(q) 

    return q




####################################################################################################################

#                                                     DeC UPDATE                                                   #

####################################################################################################################


@njit
def periodize(q, iMin, iMax, jMin, jMax, nGhost):

    # BANDES
    # bande verticale à gauche 
    q[:iMin, jMin:jMax+1] = q[iMax-nGhost+1:iMax+1, jMin:jMax+1]

    # bande verticale à droite
    q[iMax+1:, jMin:jMax+1] = q[iMin:iMin+nGhost, jMin:jMax+1]

    # bande horizontale en bas
    q[iMin:iMax+1, :jMin] = q[iMin:iMax+1, jMax-nGhost+1:jMax+1]

    # bande horizontale en haut
    q[iMin:iMax+1, jMax+1:] = q[iMin:iMax+1, jMin:jMin+nGhost]

    # COINS
    # bas gauche 
    q[:iMin, :jMin] = q[iMax-nGhost+1:iMax+1, jMax-nGhost+1:jMax+1]

    # bas droit 
    q[iMax+1:, :jMin] = q[iMin:iMin+nGhost, jMax-nGhost+1:jMax+1]

    # haut gauche 
    q[:iMin, jMax+1:] = q[iMax-nGhost+1:iMax+1, jMin:jMin+nGhost]

    # bas droit 
    q[iMax+1:, jMax+1:] = q[iMin:iMin+nGhost, jMin:jMin+nGhost]

    return q



def getLagInterpPolynCoefs(X):
  N = len(X)
  A = np.zeros((N, N))
  for i in range(N):
      for j in range(N):
          A[i, j] = (X[j])**i

  coeffMat = np.linalg.inv(A)
  return coeffMat


def getGLoPoints(nPoints):

  if nPoints == 1:
    pointsList = np.array([-1.])
  if nPoints == 2:
    pointsList = np.array([-1., 1.])
  if nPoints == 3:
    pointsList = np.array([-1., 0., 1.])
  if nPoints == 4:
    pointsList = np.array([-1., - np.sqrt(5.) / 5., np.sqrt(5) / 5., 1.])
  if nPoints == 5:
    pointsList = np.array([-1., - np.sqrt(21.) / 7., 0., np.sqrt(21) / 7., 1.])
  
  return (pointsList + 1.) / 2.


def getGLoBasisFunctionsSubIntegrals(nPoints):
  points = getGLoPoints(nPoints)
  coeffMat = getLagInterpPolynCoefs(points)

  interpPoints = np.zeros((nPoints, nPoints))
  for i in range(nPoints):
    for j in range(1, nPoints):
       interpPoints[i, j] = ((points[j])**(i+1) - (-1)**(i+1)) / (i+1.)

  vals = coeffMat @ interpPoints

  return vals



@njit
def getDecUpdate(q, dt, order, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs):

    P = order
    M = int((P+1.) / 2)
    # INITIALISATION DES VARIABLES : TOUS LES q 
    shape = np.shape(q)
    q_current_stage = np.zeros(shape + (M,))
    weighted_evolved_q_first_stage = np.zeros(shape)
    first_stage_time_weights_sums = np.zeros(M)
    weighted_evolved_q_m_current_stage = np.zeros(shape + (P-1,))
    weighted_sum_of_weighted_evolved_q_m_current_stage = np.zeros(shape)
    weighted_massed_current_stage_differences = np.zeros(shape + (P-1,))
    last_weighted_massed_current_stage_differences = np.zeros(shape)
    q_next_stage = np.zeros(shape + (M,))
    q_updated = np.zeros(shape)

    # TIME WEIGHTS theta^m_r
    theta = getGLoBasisFunctionsSubIntegrals(M)

    # INITIAL STAGE : q^{(0), m} = q_n
    for m in range(M):
        q_current_stage[:, :, :, :, :, m] = q


    # FIRST STAGE : EASY TO COMPUTE : NO NEED OF THE MASS STILL
    for m in range(M):
      first_stage_time_weights_sums[m] =  np.sum(theta[:, m])
    weighted_evolved_q_first_stage = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q, iMin, iMax, jMin, jMax, order, evolCoeffs)

    for i in range(iMin, iMax+1):
      for j in range(jMin, jMax+1):
        for k in range(order):
          for l in range(order):
            for var in range(3):
              for m in range(M):
                q_next_stage[i, j, k, l, var, m] = q[i, j, k, l, var] - dt * first_stage_time_weights_sums[m] * weighted_evolved_q_first_stage[i, j, k, l, var]

    periodize(q_next_stage, iMin, iMax, jMin, jMax, nGhost)
    q_current_stage = q_next_stage.copy()

    for p in range(2, P):
      for m in range(M):
        weighted_evolved_q_m_current_stage[:, :, :, :, :, m] = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q_current_stage[:, :, :, :, :, m], iMin, iMax, jMin, jMax, order, evolCoeffs)
        weighted_massed_current_stage_differences[:, :, :, :, :, m] = get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator(q_current_stage[:, :, :, :, :, m] - q, iMin, iMax, jMin, jMax, order, massCoeffs)

      for i in range(iMin, iMax+1):
        for j in range(jMin, jMax+1):
          for k in range(order):
            for l in range(order):
              for var in range(3):
                for m in range(M):
                  weighted_sum_of_weighted_evolved_q_m_current_stage[i, j, k, l, var] = theta[0, m] * weighted_evolved_q_m_current_stage[i, j, k, l, var, 0]
                  for r in range(1, M):
                    weighted_sum_of_weighted_evolved_q_m_current_stage[i, j, k, l, var] += theta[r, m] * weighted_evolved_q_m_current_stage[i, j, k, l, var, r]

                  q_next_stage[i, j, k, l, var, m] = q_current_stage[i, j, k, l, var, m] \
                                                      - weighted_massed_current_stage_differences[i, j, k, l, var, m] \
                                                      - dt * weighted_sum_of_weighted_evolved_q_m_current_stage[i, j, k, l, var]

      
      periodize(q_next_stage, iMin, iMax, jMin, jMax, nGhost)
      q_current_stage = q_next_stage.copy()
    
       
    # WATCH OUT : WE NEED TO APPLY THE BC EACH TIME





    # LAST STAGE P : ONLY COMPUTE q^{(P), M}
    for m in range(M):
      weighted_evolved_q_m_current_stage[:, :, :, :, :, m] = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q_current_stage[:, :, :, :, :, m], iMin, iMax, jMin, jMax, order, evolCoeffs)

    last_weighted_massed_current_stage_differences[:, :, :, :, :] = get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator(q_current_stage[:, :, :, :, :, m] - q, iMin, iMax, jMin, jMax, order, massCoeffs)

    for i in range(iMin, iMax+1):
      for j in range(jMin, jMax+1):
        for k in range(order):
          for l in range(order):
            for var in range(3):
              weighted_sum_of_weighted_evolved_q_m_current_stage[i, j, k, l, var] = theta[0, M] * weighted_evolved_q_m_current_stage[i, j, k, l, var, 0]
              for r in range(1, M):
                weighted_sum_of_weighted_evolved_q_m_current_stage[i, j, k, l, var] += theta[r, M] * weighted_evolved_q_m_current_stage[i, j, k, l, var, r]

              q_updated[i, j, k, l, var] = q_current_stage[i, j, k, l, var, M] \
                                            - last_weighted_massed_current_stage_differences[i, j, k, l, var] \
                                            - dt * weighted_sum_of_weighted_evolved_q_m_current_stage[i, j, k, l, var]

    return q_updated