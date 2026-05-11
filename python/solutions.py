import numpy as np

# Modules perso
import config
from config import XVEL, YVEL, PRES


# Analytical solutions : Periodic BC
def getAnalyticalPeriodicSolution(t, x, y, physParams):
    theta = physParams["theta"] * np.pi
    lamb = physParams["lambda"]
    c = physParams["speed"]
    
    alpha = 2. * np.pi / (lamb * np.cos(theta)) 
    ksi = x * np.cos(theta) + y * np.sin(theta)

    u = -0.5 / c * (np.cos(alpha * ksi + c*t) - np.cos(alpha * ksi - c*t)) * np.cos(theta)
    v = -0.5 / c * (np.cos(alpha * ksi + c*t) - np.cos(alpha * ksi - c*t)) * np.sin(theta)

    p = 0.5 * (np.cos(alpha * ksi + c*t) + np.cos(alpha * ksi - c*t))

    return np.stack([u, v, p], axis=-1)




# Analytical divergence-free solutions : Dirichlet BC





# SOLUTION 
def getSolution(t, x, y, solParams):
    simChoice = solParams["simulation_choice"]
    if simChoice == 1 :
        u0, v0, p0 = solParams["constant"]
        return np.stack([np.full_like(x, u0), np.full_like(x, v0), np.full_like(x, p0)], axis=-1)
    if simChoice == 2 :
        return getAnalyticalPeriodicSolution(t, x, y, solParams["analytical_periodic"])