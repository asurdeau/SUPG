import numpy as np

# Modules perso
import config
from config import XVEL, YVEL, PRES


# Analytical solutions : Periodic BC
def getAnalyticalPeriodicSolution(t, x, y, physParams):
    theta = physParams["theta"] * np.pi
    lamb = physParams["lambda"]
    c = physParams["speed"]
    
    alpha = 2. * np.pi / (lamb * max(np.cos(theta), np.sin(theta))) 
    ksi = x * np.cos(theta) + y * np.sin(theta)

    u = -0.5 / c * (np.cos(alpha * ksi + c*t) - np.cos(alpha * ksi - c*t)) * np.cos(theta)
    v = -0.5 / c * (np.cos(alpha * ksi + c*t) - np.cos(alpha * ksi - c*t)) * np.sin(theta)

    p = 0.5 * (np.cos(alpha * ksi + c*t) + np.cos(alpha * ksi - c*t))

    return np.stack([u, v, p], axis=-1)




# Analytical divergence-free solutions : Dirichlet BC





# SOLUTION 
def getSolution(t, grid, solParams):
    simChoice = solParams["simulation_choice"]
    X = grid.xValidGrid
    Y = grid.yValidGrid

    if simChoice == 1 : # 1 : constant
        u = solParams["constant"][0] * np.ones((np.shape(X))) 
        v = solParams["constant"][1] * np.ones((np.shape(X))) 
        p = solParams["constant"][2] * np.ones((np.shape(X))) 
        return np.stack([u, v, p], axis=-1)
    

    if simChoice == 2 : # 2 : 0 + random noise
        # parameters extraction
        n1, n2 = solParams["noise_range"]

        # draw of uniform matrices for the absolute values and for the sign
        rng = np.random.default_rng()
        random_abs = np.exp(- ((n2 - n1) * rng.binomial(n=1, p=0.5, size=(np.shape(X) + (3,))) + n1) * np.log(10.) )
        random_sign = rng.random(np.shape(X) + (3,))


        u = random_abs[:, :, 0] * random_sign[:, :, 0]
        v = random_abs[:, :, 1] * random_sign[:, :, 1]
        p = random_abs[:, :, 2] * random_sign[:, :, 2]
        return np.stack([u, v, p], axis=-1)
    

    if simChoice == 3 : # 3 : constant + small gaussian
        # parameters extraction
        u = np.zeros((np.shape(X)))
        v = np.zeros((np.shape(X)))
        p = np.zeros((np.shape(X)))

        n, x0, y0, r0 = solParams["gaussian_noise"]
        eps = 0.1**n

        rho = np.sqrt( (X - x0)**2 + (Y - y0)**2 ) / r0
        p[rho < 1.] = eps * np.exp( 0.5 * ( 1. - 1./(1. - rho[rho < 1])**2 ) )

        return np.stack([u, v, p], axis=-1)
    

    if simChoice == 4 : # 4 : checkerboard
        u0, v0, p0 = solParams["checkerboard"]
        u = u0[0] * np.ones((np.shape(x)))
        v = v0[0] * np.ones((np.shape(x)))
        p = v0[0] * np.ones((np.shape(x)))

        u[1::2, 1::2] = u0[1]
        v[1::2, 1::2] = v0[1]
        p[1::2, 1::2] = p0[1]

        return np.stack([u, v, p], axis=-1)
    

    if simChoice == 5 : # 5 : analytical periodic
        return getAnalyticalPeriodicSolution(t, X, Y, solParams["analytical_periodic"])