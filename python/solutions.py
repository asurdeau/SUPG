import numpy as np

# Modules perso
import config
from config import XVEL, YVEL, PRES


# Analytical solutions : Periodic BC
def getAnalyticalPeriodicSolution(t, x, y, physParams):
    theta = (physParams["theta"] / 180.) * np.pi
    lamb = physParams["lambda"]
    c = physParams["speed"]
    
    alpha = 2. * np.pi / (lamb * max(np.cos(theta), np.sin(theta))) 
    ksi = x * np.cos(theta) + y * np.sin(theta)

    u = -0.5 / c * (np.cos(alpha * (ksi + c*t)) - np.cos(alpha * (ksi - c*t))) * np.cos(theta)
    v = -0.5 / c * (np.cos(alpha * (ksi + c*t)) - np.cos(alpha * (ksi - c*t))) * np.sin(theta)

    p = 0.5 * (np.cos(alpha * (ksi + c*t)) + np.cos(alpha * (ksi - c*t)))

    return np.stack([u, v, p], axis=-1)




# Analytical stationary divergence-free solutions : Dirichlet BC
def getRadialProfile(X, Y, x0, y0, r0, profileChoice):
    Rho = np.sqrt((X- x0)**2 + (Y- y0)**2) / r0
    F = np.zeros((np.shape(X)))

    if (profileChoice == 1):
        F[Rho < 1] = 12. * np.pi * np.sqrt(0.981) / (r0 * (np.sqrt(315. * np.pi**2 - 2048.))) \
                     * (1. + np.cos(np.pi * Rho[Rho < 1]))**2
    if (profileChoice == 2):
        F[Rho < 1] = 0.4 * np.exp(- 0.5 / (1. - Rho[Rho < 1])**2) \
                     * np.sqrt(9.81 / (r0 * (1. - Rho[Rho < 1])**3))
    
    return F
    

def getVortexSolution(X, Y, solParams):

    r0, profileChoice, [x0, y0] = solParams["vortex"].values()
    F = getRadialProfile(X, Y, x0, y0, r0, profileChoice)

    u = F * (Y - y0)
    v = - F * (X - x0)
    p = np.ones((np.shape(X)))

    return np.stack([u, v, p], axis=-1)


def addPerturbation(q, X, Y, solParams):
    
    n, [x0, y0], r0 = solParams["gaussian_noise"].values()
    eps = 0.1**n

    p = np.zeros((np.shape(X)))
    rho = np.sqrt( (X - x0)**2 + (Y - y0)**2 ) / r0
    p[rho < 1.] = eps * np.exp( 0.5 * ( 1. - 1./(1. - rho[rho < 1])**2 ) )

    q[:, :, PRES] += p

    return q




# SOLUTION 
def getSolution(t, X, Y, simulationChoice, solParams):

    if simulationChoice == 1 : # 1 : constant
        u = np.full_like(X, solParams["constant"][XVEL])
        v = np.full_like(X, solParams["constant"][YVEL])
        p = np.full_like(X, solParams["constant"][PRES])

        return np.stack([u, v, p], axis=-1)


    elif simulationChoice == 2 : # 2 : analytical periodic
        return getAnalyticalPeriodicSolution(t, X, Y, solParams["analytical_periodic"])


    elif simulationChoice == 3 : # 3 : stationary vortex
            return getVortexSolution(X, Y, solParams)




    elif simulationChoice == 6 : # 6 : 0 + random noise
        # parameters extraction
        n1, n2 = solParams["noise_range"]

        # draw of uniform matrices for the absolute values and for the sign
        rng = np.random.default_rng()
        random_abs = np.exp(- ((n2 - n1) * rng.binomial(n=1, p=0.5, size=(np.shape(X) + (3,))) + n1) * np.log(10.) )
        random_sign = rng.random(np.shape(X) + (3,))

        q = random_abs * random_sign
        return q
    

    if simulationChoice == 7 : # 7 : checkerboard
        modes = solParams["checkerboard"]["modes"]
        amplitudes = solParams["checkerboard"]["amplitudes"]

        q = np.zeros((np.shape(X))+(3,))
        varList = [XVEL, YVEL, PRES]

        for var in varList:
            q[:, :, var] = amplitudes[var]
            if modes[var] == 1:
                q[1::2, 0::2, var] = - amplitudes[var]
                q[0::2, 1::2, var] = - amplitudes[var]

            if modes[var] == 2:
                q[1::2, :, var] = - amplitudes[var]
            
            if modes[var] == 3:
                q[:, 1::2, var] = - amplitudes[var]

        return q





####################################################################################################################

#                                                  HIGHER ORDER                                                    #

####################################################################################################################


