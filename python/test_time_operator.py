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



# TEST DE LA RESOLUTION DE L'EDO M d/dt q + E q = 0, 
# A MAILLAGE FIXE


####################################################################################################################

#                                            TEST FONCTIONS AND MATRCIES                                           #

####################################################################################################################


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



def getHeunUpdate(q, M, dt):
    return q - dt * M @ q + 0.5 * dt**2 * M @ M @ q




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




def getTest4Solution(q, lamb, time):
    return getTest1Solution(q, lamb, time)



def get_flattened_quantity(q, xCellNumber, yCellNumber, spacePolynOrder):

    q_flattened = np.zeros((xCellNumber * spacePolynOrder, yCellNumber * spacePolynOrder, 3))

    for k in range(spacePolynOrder):
        for l in range(spacePolynOrder):
            q_flattened[k::spacePolynOrder, l::spacePolynOrder] = q[:, :, k, l]

    return q_flattened





####################################################################################################################

#                                        TIME EVOLUTION COMPARISON WITH EXACT                                      #

####################################################################################################################



# TIME EVOLUTION FOR ONE VALUE OF dt
# if (True) :
if (False) :
    # PARAMETERS
    ### mesh
    Nx, Ny = 100, 100
    nGhost = 2
    iMin, iMax, jMin, jMax = nGhost, Nx + nGhost - 1, nGhost, Ny + nGhost - 1

    ### time
    dt = 0.01
    taux_progression = 0.1
    finalTime = 1.

    ### basis and order
    spacePolynOrder = 2
    basisChoice = "basic"
    evolCoeffs, massCoeffs = getTestCoefficients(basisChoice, spacePolynOrder, params)

    ### test choice
    testChoice = 3

    ### initial data
    norm_q0_min, norm_q0_max = 1., 3.
    lamb_min, lamb_max = 0.1, 1.

    ### plots options
    plotsLocation = "Plots/DeC/"

    

    # INITIAL DATA DRAW
    norm_q0 = (norm_q0_max - norm_q0_min) * np.random.rand() + norm_q0_min
    drawn_variable_q0 = np.random.randn(3)
    u0, v0, p0 = norm_q0 * abs(drawn_variable_q0) / np.linalg.norm(drawn_variable_q0, ord=2)
    q_0 = np.full((Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder, spacePolynOrder, 3), [u0, v0, p0])
    print("Valeurs de départ : ", q_0[0, 0, 0, 0, XVEL], q_0[0, 0, 0, 0, YVEL], q_0[0, 0, 0, 0, PRES])



    ### TEST CHOICE
    if testChoice == 1:
        # TIRAGE UNIFORME SUR [-LAMB_MAX, -LAMB_MIN] U [LAMB_MIN, LAMB_MAX]
        lamb = (2. * np.random.binomial(1, 0.5) - 1.) * ((lamb_max - lamb_min) * np.random.rand() + lamb_min)
        
        params = lamb
        getTestCoefficients = getTest1Coefficients
        getTestSolution = getTest1Solution


    elif testChoice == 2:
        # Tirage d'un vecteur de norme 2pi uniforme sur la sphère
        # tirage d'un vecteur gaussien
        drawn_variable_omega = np.random.randn(3)
        # renormalisation
        omega = 2. * np.pi * drawn_variable_omega / np.linalg.norm(drawn_variable_omega, ord=2)

        params = omega
        getTestCoefficients = getTest2Coefficients
        getTestSolution = getTest2Solution


    if testChoice == 3:
        # TIRAGE UNIFORME SUR [-LAMB_MAX, -LAMB_MIN] U [LAMB_MIN, LAMB_MAX]
        lamb = (2. * np.random.binomial(1, 0.5) - 1.) * ((lamb_max - lamb_min) * np.random.rand() + lamb_min)
        
        params = lamb
        getTestCoefficients = getTest3Coefficients
        getTestSolution = getTest3Solution

    
    # Operators coefficients depending on the choice
    evolCoeffs, massCoeffs = getTestCoefficients(basisChoice, spacePolynOrder, params)


    # OBSERVABLES INITIALISATION
    ### Array building
    max_u_test, max_v_test, max_p_test = - np.ones((3, int(2. * finalTime / dt)))
    u_test_ecart, v_test_ecart, p_test_ecart = - np.ones((3, int(2. * finalTime / dt)))
    ### First observation
    max_u_test[0] = np.max(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, XVEL])
    max_v_test[0] = np.max(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, YVEL])
    max_p_test[0] = np.max(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, PRES])



    print("Début des calculs : ")
    currentTime = 0.
    k = 0

    while ( finalTime > currentTime + 1.e-14  ):

        if ( abs(currentTime + dt - finalTime * taux_progression) < 0.5 * dt ):
            print(" - Progression du calcul : " + str( 100 * round(taux_progression, 3) )+"%")
            taux_progression += 0.1
        if currentTime + dt > finalTime :
            dt = finalTime - currentTime
            currentTime = finalTime
            print(" - Temps final dépassé, on raccourcit le pas de temps")
        k += 1


        # Update approx value
        q_approx = time_operators.getDeCUpdate(q_approx, dt, spacePolynOrder, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs)


        # Add observations
        max_u_test[k] = np.max(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, XVEL])
        max_v_test[k] = np.max(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, YVEL])
        max_p_test[k] = np.max(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, PRES])

        u_test_ecart[k] = max_u_test[k] - np.min(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, XVEL])
        v_test_ecart[k] = max_v_test[k] - np.min(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, YVEL])
        p_test_ecart[k] = max_p_test[k] - np.min(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, PRES])

        currentTime += dt


    # Arrays for plots
    iterList = np.arange(k+1)
    timesList = np.linspace(0, finalTime, k+1)
    q_exact = getTestSolution(np.array([u0, v0, p0]), params, timesList)


    # Title for the plot
    if testChoice == 1 :
        plotTitle = f"Comparaisons de solutions de $\\frac{{d}}{{dt}}q + \\lambda q = 0$ sur $[0, ${finalTime}$]$" \
                  f"\n avec $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}], $\\lambda$ = {round(lamb, 3)} et dt = {round(dt, 4)}"
    elif testChoice == 2 :
        plotTitle = f"Comparaisons de solutions de $\\frac{{d}}{{dt}}q + \\omega \\times q = 0$ sur $[0, ${finalTime}$]$" \
                  f"\n avec $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}], $\\omega$ = [{round(omega[0], 3)}, {round(omega[1], 3)}, {round(omega[2], 3)}] et dt = {round(dt, 4)}"
    if testChoice == 3 :
        plotTitle = f"Comparaisons de solutions de $(M_x \\otimes M_y)\\frac{{d}}{{dt}}q + \\lambda (M_x \\otimes M_y) q = 0$ sur $[0, ${finalTime}$]$" \
                  f"\n avec $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}], $\\lambda$ = {round(lamb, 3)} et dt = {round(dt, 4)}"


    # PLOTS : 
    with PdfPages(f"{plotsLocation}Test_Evol_DeC_{spacePolynOrder}_dt_{round(dt, 4)}_h_{round(1./Nx, 4)}") as pdf:
        # Evolutions des max :
        plt.figure()
        plt.title(plotTitle)
        plt.plot(iterList, max_u_test[:k+1], label="u approchée", color="blue", alpha=0.5, marker="o", markevery=k // 10)
        plt.plot(iterList, q_exact[:, XVEL], label="u exacte", color="blue", alpha=0.5, ls="--")
        plt.plot(iterList, max_v_test[:k+1], label="v approchée", color="green", alpha=0.5, marker="s", markevery=k // 10)
        plt.plot(iterList, q_exact[:, YVEL], label="v exacte", color="green", alpha=0.5, ls="--")
        plt.plot(iterList, max_p_test[:k+1], label="p approchée", color="red", alpha=0.5, marker="^", markevery=k // 10)
        plt.plot(iterList, q_exact[:, PRES], label="p exacte", color="red", alpha=0.5, ls="--")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()
        plt.close()

        # Evolutions des ecarts :
        plt.figure()
        plt.title(f"Ecarts en espace")
        plt.plot(iterList[1:], u_test_ecart[1:k+1], label="u", color="blue", alpha=0.5, marker="o", markevery=k // 10)
        plt.plot(iterList[1:], u_test_ecart[1:k+1], label="v", color="green", alpha=0.5, marker="s", markevery=k // 10)
        plt.plot(iterList[1:], p_test_ecart[1:k+1], label="p", color="red", alpha=0.5, marker="^", markevery=k // 10)
        plt.legend()
        pdf.savefig()
        plt.close()





# Snapshots at different times of approx value, exact and difference
# if (True) :
if (False) :
    # PARAMETERS
    ### mesh 
    xL, xR, yL, yR = 0., 1., 0., 1.
    Nx, Ny = 20, 20
    nGhost = 2
    iMin, iMax, jMin, jMax = nGhost, Nx + nGhost - 1, nGhost, Ny + nGhost - 1

    ### time
    finalTime = 1.
    dt_apriori = 0.01

    ### basis and order
    spacePolynOrder = 2
    nbSubTimeSteps = 2
    nbCorrections = 2
    basisChoice = "basic"

    ### test choice
    testChoice = 3

    ### plots options
    plotsLocation = "Plots/DeC/"
    nbSnapshots = 11
    timeBetweenSnapshots = finalTime / (nbSnapshots-1.)
    ##### 2D mesh for plots
    xCoord_2D = np.linspace(xL - dx * nGhost, xR + dx * nGhost, (Nx + 2*nGhost)*spacePolynOrder, endpoint=False)
    yCoord_2D = np.linspace(yL - dy * nGhost, yR + dy * nGhost, (Ny + 2*nGhost)*spacePolynOrder, endpoint=False)
    X_2D, Y_2D = np.meshgrid(xCoord_2D, yCoord_2D, indexing='ij')

    ### initial data
    norm_q0_min, norm_q0_max = 1., 3.
    lamb_min, lamb_max = 0.1, 1.

    

    # INITIAL DATA DRAW
    norm_q0 = (norm_q0_max - norm_q0_min) * np.random.rand() + norm_q0_min
    drawn_variable_q0 = np.random.randn(3)
    u0, v0, p0 = norm_q0 * abs(drawn_variable_q0) / np.linalg.norm(drawn_variable_q0, ord=2)
    q_0 = np.full((Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder, spacePolynOrder, 3), [u0, v0, p0])
    print("Valeurs de départ : ", q_0[0, 0, 0, 0, XVEL], q_0[0, 0, 0, 0, YVEL], q_0[0, 0, 0, 0, PRES])

    ### Test choice
    if testChoice == 1:
        # TIRAGE UNIFORME SUR [-LAMB_MAX, -LAMB_MIN] U [LAMB_MIN, LAMB_MAX]
        lamb = (2. * np.random.binomial(1, 0.5) - 1.) * ((lamb_max - lamb_min) * np.random.rand() + lamb_min)
        
        params = lamb
        getTestCoefficients = getTest1Coefficients
        getTestSolution = getTest1Solution

    elif testChoice == 2:
        # Tirage d'un vecteur de norme 2pi uniforme sur la sphère
        # tirage d'un vecteur gaussien
        drawn_variable_omega = np.random.randn(3)
        # renormalisation
        omega = 2. * np.pi * drawn_variable_omega / np.linalg.norm(drawn_variable_omega, ord=2)

        params = omega
        getTestCoefficients = getTest2Coefficients
        getTestSolution = getTest2Solution

    elif testChoice == 3:
        # TIRAGE UNIFORME SUR [-LAMB_MAX, -LAMB_MIN] U [LAMB_MIN, LAMB_MAX]
        lamb = (2. * np.random.binomial(1, 0.5) - 1.) * ((lamb_max - lamb_min) * np.random.rand() + lamb_min)
        
        params = lamb
        getTestCoefficients = getTest3Coefficients
        getTestSolution = getTest3Solution


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
    doSnapshotNow = False
    nbSnapshotsDone = 0
    currentTime = 0.
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
    for var, varName in [(XVEL, "U"), (YVEL, "V"), (PRES, "P")] :
        with PdfPages(f"{plotsLocation}Test_Evol_DeC_{spacePolynOrder}_dt_{round(dt, 4)}_h_{round(1./Nx, 4)}_{varName}") as pdf:
            for k in range(nbSnapshots):
                if testChoice == 1 :
                    plotTitle = f"Solutions de $\\frac{{d}}{{dt}}q + \\lambda q = 0$ à t = {time}" \
                            f"\n avec $q_0 \\simeq$ [{round(q0[0], 3)}, {round(q0[1], 3)}, {round(q0[2], 3)}], $\\lambda$ = {round(params, 3)} et dt = {round(dt, 4)}"
                elif testChoice == 2 :
                    plotTitle = f"Solutions de $\\frac{{d}}{{dt}}q + \\omega \\times q = 0$ à t = {time}" \
                            f"\n avec $q_0 \\simeq$ [{round(q0[0], 3)}, {round(q0[1], 3)}, {round(q0[2], 3)}], $\\omega$ = [{round(params[0], 3)}, {round(params[1], 3)}, {round(params[2], 3)}] et dt = {round(dt, 4)}"
                elif testChoice == 3 :
                    plotTitle = f"Solutions de $(M_x \\otimes M_y)\\frac{{d}}{{dt}}q + \\lambda (M_x \\otimes M_y) q = 0$ à t = {time}" \
                            f"\n avec $q_0 \\simeq$ [{round(q0[0], 3)}, {round(q0[1], 3)}, {round(q0[2], 3)}], $\\lambda$ = {round(params, 3)} et dt = {round(dt, 4)}"

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



# CONVERGENCE TEST : END TIME ERRORS WITH VARRYING dt
if (True) :
# if (False) :
    # PARAMETERS
    ### mesh
    xL, xR, yL, yR = 0., 1., 0., 1.
    Nx, Ny = 100, 100
    dx, dy = (xR - xL) / Nx, (yR - yL) / Ny
    nGhost = 2
    iMin, iMax, jMin, jMax = nGhost, Nx + nGhost - 1, nGhost, Ny + nGhost - 1

    ### time
    finalTime = 1.
    dtList = 10**(np.linspace(-1, -2, 5))

    ### basis and order
    spacePolynOrder = 2
    nbSubTimeSteps = 2
    nbCorrections = 3
    basisChoice = "basic"

    ### test choice
    testChoice = 3

    ### initial data
    norm_q0_min, norm_q0_max = 1., 3.
    lamb_min, lamb_max = 0.1, 1.

    ### plots options
    plotsLocation = "Plots/DeC/"
    


    # INITIAL DATA DRAW
    norm_q0 = (norm_q0_max - norm_q0_min) * np.random.rand() + norm_q0_min
    drawn_variable_q0 = np.random.randn(3)
    u0, v0, p0 = norm_q0 * abs(drawn_variable_q0) / np.linalg.norm(drawn_variable_q0, ord=2)
    q_0 = np.full((Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder, spacePolynOrder, 3), [u0, v0, p0])
    print("Valeurs de départ : ", q_0[0, 0, 0, 0, XVEL], q_0[0, 0, 0, 0, YVEL], q_0[0, 0, 0, 0, PRES])

    ### Test choice
    if testChoice == 1:
        # TIRAGE UNIFORME SUR [-LAMB_MAX, -LAMB_MIN] U [LAMB_MIN, LAMB_MAX]
        lamb = (2. * np.random.binomial(1, 0.5) - 1.) * ((lamb_max - lamb_min) * np.random.rand() + lamb_min)
        
        params = lamb
        getTestCoefficients = getTest1Coefficients
        getTestSolution = getTest1Solution

    elif testChoice == 2:
        # Tirage d'un vecteur de norme 2pi uniforme sur la sphère
        # tirage d'un vecteur gaussien
        drawn_variable_omega = np.random.randn(3)
        # renormalisation
        omega = 2. * np.pi * drawn_variable_omega / np.linalg.norm(drawn_variable_omega, ord=2)

        params = omega
        getTestCoefficients = getTest2Coefficients
        getTestSolution = getTest2Solution

    elif testChoice == 3:
        # TIRAGE UNIFORME SUR [-LAMB_MAX, -LAMB_MIN] U [LAMB_MIN, LAMB_MAX]
        lamb = (2. * np.random.binomial(1, 0.5) - 1.) * ((lamb_max - lamb_min) * np.random.rand() + lamb_min)
        
        params = lamb
        getTestCoefficients = getTest3Coefficients
        getTestSolution = getTest3Solution


    ### "4D" mesh for initial data
    xCoord = np.linspace(xL - dx * nGhost, xR + dx * nGhost, Nx + 2*nGhost , endpoint=False)
    yCoord = np.linspace(yL - dy * nGhost, yR + dy * nGhost, Ny + 2*nGhost, endpoint=False)
    X_FO, Y_FO = np.meshgrid(xCoord, yCoord, indexing='ij')
    X, Y = np.zeros((2, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder, spacePolynOrder, 3)) 
    for k in range(spacePolynOrder):
        for l in range(spacePolynOrder):
            for var in [XVEL, YVEL, PRES]:
                X[:, :, k, l, var] = X_FO + dx * (1. * k) / spacePolynOrder
                Y[:, :, k, l, var] = Y_FO + dy * (1. * l) / spacePolynOrder

    # FINAL TIME SOLUTION
    q_exact = getTestSolution(q_0, params, finalTime, X, Y)

    # Operators coefficients depending on the choice
    evolCoeffs, massCoeffs = getTestCoefficients(basisChoice, spacePolynOrder, params)


    # OBSERVABLES INITIALISATION
    ### Array building
    u_finalTime_errors, v_finalTime_errors, p_finalTime_errors = - np.ones((3, len(dtList)))
    k = 0

    for dt in dtList :

        # Reset values 
        print("Début des calculs pour dt = ", dt, " : ", flush=True)
        q_approx = getTestSolution(q_0, params, 0., X, Y)
        currentTime = 0.
        taux_progression = 0.2

        # Time loop for the given dt
        while ( abs(finalTime - currentTime) > 1.e-14  ):

            # dt and currentTime adaptation in case near end
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
            q_approx = time_operators.getDeCUpdate(q_approx, dt, spacePolynOrder, nbSubTimeSteps, nbCorrections, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs)


        u_finalTime_errors[k] = np.max(abs(q_approx[:, :, :, :, XVEL] - q_exact[:, :, :, :, XVEL]))
        v_finalTime_errors[k] = np.max(abs(q_approx[:, :, :, :, YVEL] - q_exact[:, :, :, :, YVEL]))
        p_finalTime_errors[k] = np.max(abs(q_approx[:, :, :, :, PRES] - q_exact[:, :, :, :, PRES]))
        k += 1


    if testChoice == 1 :
        resultTitle = f"q' + lambda q = 0, avec lambda ~ {round(lamb, 3)}"
    elif testChoice == 2 :
        resultTitle = f"q' + omega x q = 0, avec omega ~ [{round(omega[0], 3)}, {round(omega[1], 3)}, {round(omega[2], 3)}]"
    elif testChoice == 3 :
        resultTitle = f"(M_x (x) M_y + \\mathcal{{A}}_{{\\mathsf{{SU}}}}) q' + lambda (M_x (x) M_y + \\mathcal{{E}}_{{\\mathsf{{SU}}}}) q = 0, avec lambda ~ {round(lamb, 3)}"

    PlotsLocation = "Plots/DevDeC/"
    with open(f"{plotsLocation}Test_Conv_DeC_{spacePolynOrder}.txt", 'a') as f:
        f.write(f"=== Test de DeC ordre {spacePolynOrder} sur l'EDO {resultTitle} et q0 ~ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}]  ===" + "\n" * 2)
        f.write(f"{'dt':>12} | {'u':>14} | {'ordre':>14} | {'v':>14} | {'ordre':>14} | {'p':>14} | {'ordre':>14}\n")
        f.write("-" * 13 + ("+" + "-" * 16) * 6 + "" + "\n")
        f.write(f"{dtList[0]:>12.3e} | {u_finalTime_errors[0]:>14.3e} | {" - ":>14} | {v_finalTime_errors[0]:>14.3e} | {" - ":>14} | {p_finalTime_errors[0]:>14.3e} | {" - ":>14} \n")
        for k in range(1, len(dtList)):
            ordre_u =  (np.log(u_finalTime_errors[k]) - np.log(u_finalTime_errors[k-1])) / (np.log(dtList[k]) - np.log(dtList[k-1]))
            ordre_v =  (np.log(v_finalTime_errors[k]) - np.log(v_finalTime_errors[k-1])) / (np.log(dtList[k]) - np.log(dtList[k-1]))
            ordre_p =  (np.log(p_finalTime_errors[k]) - np.log(p_finalTime_errors[k-1])) / (np.log(dtList[k]) - np.log(dtList[k-1]))
            # f.write(f"{dtList[k]:>12.3e} | {u_finalTime_errors[k]:>14.3e} | {ordre_u:>14.3e} | {v_finalTime_errors[k]:>14.3e} | {ordre_v:>14.3e} | {p_finalTime_errors[k]:>14.3e} | {ordre_p:>14.3e} \n")
            f.write(f"{dtList[k]:>12.3e} | {u_finalTime_errors[k]:>14.3e} | {round(ordre_u, 3):>14} | {v_finalTime_errors[k]:>14.3e} | {round(ordre_v, 3):>14} | {p_finalTime_errors[k]:>14.3e} | {round(ordre_p, 3):>14} \n")
        f.write("\n" * 3)


    if testChoice == 1 :
        plotTitle = f"Erreurs sup à t = {finalTime} en fonction de $\\Delta t$ pour $\\frac{{d}}{{dt}}q + \\lambda q = 0$" \
                    f"\n avec $\\lambda$ = {round(lamb, 3)} et $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}]"
    elif testChoice == 2 :
        plotTitle = f"Erreurs sup à t = {finalTime} en fonction de $\\Delta t$ pour $\\frac{{d}}{{dt}}q + \\omega \\times q = 0$" \
                    f"\n avec $\\omega \\simeq$ [{round(omega[0], 3)}, {round(omega[1], 3)}, {round(omega[2], 3)}]" \
                    f" et $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}]"
    elif testChoice == 3 :
        plotTitle = f"Erreurs sup à t = {finalTime} en fonction de $\\Delta t$ pour $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}]" \
                    f"\n $(M_x \\otimes M_y) \\frac{{d}}{{dt}}q + \\lambda (M_x \\otimes M_y) q = 0$ avec $\\lambda$ = {round(lamb, 3)}"

    # Comparaison avec solution de référence
    with PdfPages(f"{PlotsLocation}Test_Conv_DeC_{spacePolynOrder}") as pdf:
        # Evolutions des max :
        plt.figure()
        plt.title(plotTitle)
        plt.plot(dtList, u_finalTime_errors, label="u", marker="o")
        plt.plot(dtList, v_finalTime_errors, label="v", marker="s")
        plt.plot(dtList, p_finalTime_errors, label="p", marker="^")
        plt.grid(alpha=0.5, ls="--")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("$\\Delta t$")
        plt.ylabel("$||q^N - q(T_f)||$")
        plt.legend()
        pdf.savefig()
        plt.close()




