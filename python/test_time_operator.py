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
# from plots import makePlots, openGifWriters, closeGifWriters
import time_operators
from spatial_operators import getApproxDivFlux, get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator, get_arbitrar_order_SUPG_weighted_evolution_operator_coeffs
from plots import getVisualisationData



# TEST DE LA RESOLUTION DE L'EDO M d/dt q + E q = 0, 
# A MAILLAGE FIXE


####################################################################################################################

#                                            TEST FONCTIONS AND MATRCIES                                           #

####################################################################################################################


# LINEAR SYSTEM OF DECOUPLED EQUATIONS
def getTest1Coefficients(lamb):
    evolCoeffs = np.zeros((20, 3, order, order)) 
    massCoeffs = np.zeros((20, 3, order, order))

    Id = np.array([np.zeros((order, order)), np.eye(order), np.zeros((order, order))])
    
    listCoeffs = [0, 8, 16]
    for i in listCoeffs :
        evolCoeffs[i] = lamb * Id
        evolCoeffs[i+1] = Id
        massCoeffs[i] = Id
        massCoeffs[i+1] = Id

    return evolCoeffs, massCoeffs


def getTest1Solution(q, lamb, time):
    if len(np.shape(time)) > 1:
        time = time.reshape(time.shape + (1,) * q.ndim)
    return q.copy() * np.exp(- lamb * time)




# LINEAR SYSTEM OF COUPLED EQUATIONS : ROTATION AROUND omega
def getTest2Coefficients(omega):
    a, b, c = omega
    evolCoeffs = np.zeros((20, 3, order, order)) 
    massCoeffs = np.zeros((20, 3, order, order))

    Id = np.array([np.zeros((order, order)), np.eye(order), np.zeros((order, order))])

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


def getTest2Solution(q, omega, time):
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



####################################################################################################################

#                                              OPERATORS EVALUATIONS                                               #

####################################################################################################################



# if (True) :
if (False) :
    rng = np.random.default_rng()
    random_sign = 2. * np.random.binomial(1, 0.5, (10, 10)) - 1.

    print(random_sign)




####################################################################################################################

#                                        TIME EVOLUTION COMPARISON WITH EXACT                                      #

####################################################################################################################



# TIME EVOLUTION FOR ONE VALUE OF dt
# if (True) :
if (False) :
    Nx, Ny = 20, 20
    nGhost = 2
    order = 1

    testChoice = 2

    iMin, iMax, jMin, jMax = nGhost, Nx + nGhost - 1, nGhost, Ny + nGhost - 1

    lamb = 2. * np.random.rand() - 1.
    # omega = np.array([1., 0., 0.])
    # omega = np.array([0., 1., 0.])
    # omega = np.array([0., 0., 1.])
    omega = np.array([1., 1., 1.]) / np.sqrt(3.)

    omega = 2. * np.random.rand(3) - 1.
    omega = 2. * np.pi * omega / np.linalg.norm(omega, ord=2)

    # INITIAL DATA
    u0, v0, p0 = 1., 2., 3.
    # u0, v0, p0 = 1.j, 2.j, 3.j
    q_approx = np.full((Nx + 2*nGhost, Ny + 2*nGhost, order, order, 3), [u0, v0, p0])
    finalTime = 1.


    # Test choice :
    if testChoice == 1:
        params = lamb
        getTestCoefficients = getTest1Coefficients
        getTestSolution = getTest1Solution
    elif testChoice == 2:
        params = omega
        getTestCoefficients = getTest2Coefficients
        getTestSolution = getTest2Solution
    
    evolCoeffs, massCoeffs = getTestCoefficients(params)

    currentTime = 0.
    dt = 0.01
    taux_progression = 0.1
    k = 0

    max_u_test, max_v_test, max_p_test = - np.ones((3, int(2. * finalTime / dt)))
    u_test_ecart, v_test_ecart, p_test_ecart = - np.ones((3, int(2. * finalTime / dt)))
    max_u_test[0] = np.max(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, XVEL])
    max_v_test[0] = np.max(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, YVEL])
    max_p_test[0] = np.max(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, PRES])

    a, b, c = omega
    M = np.array([[0., -c, b], [c, 0, -a], [-b, a, 0]])
    # M = lamb * np.eye(3)
    q0 = np.array([u0, v0, p0])
    q_1D_approx = q0

    print("Début des calculs : ")
    while ( finalTime > currentTime - 1.e-14  ):

        if ( abs(currentTime + dt - finalTime * taux_progression) < 0.5 * dt ):
            print(" - Progression du calcul : " + str( 100 * round(taux_progression, 3) )+"%")
            taux_progression += 0.1
        if currentTime + dt > finalTime :
            dt = finalTime - currentTime
            currentTime = finalTime
            print(" - Temps final dépassé, on raccourcit le pas de temps")
        k += 1

        q_approx = time_operators.getDecUpdate(q_approx, dt, order, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs)
        # q_1D_approx = getHeunUpdate(q_1D_approx, M, dt)
        # q_approx = np.full_like(q_approx, q_1D_approx)

        max_u_test[k] = np.max(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, XVEL])
        max_v_test[k] = np.max(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, YVEL])
        max_p_test[k] = np.max(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, PRES])

        u_test_ecart[k] = max_u_test[k] - np.min(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, XVEL])
        v_test_ecart[k] = max_v_test[k] - np.min(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, YVEL])
        p_test_ecart[k] = max_p_test[k] - np.min(q_approx[iMin:iMax+1, jMin:jMax+1, :, :, PRES])


        currentTime += dt

    iterList = np.arange(k+1)
    timesList = np.linspace(0, finalTime, k+1)
    q_exact = getTestSolution(np.array([u0, v0, p0]), params, timesList)

    if testChoice == 1 :
        plotTitle = f"Comparaisons solutions approchées et exact de $\\frac{{d}}{{dt}}q + \\lambda q = 0$" \
                  f"\n avec $\\lambda$ = {round(lamb, 3)} et dt = {round(dt, 4)}"
    elif testChoice == 2 :
        plotTitle = f"Comparaisons solutions approchées et exact de $\\frac{{d}}{{dt}}q + \\omega \\times q = 0$" \
                  f"\n avec $\\omega$ = [{omega[0]:>.3f}, {omega[1]:>.3f}, {omega[2]:>.3f}] et dt = {round(dt, 4)}"
        plotTitle = f"Comparaisons solutions approchées et exact de $\\frac{{d}}{{dt}}q + \\omega \\times q = 0$" \
                  f"\n avec $\\omega$ = [{round(omega[0], 3)}, {round(omega[1], 3)}, {round(omega[2], 3)}] et dt = {round(dt, 4)}"

    # Comparaison avec solution de référence
    with PdfPages(f"Plots/DeC/Test_Evol_DeC_{order}") as pdf:
        # Evolutions des max :
        plt.figure()
        plt.title(plotTitle)
        plt.plot(iterList, max_u_test[:k+1], label="u approchée", marker="^", markevery=k // 10)
        plt.plot(iterList, q_exact[:, XVEL], label="u exacte")
        plt.plot(iterList, max_v_test[:k+1], label="v approchée", marker="v", markevery=k // 10)
        plt.plot(iterList, q_exact[:, YVEL], label="v exacte")
        plt.plot(iterList, max_p_test[:k+1], label="p approchée", marker="o", markevery=k // 10)
        plt.plot(iterList, q_exact[:, PRES], label="p exacte")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()
        plt.close()

        # Evolutions des ecarts :
        plt.figure()
        plt.title(f"Ecarts en espace")
        plt.plot(iterList[1:], u_test_ecart[1:k+1], label="u")
        plt.plot(iterList[1:], u_test_ecart[1:k+1], label="v")
        plt.plot(iterList[1:], p_test_ecart[1:k+1], label="p")
        plt.legend()
        pdf.savefig()
        plt.close()



####################################################################################################################

#                                                  CONVERGENCE TEST                                                #

####################################################################################################################



# CONVERGENCE TEST : END TIME ERRORS WITH VARRYING dt
if (True) :
# if (False) :
    Nx, Ny = 20, 20
    nGhost = 2
    order = 5

    testChoice = 2

    iMin, iMax, jMin, jMax = nGhost, Nx + nGhost - 1, nGhost, Ny + nGhost - 1

    # INITIAL DATA
    # Tirage d'un vecteur de norme entre norm_min et norm_max uniforme sur par rapport à l'angle
    norm_min, norm_max = 1., 3.
    norm_q0 = (norm_max - norm_min) * np.random.rand() + norm_min
    # tirage d'un vecteur gaussien
    drawn_variable = np.random.randn(3)
    # renormalisation
    u0, v0, p0 = norm_q0 * drawn_variable / np.linalg.norm(drawn_variable, ord=2)


    q_approx = np.full((Nx + 2*nGhost, Ny + 2*nGhost, order, order, 3), [u0, v0, p0])
    finalTime = 1.


    # Test choice :
    if testChoice == 1:
        # TIRAGE UNIFORME SUR [-LAMB_MAX, -LAMB_MIN] U [LAMB_MIN, LAMB_MAX]
        lamb_min, lamb_max = 0.1, 1.
        lamb = (2. * np.random.binomial(1, 0.5) - 1.) * ((lamb_max - lamb_min) * np.random.rand() + lamb_min)
        
        params = lamb
        getTestCoefficients = getTest1Coefficients
        getTestSolution = getTest1Solution
    elif testChoice == 2:
        # Tirage d'un vecteur de norme 2pi uniforme sur la sphère
        # tirage d'un vecteur gaussien
        drawn_variable = np.random.randn(3)
        # renormalisation
        omega = 2. * np.pi * drawn_variable / np.linalg.norm(drawn_variable, ord=2)

        params = omega
        getTestCoefficients = getTest2Coefficients
        getTestSolution = getTest2Solution
    
    evolCoeffs, massCoeffs = getTestCoefficients(params)

    # FINAL TIME SOLUTION
    finalTime = 1.
    q_exact = getTestSolution(q_approx, params, finalTime)


    # dtList = 10**(np.linspace(-1, -3, 5))

    dtList = 10**(np.linspace(-1, -2, 5))
    k = 0

    u_finalTime_errors, v_finalTime_errors, p_finalTime_errors = - np.ones((3, len(dtList)))

    for dt in dtList :

        print("Début des calculs pour dt = ", dt, " : ", flush=True)
        currentTime = 0.
        taux_progression = 0.2

        while ( abs(finalTime - currentTime) > 1.e-14  ):

            if currentTime + dt > finalTime :
                dt = finalTime - currentTime
                currentTime = finalTime
                print(" -  Temps final dépassé, on raccourcit le pas de temps \n \n", flush=True)
            else : 
                currentTime += dt
                if ( abs(currentTime + dt - finalTime * taux_progression) < 0.5 * dt ):
                    print(" -  Progression du calcul : " + str( 100 * round(taux_progression, 3) )+"%", flush=True)
                    taux_progression += 0.2
            

            q_approx = time_operators.getDecUpdate(q_approx, dt, order, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs)


        u_finalTime_errors[k] = np.max(abs(q_approx[:, :, :, :, XVEL] - q_exact[:, :, :, :, XVEL]))
        v_finalTime_errors[k] = np.max(abs(q_approx[:, :, :, :, YVEL] - q_exact[:, :, :, :, YVEL]))
        p_finalTime_errors[k] = np.max(abs(q_approx[:, :, :, :, PRES] - q_exact[:, :, :, :, PRES]))

        # PREPARING THE NEXT TURN : RESET q_approx values
        k += 1
        q_approx[:, :, :, :, XVEL] = u0
        q_approx[:, :, :, :, YVEL] = v0
        q_approx[:, :, :, :, PRES] = p0


    if testChoice == 1 :
        resultTitle = f"q' + lambda q = 0, avec lambda ~ {round(lamb, 3)}"
    elif testChoice == 2 :
        resultTitle = f"q' + omega x q = 0, avec omega ~ [{round(omega[0], 3)}, {round(omega[1], 3)}, {round(omega[2], 3)}]"


    with open(f"Plots/DeC/Test_Conv_DeC_{order}.txt", 'a') as f:
        f.write(f"=== Test de DeC ordre {order} sur l'EDO {resultTitle} et q0 ~ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}]  ===" + "\n" * 2)
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
        plotTitle = f"Erreurs sup à l'instant final en fonction de $\\Delta t$ pour $\\frac{{d}}{{dt}}q + \\lambda q = 0$" \
                    f"\n avec $\\lambda$ = {round(lamb, 3)} et $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}]"
    elif testChoice == 2 :
        plotTitle = f"Erreurs sup à l'instant final en fonction de $\\Delta t$ pour $\\frac{{d}}{{dt}}q + \\omega \\times q = 0$" \
                    f"\n avec $\\omega \\simeq$ [{round(omega[0], 3)}, {round(omega[1], 3)}, {round(omega[2], 3)}]" \
                    f" et $q_0 \\simeq$ [{round(u0, 3)}, {round(v0, 3)}, {round(p0, 3)}]"

    # Comparaison avec solution de référence
    with PdfPages(f"Plots/DeC/Test_Conv_DeC_{order}") as pdf:
        # Evolutions des max :
        plt.figure()
        plt.title(plotTitle)
        plt.plot(dtList, u_finalTime_errors, label="u", marker="^")
        plt.plot(dtList, v_finalTime_errors, label="v", marker="v")
        plt.plot(dtList, p_finalTime_errors, label="p", marker="o")
        plt.grid(alpha=0.5, ls="--")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("$\\Delta t$")
        plt.ylabel("$||q^N - q(T_f)||$")
        plt.legend()
        pdf.savefig()
        plt.close()




