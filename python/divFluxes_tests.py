# Modules généraux
import numpy as np
import yaml
import math
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Modules perso
import config
from config import XVEL, YVEL, PRES
import grid_mod
import schemes
from plots import makePlots, openGifWriters, closeGifWriters
import solutions
from schemes import getApproxDivFlux



####################################################################################################################

#                                                  TEST FUNCTIONS                                                  #

####################################################################################################################

def test1(X, Y): # f(x, y) = cos(2 kx pi x) + sin(2 ky pi y)
    kx = 2.
    ky = 5.

    # f
    func = np.cos(2. * kx * np.pi * X) + np.sin(2. * ky * np.pi * Y)
    # f_x
    func_x = - 2. * kx * np.pi * np.sin(2. * kx * np.pi * X)
    # f_y
    func_y = 2. * ky * np.pi * np.cos(2. * ky * np.pi * Y)
    # f_xx
    func_xx = - 4. * kx**2 * np.pi**2. * np.cos(2. * kx * np.pi * X)
    # f_yy
    func_yy = - 4. * ky**2 * np.pi**2. * np.sin(2. * ky * np.pi * Y)
    # f_xy
    func_xy = np.zeros((np.shape(X)))

    return func, func_x, func_y, func_xx, func_yy, func_xy


def test2(X, Y): # f(x, y) = a x^2 + b y^2 + c x + d y + e
    a, b, c, d, e = np.pi, np.exp(1), 13., np.sqrt(2.), 3.

    # f
    func = a * X**2 + b * Y**2 + c * X + d * Y + e
    # f_x
    func_x = 2. * a * X + c
    # f_y
    func_y = 2. * b * Y + d
    # f_xx
    func_xx = 2. * a * np.ones((np.shape(X)))
    # f_yy
    func_yy = 2. * b * np.ones((np.shape(Y)))
    # f_xy
    func_xy = np.zeros((np.shape(X)))

    return func, func_x, func_y, func_xx, func_yy, func_xy


def test3(X, Y): # f(x, y) = cos(2 kx pi x) * cos(2 ky pi y)
    kx = 2.
    ky = 5.

    # f
    func = np.cos(2. * kx * np.pi * X) * np.cos(2. * ky * np.pi * Y)
    # f_x
    func_x = - 2. * kx * np.pi * np.sin(2. * kx * np.pi * X) * np.cos(2. * ky * np.pi * Y)
    # f_y
    func_y = - 2. * ky * np.pi * np.cos(2. * kx * np.pi * X) * np.sin(2. * ky * np.pi * Y)
    # f_xx
    func_xx = - 4. * kx**2 * np.pi**2. * np.cos(2. * kx * np.pi * X) * np.cos(2. * ky * np.pi * Y)
    # f_yy
    func_yy = - 4. * ky**2 * np.pi**2. * np.cos(2. * kx * np.pi * X) * np.cos(2. * ky * np.pi * Y)
    # f_xy
    func_xy = 4. * kx * ky * np.pi**2 * np.sin(2. * kx * np.pi * X) * np.sin(2. * ky * np.pi * Y)

    return func, func_x, func_y, func_xx, func_yy, func_xy


def test4(X, Y): # f(x, y) = a x^2 + b y^2 + c x*y + d x + e y + f
    a, b, c, d, e, f = np.pi, -np.exp(1), -13., np.sqrt(2.), 3., -7./3.

    # f
    func = a * X**2 + b * Y**2 + c * X * Y + d * X + e * Y + f
    # f_x
    func_x = 2. * a * X + c * Y + d
    # f_y
    func_y = 2. * b * Y + c * X + e
    # f_xx
    func_xx = 2. * a * np.ones((np.shape(X)))
    # f_yy
    func_yy = 2. * b * np.ones((np.shape(Y)))
    # f_xy
    func_xy = c * np.ones((np.shape(X)))

    return func, func_x, func_y, func_xx, func_yy, func_xy

####################################################################################################################

#                                TESTING FUNCTIONS FOR EACH INDIVIDUAL COMPONENT                                   #

####################################################################################################################

# U has x-first-derivative on the last component and x-second-derivative on the first component

def first_component_operations_errors(q, divFlux_method, x_der, x_second_der, xy_der, grid, schemeChoice, operators):
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx = grid.steps[0]

    divFlux = divFlux_method(q, grid, schemeChoice, operators)
    print("test periodicité divFlux", grid.isPeriodic(divFlux))
    print("ecart divFlux bandes horizontales : ", np.max(divFlux[iMin, jMin:jMax+1] - divFlux[iMax, jMin:jMax+1]))
    print("ecart divFlux bandes verticales : ", np.max(divFlux[iMin:iMax+1, jMin] - divFlux[iMin:iMax+1, jMax]))

    first_component_error = np.max(abs(- 2. / dx * divFlux[iMin:iMax+1, jMin:jMax+1, 0] - x_second_der[iMin:iMax+1, jMin:jMax+1]))

    second_component_error = np.max(abs(- 2. / dx * divFlux[iMin:iMax+1, jMin:jMax+1, 1] - xy_der[iMin:iMax+1, jMin:jMax+1]))

    third_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, 2] - x_der[iMin:iMax+1, jMin:jMax+1]))

    return first_component_error, second_component_error, third_component_error



def second_component_operations_errors(q, divFlux_method, y_der, y_second_der, xy_der, grid, schemeChoice, operators):
    iMin, iMax, jMin, jMax = grid.valid_grid
    dy = grid.steps[1]

    divFlux = divFlux_method(q, grid, schemeChoice, operators)

    first_component_error = np.max(abs(- 2. / dy * divFlux[iMin:iMax+1, jMin:jMax+1, 0] - xy_der[iMin:iMax+1, jMin:jMax+1]))

    second_component_error = np.max(abs(- 2. / dy * divFlux[iMin:iMax+1, jMin:jMax+1, 1] - y_second_der[iMin:iMax+1, jMin:jMax+1]))

    third_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, 2] - y_der[iMin:iMax+1, jMin:jMax+1]))

    return first_component_error, second_component_error, third_component_error



def third_component_operations_errors(q, divFlux_method, x_der, y_der, x_second_der, y_second_der, grid, schemeChoice, operators):
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx, dy = grid.steps

    divFlux = divFlux_method(q, grid, schemeChoice, operators)

    first_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, 0] - x_der[iMin:iMax+1, jMin:jMax+1]))

    second_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, 1] - y_der[iMin:iMax+1, jMin:jMax+1]))

    third_component_error = np.max(abs( (- 2. * divFlux[iMin:iMax+1, jMin:jMax+1, 2] \
                                         - dx * x_second_der[iMin:iMax+1, jMin:jMax+1] \
                                         - dy * y_second_der[iMin:iMax+1, jMin:jMax+1]) / max(dx, dy) ))

    return first_component_error, second_component_error, third_component_error



####################################################################################################################

#                                               INDIVIDUAL TESTS                                                   #

####################################################################################################################



# if (True) :
if (False) :
    # Creation of the grid_parameters dictionnary by hand
    xL, xR = 0., 1.
    yL, yR = 0., 1.
    Nx, Ny = 401, 401
    nGhost = 1
    schemeChoice = 1
    operators = yaml.load(open("operators.yaml"),Loader=yaml.SafeLoader)

    grid_params = {"domain_parameters" : {"xL" : xL, "xR" : xR, "yL" : yL, "yR" : yR}, "mesh_parameters" : {"Nx" : Nx, "Ny" : Ny, "nGhost" : nGhost}}

    gridOp = grid_mod.gridOperator(grid_params)
    iMin, iMax, jMin, jMax = gridOp.valid_grid
    dx, dy = gridOp.steps


    X, Y = gridOp.xGrid, gridOp.yGrid
    f, x_der, x_second_der, y_der, y_second_der = test1(X, Y)
    print("test periodicité donnée : ", gridOp.isPeriodic(f))

    # print(X[iMin:iMax+1, jMin])


    # # PLOTTING TEST FUNCTION
    # fig, ax = plt.subplots()
    # cf = ax.contourf(X, Y, f[iMin:iMax+1, jMin:jMax+1], levels=50, cmap="plasma")
    # ax.contour(X, Y, f[iMin:iMax+1, jMin:jMax+1], levels=50, colors="k", linewidths=0.3)
    # plt.colorbar(cf, ax=ax, label="fonction test", format="%.2f")
    # ax.set_title("f_test")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_aspect("equal")
    # plt.tight_layout()
    # plt.savefig("Plots/fonc_test.pdf")

    q1 = np.zeros((iMax+nGhost+1, jMax+nGhost+1, 3))
    q1[:, :, 0] = f
    q2 = np.zeros((iMax+nGhost+1, jMax+nGhost+1, 3))
    q2[:, :, 1] = f
    q3 = np.zeros((iMax+nGhost+1, jMax+nGhost+1, 3))
    q3[:, :, 2] = f

    # FIRST COMPONENT OPERATIONS :
    divFlux_method = getApproxDivFlux
    Error_q1_1, Error_q1_2, Error_q1_3 = first_component_operations_errors(q1, divFlux_method, x_der, x_second_der, gridOp, schemeChoice, operators)
    Error_q2_1, Error_q2_2, Error_q2_3 = second_component_operations_errors(q2, divFlux_method, y_der, y_second_der, gridOp, schemeChoice, operators)
    Error_q3_1, Error_q3_2, Error_q3_3 = third_component_operations_errors(q3, divFlux_method, x_der, x_second_der, y_der, y_second_der, gridOp, schemeChoice, operators)


    print("Erreurs opérations sur 1e compsante : ", Error_q1_1, Error_q1_2, Error_q1_3)
    print("Erreurs opérations sur 2nd compsante : ", Error_q2_1, Error_q2_2, Error_q2_3)
    print("Erreurs opérations sur 3e compsante : ", Error_q3_1, Error_q3_2, Error_q3_3)

    divFlux = np.zeros((iMax+nGhost+1, jMax+nGhost+1, 3))
    divFlux = divFlux_method(q1, gridOp, schemeChoice, operators)

    # Dérivée simple
    plt.figure()
    plt.plot(X[iMin:iMax+1, 0], x_der[iMin:iMax+1, 0], label="Analytical derivative")
    plt.plot(X[iMin:iMax+1, 0], divFlux[iMin:iMax+1, jMin, 2], label="divFlux's derivative")
    plt.legend()
    plt.savefig("Plots/test_deriv.pdf")

    plt.figure()
    plt.plot(X[iMin:iMax+1, 0], abs(x_der[iMin:iMax+1, 0] - divFlux[iMin:iMax+1, jMin, 2]))
    plt.legend()
    plt.savefig("Plots/test_deriv_ecart")


    # Dérivée seconde
    plt.figure()
    plt.plot(X[iMin:iMax+1, 0], x_second_der[iMin:iMax+1, 0], label="Analytical derivative")
    plt.plot(X[iMin:iMax+1, 0], - 2. * divFlux[iMin:iMax+1, jMin, 0] / dx, label="divFlux's second derivative")
    plt.legend()
    plt.savefig("Plots/test_second_deriv.pdf")

    plt.figure()
    plt.plot(X[iMin:iMax+1, 0], abs(x_second_der[iMin:iMax+1, 0] + 2. * divFlux[iMin:iMax+1, jMin, 0] / dx))
    plt.legend()
    plt.savefig("Plots/test_second_deriv_ecart")



####################################################################################################################

#                                              CONVERGENCE TESTS                                                   #

####################################################################################################################



if (True) :
# if (False) :
    # Creation of the grid_parameters dictionnary by hand
    xL, xR = 0., 1.
    yL, yR = 0., 1.
    nGhost = 1
    grid_params = {"domain_parameters" : {"xL" : xL, "xR" : xR, "yL" : yL, "yR" : yR}, "mesh_parameters" : {"Nx" : 0, "Ny" : 0, "nGhost" : nGhost}}
    schemeChoice = 2
    operators = yaml.load(open("operators.yaml"),Loader=yaml.SafeLoader)
    divFlux_method = getApproxDivFlux
    
    
    nList = [101, 201, 401, 801, 1601]
    hList = np.ones(len(nList))
    Error_q1_1 = np.ones(len(nList))
    Error_q1_2 = np.ones(len(nList))
    Error_q1_3 = np.ones(len(nList))
    Error_q2_1 = np.ones(len(nList))
    Error_q2_2 = np.ones(len(nList))
    Error_q2_3 = np.ones(len(nList))
    Error_q3_1 = np.ones(len(nList))
    Error_q3_2 = np.ones(len(nList))
    Error_q3_3 = np.ones(len(nList))

    k = 0
    for n in nList :

        grid_params["mesh_parameters"]["Nx"] = n
        grid_params["mesh_parameters"]["Ny"] = n
        gridOp = grid_mod.gridOperator(grid_params)
        iMin, iMax, jMin, jMax = gridOp.valid_grid
        dx, dy = gridOp.steps

        hList[k] = dx


        X, Y = gridOp.xGrid, gridOp.yGrid
        f, x_der, y_der, x_second_der, y_second_der, xy_der = test3(X, Y)
        print("test periodicité donnée : ", gridOp.isPeriodic(f))

        if (schemeChoice == 1) :
            xy_der[:, :] = 0.

        q1 = np.zeros((iMax+nGhost+1, jMax+nGhost+1, 3))
        q1[:, :, 0] = f
        q2 = np.zeros((iMax+nGhost+1, jMax+nGhost+1, 3))
        q2[:, :, 1] = f
        q3 = np.zeros((iMax+nGhost+1, jMax+nGhost+1, 3))
        q3[:, :, 2] = f

        # FIRST COMPONENT OPERATIONS :
        Error_q1_1[k], Error_q1_2[k], Error_q1_3[k] = \
            first_component_operations_errors(q1, divFlux_method, x_der, x_second_der, xy_der, gridOp, schemeChoice, operators)
        Error_q2_1[k], Error_q2_2[k], Error_q2_3[k] = \
            second_component_operations_errors(q2, divFlux_method, y_der, y_second_der, xy_der, gridOp, schemeChoice, operators)
        Error_q3_1[k], Error_q3_2[k], Error_q3_3[k] = \
            third_component_operations_errors(q3, divFlux_method, x_der, y_der, x_second_der, y_second_der, gridOp, schemeChoice, operators)

        print("")
        print("nombre de points par côté : ", n)
        print("Erreurs opérations sur 1e compsante : ", Error_q1_1[k], Error_q1_2[k], Error_q1_3[k])
        print("Erreurs opérations sur 2nd compsante : ", Error_q2_1[k], Error_q2_2[k], Error_q2_3[k])
        print("Erreurs opérations sur 3e compsante : ", Error_q3_1[k], Error_q3_2[k], Error_q3_3[k])
        print("")

        # Plots opérations 1e composante :
        if (True) :
        # if (False) :
            divFlux = np.zeros((iMax+nGhost+1, jMax+nGhost+1, 3))
            divFlux = divFlux_method(q1, gridOp, schemeChoice, operators)

            # Dérivée simple
            plt.figure()
            plt.title("Ecart entre dérivée simple et approchée, de max "+str(Error_q1_3[k]))
            plt.plot(X[iMin:iMax+1, 0], abs(x_der[iMin:iMax+1, 0] - divFlux[iMin:iMax+1, jMin, 2]))
            plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            plt.legend()
            plt.savefig("Plots/Ecarts/test_deriv_ecart"+str(n))


            # Dérivée seconde
            plt.figure()
            plt.title("Ecart entre dérivée seconde et approchée, de max "+str(Error_q1_1[k]))
            plt.plot(X[iMin:iMax+1, 0], abs(x_second_der[iMin:iMax+1, 0] + 2. * divFlux[iMin:iMax+1, jMin, 0] / dx))
            plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            plt.legend()
            plt.savefig("Plots/Ecarts/test_second_deriv_ecart"+str(n))

        k += 1

    cross_term_u = "$\partial_x \partial_y v$"
    cross_term_v = "$\partial_y \partial_x u$"
    if (schemeChoice == 1):
        cross_term_u = "0"
        cross_term_v = "0"
        schemeName = "Upwind"
        schemeShort = "UPW"
    elif (schemeChoice == 2):
        schemeName = "SUPG"
        schemeShort = "SUPG"
    elif (schemeChoice == 3):
        schemeName = "modified SUPG"
        schemeShort = "modSUPG"
    


    

    # Résultats des opérations sur la première composante
    plt.figure()
    plt.title("Erreurs en norme $L^\infty$ sur les différentes composantes de $\widetilde{\\nabla} f$ \n pour le schéma "+schemeName+" avec v = p = 0")
    plt.plot(hList, Error_q1_1, label="$\partial^2_x u$", marker="o")
    plt.plot(hList, Error_q1_2, label=cross_term_u, marker="^")
    plt.plot(hList, Error_q1_3, label="$\partial_x u$", marker="s")
    plt.plot(hList, ( max(Error_q1_1[0], Error_q1_2[0], Error_q1_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
    plt.plot(hList, ( max(Error_q1_1[0], Error_q1_2[0], Error_q1_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("Plots/test_operators_1_"+schemeShort)

    # Résultats des opérations sur la seonde composante
    plt.figure()
    plt.title("Erreurs en norme $L^\infty$ sur les différentes composantes de $\widetilde{\\nabla} f$ \n pour le schéma "+schemeName+" avec u = p = 0")
    plt.plot(hList, Error_q2_1, label=cross_term_v, marker="o")
    plt.plot(hList, Error_q2_2, label="$\partial^2_y v$", marker="^")
    plt.plot(hList, Error_q2_3, label="$\partial_y v$", marker="s")
    plt.plot(hList, ( max(Error_q2_1[0], Error_q2_2[0], Error_q2_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
    plt.plot(hList, ( max(Error_q2_1[0], Error_q2_2[0], Error_q2_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("Plots/test_operators_2_"+schemeShort)

    # Résultats des opérations sur la première composante
    plt.figure()
    plt.title("Erreurs en norme $L^\infty$ sur les différentes composantes de $\widetilde{\\nabla} f$ \n pour le schéma "+schemeName+" avec u = v = 0")
    plt.plot(hList, Error_q3_1, label="$\partial_x p$", marker="o")
    plt.plot(hList, Error_q3_2, label="$\partial_y p$", marker="^")
    plt.plot(hList, Error_q3_3, label="$\Delta p$", marker="s")
    plt.plot(hList, ( max(Error_q3_1[0], Error_q3_2[0], Error_q3_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
    plt.plot(hList, ( max(Error_q3_1[0], Error_q3_2[0], Error_q3_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("Plots/test_operators_3_"+schemeShort)
