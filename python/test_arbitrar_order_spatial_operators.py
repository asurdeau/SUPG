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
import solutions
from spatial_operators import get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator, get_arbitrar_order_SUPG_weighted_evolution_operator_coeffs, \
                                get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator, get_arbitrar_order_SUPG_weighted_mass_operator_coeffs, \
                                get_arbitrar_order_SUPG_evolution_operator_coeffs
from plots import getVisualisationData



####################################################################################################################

#                                                  TEST FUNCTIONS                                                  #

####################################################################################################################

def test1(X, Y, params): # f(x, y) = cos(2 kx pi x) + sin(2 ky pi y)
    kx, ky = params

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


def test2(X, Y, params): # f(x, y) = a x^2 + b y^2 + c x + d y + e
    a, b, c, d, e = params

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


def test3(X, Y, params): # f(x, y) = cos(2 kx pi x) * cos(2 ky pi y)
    kx, ky = params

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


def test4(X, Y, params): # f(x, y) = a x^2 + b y^2 + c x*y + d x + e y + f
    a, b, c, d, e, f = params

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


def test5(X, Y, params): # f(x, y) = sin(2 kx pi x) * sin(2 ky pi y)
    kx, ky = params

    # f
    func = np.sin(2. * kx * np.pi * X) * np.sin(2. * ky * np.pi * Y)
    # f_x
    func_x = 2. * kx * np.pi * np.cos(2. * kx * np.pi * X) * np.sin(2. * ky * np.pi * Y)
    # f_y
    func_y = 2. * ky * np.pi * np.sin(2. * kx * np.pi * X) * np.cos(2. * ky * np.pi * Y)
    # f_xx
    func_xx = - 4. * kx**2 * np.pi**2. * np.sin(2. * kx * np.pi * X) * np.sin(2. * ky * np.pi * Y)
    # f_yy
    func_yy = - 4. * ky**2 * np.pi**2. * np.sin(2. * kx * np.pi * X) * np.sin(2. * ky * np.pi * Y)
    # f_xy
    func_xy = 4. * kx * ky * np.pi**2 * np.cos(2. * kx * np.pi * X) * np.cos(2. * ky * np.pi * Y)

    return func, func_x, func_y, func_xx, func_yy, func_xy


def test6(X, Y, params): # x^a . (1-x)^b . y^c . (1-y)^d
    a, b, c, d = params

    # f
    func = X**a*(1.-X)**b * Y**c*(1-Y)**d
    # f_x
    func_x = (a*X**(a-1)*(1-X)**b - b*X**a*(1-X)**(b-1)) * Y**c * (1-Y)**d
    # f_y
    func_y = X**a*(1.-X)**b * (c*Y**(c-1)*(1-Y)**d - d*Y**c*(1-Y)**(d-1))
    # f_xx
    func_xx = (a*(a-1)*X**(max(a-2, 0))*(1-X)**b - 2.*a*b*X**(a-1)*(1-X)**(b-1) + b*(b-1.)*X**a*(1-X)**(max(b-2, 0))) * Y**c * (1-Y)**d
    # f_yy
    func_yy = (X**a*(1.-X)**b) * (c*(c-1.)*Y**(max(c-2, 0))*(1-Y)**d - 2.*c*d*Y**(c-1.)*(1-Y)**(d-1) + d*(d-1.)*Y**c*(1-Y)**(max(d-2, 0)))
    # f_xy
    func_xy = (a*X**(a-1)*(1-X)**b - b*X**a*(1-X)**(b-1)) * (c*Y**(c-1)*(1-Y)**d - d*Y**c*(1-Y)**(d-1))

    return func, func_x, func_y, func_xx, func_yy, func_xy



def getTestFonctionLagrangeArbitrarOrderInterpolation(xL, xR, yL, yR, Nx, Ny, nGhost, spacePolynOrder, testFonction, params):
    dx, dy = (xR - xL) / (1. * Nx), (yR - yL) / (1. * Ny)

    fTab = np.zeros((6, Nx + 2 * nGhost, Ny + 2 * nGhost, spacePolynOrder, spacePolynOrder))
    iMin, jMin = nGhost, nGhost

    for i in range(Nx + 2 * nGhost):
        for j in range(Ny + 2 * nGhost):
            for k in range(spacePolynOrder):
                for l in range(spacePolynOrder):
                    x_ik = xL + (i - iMin + (1. * k) / spacePolynOrder) * dx
                    y_jl = yL + (j - jMin + (1. * l) / spacePolynOrder) * dy
                    fTab[:, i, j, k, l] = testFonction(x_ik, y_jl, params)

    return fTab

####################################################################################################################

#                                TESTING FUNCTIONS FOR EACH INDIVIDUAL COMPONENT                                   #

####################################################################################################################

# divF OPERATOR
# U has x-first-derivative on the last component and x-second-derivative on the first component

def evol_operator_first_component_operations_errors(q, iMin, iMax, jMin, jMax, dx, dy, f_x, f_xx, f_xy, \
                                      grid, spacePolynOrder, evolCoeffs):
    
    iMin, iMax, jMin, jMax = grid.validGridIndices
    dx = grid.steps[0]

    divFlux = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q, iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs)
    # print("test periodicité divFlux", grid.isPeriodic(divFlux))
    # print("ecart divFlux bandes horizontales : ", np.max(divFlux[iMin, jMin:jMax+1] - divFlux[iMax, jMin:jMax+1]))
    # print("ecart divFlux bandes verticales : ", np.max(divFlux[iMin:iMax+1, jMin] - divFlux[iMin:iMax+1, jMax]))

    first_component_error = np.max(abs(- 2. * divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 0] / dx - f_xx[iMin:iMax+1, jMin:jMax+1, :, :]))

    second_component_error = np.max(abs(- 2. * divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 1] / dy - f_xy[iMin:iMax+1, jMin:jMax+1, :, :]))

    third_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 2] - f_x[iMin:iMax+1, jMin:jMax+1, :, :]))

    return first_component_error, second_component_error, third_component_error



def evol_operator_second_component_operations_errors(q, iMin, iMax, jMin, jMax, dx, dy, f_y, f_yy, f_xy, \
                                       grid, spacePolynOrder, evolCoeffs):
    iMin, iMax, jMin, jMax = grid.validGridIndices
    dy = grid.steps[1]

    divFlux = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q, iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs)

    first_component_error = np.max(abs(- 2. * divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 0] / dx - f_xy[iMin:iMax+1, jMin:jMax+1, :, :]))

    second_component_error = np.max(abs(- 2. * divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 1] / dy - f_yy[iMin:iMax+1, jMin:jMax+1, :, :]))

    third_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 2] - f_y[iMin:iMax+1, jMin:jMax+1, :, :]))

    return first_component_error, second_component_error, third_component_error



def evol_operator_third_component_operations_errors(q, iMin, iMax, jMin, jMax, dx, dy, f_x, f_y, f_xx, f_yy, \
                                      grid, spacePolynOrder, evolCoeffs):
    iMin, iMax, jMin, jMax = grid.validGridIndices
    dx, dy = grid.steps

    divFlux = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q, iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs)

    first_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 0] - f_x[iMin:iMax+1, jMin:jMax+1, :, :]))

    second_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 1] - f_y[iMin:iMax+1, jMin:jMax+1, :, :]))

    third_component_error = np.max(abs( (- 2. * divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 2] / max(dx, dy) \
                                         - f_xx[iMin:iMax+1, jMin:jMax+1, :, :] \
                                         - f_yy[iMin:iMax+1, jMin:jMax+1, :, :]) ))

    return first_component_error, second_component_error, third_component_error




# MASS OPERATOR
# U has x-first-derivative on the last component and x-second-derivative on the first component

def mass_operator_first_component_operations_errors(q, iMin, iMax, jMin, jMax, dx, dy, f, f_x, grid, spacePolynOrder, massCoeffs):
    
    iMin, iMax, jMin, jMax = grid.validGridIndices
    dx = grid.steps[0]

    mass = get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator(q, iMin, iMax, jMin, jMax, spacePolynOrder, massCoeffs)
    # print("test periodicité divFlux", grid.isPeriodic(divFlux))
    # print("ecart divFlux bandes horizontales : ", np.max(divFlux[iMin, jMin:jMax+1] - divFlux[iMax, jMin:jMax+1]))
    # print("ecart divFlux bandes verticales : ", np.max(divFlux[iMin:iMax+1, jMin] - divFlux[iMin:iMax+1, jMax]))

    first_component_error = np.max(abs(mass[iMin:iMax+1, jMin:jMax+1, :, :, 0] - f[iMin:iMax+1, jMin:jMax+1, :, :]))

    second_component_error = np.max(abs(mass[iMin:iMax+1, jMin:jMax+1, :, :, 1]))

    third_component_error = np.max(abs(-2. * mass[iMin:iMax+1, jMin:jMax+1, :, :, 2] / dx - f_x[iMin:iMax+1, jMin:jMax+1, :, :]))

    return first_component_error, second_component_error, third_component_error



def mass_operator_second_component_operations_errors(q, iMin, iMax, jMin, jMax, dx, dy, f, f_y, grid, spacePolynOrder, massCoeffs):
    iMin, iMax, jMin, jMax = grid.validGridIndices
    dy = grid.steps[1]

    mass = get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator(q, iMin, iMax, jMin, jMax, spacePolynOrder, massCoeffs)

    first_component_error = np.max(abs(mass[iMin:iMax+1, jMin:jMax+1, :, :, 0]))

    second_component_error = np.max(abs(mass[iMin:iMax+1, jMin:jMax+1, :, :, 1] - f[iMin:iMax+1, jMin:jMax+1, :, :]))

    third_component_error = np.max(abs(-2. * mass[iMin:iMax+1, jMin:jMax+1, :, :, 2] / dy - f_y[iMin:iMax+1, jMin:jMax+1, :, :]))

    return first_component_error, second_component_error, third_component_error



def mass_operator_third_component_operations_errors(q, iMin, iMax, jMin, jMax, dx, dy, f, f_x, f_y, grid, spacePolynOrder, massCoeffs):
    iMin, iMax, jMin, jMax = grid.validGridIndices
    dx, dy = grid.steps

    mass = get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator(q, iMin, iMax, jMin, jMax, spacePolynOrder, massCoeffs)

    first_component_error = np.max(abs(- 2. * mass[iMin:iMax+1, jMin:jMax+1, :, :, 0] / dx - f_x[iMin:iMax+1, jMin:jMax+1, :, :]))

    second_component_error = np.max(abs(- 2. * mass[iMin:iMax+1, jMin:jMax+1, :, :, 1] / dy - f_y[iMin:iMax+1, jMin:jMax+1, :, :]))

    third_component_error = np.max(abs( (mass[iMin:iMax+1, jMin:jMax+1, :, :, 2] - f[iMin:iMax+1, jMin:jMax+1, :, :]) ))

    return first_component_error, second_component_error, third_component_error




####################################################################################################################

#                                            COMPARAISON AVEC L'ORDRE 1                                            #

####################################################################################################################





# if (True) :
if (False) :
    xL, xR, yL, yR = 0., 1., 0., 1.
    Nx, Ny = 100, 100
    nGhost = 2
    spacePolynOrder = 1
    alpha = 0.5

    iMin, iMax, jMin, jMax = nGhost, Nx + nGhost - 1, nGhost, Ny + nGhost - 1
    dx, dy = (xR - xL) / (1. * Nx), (yR - yL) / (1. * Ny)
    h = max(dx, dy)

    coeffs = spatial_operators.extract_operators_improved(dx, dy)
    basisChoice = "basic"
    modificationChoice = ""
    evolCoeffs = spatial_operators.get_arbitrar_order_SUPG_weighted_evolution_operator_coeffs(basisChoice, modificationChoice, spacePolynOrder, dx, dy, alpha, h)

    n1, n2 = -3, 3
    rng = np.random.default_rng()
    random_abs = np.exp(- ((n2 - n1) * rng.random(size=((Nx + 2*nGhost, Ny + 2* nGhost, 3))) + n1) * np.log(10.) )
    random_sign = 2. * rng.binomial(n=1, p=0.5, size=(Nx + 2*nGhost, Ny + 2* nGhost, 3)) - 1.

    q_test = random_abs * random_sign

    # var = XVEL
    # q_order_0 = np.zeros((Nx + 2 * nGhost, Ny + 2* nGhost, 3))
    # q_order_0[:, :, var] = q_test[:, :, var].copy()
    # q_order_1 = np.zeros((Nx + 2 * nGhost, Ny + 2* nGhost, spacePolynOrder, spacePolynOrder, 3))
    # q_order_1[:, :, 0, 0, var] = q_test[:, :, var].copy()


    q_order_0 = np.zeros((Nx + 2 * nGhost, Ny + 2* nGhost, 3))
    q_order_0 = q_test.copy()
    q_order_1 = np.zeros((Nx + 2 * nGhost, Ny + 2* nGhost, spacePolynOrder, spacePolynOrder, 3))
    q_order_1[:, :, 0, 0] = q_test.copy()

    print("test mêmes qttés de départ : ", np.max(abs(q_order_0 - q_order_1[:, :, 0, 0, :])))

    divF_order0 = spatial_operators.get_optim_modif_SUPG_developped_divFlux(q_order_0, iMin, iMax, jMin, jMax, coeffs)
    divF_order1 = spatial_operators.get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q_order_1, iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs)

    # comparaison entre les deux versions 
    print("comparaisons ordres 0 et 1 : 1e comp : ", np.max(abs(divF_order0[iMin:iMax+1, jMin:jMax+1, XVEL] - divF_order1[iMin:iMax+1, jMin:jMax+1, 0, 0, XVEL])) )
    print("comparaisons ordres 0 et 1 : 2e comp : ", np.max(abs(divF_order0[iMin:iMax+1, jMin:jMax+1, YVEL] - divF_order1[iMin:iMax+1, jMin:jMax+1, 0, 0, YVEL])) )
    print("comparaisons ordres 0 et 1 : 3e comp : ", np.max(abs(divF_order0[iMin:iMax+1, jMin:jMax+1, PRES] - divF_order1[iMin:iMax+1, jMin:jMax+1, 0, 0, PRES])) )







####################################################################################################################

#                                     EVOL OPERATOR : INDIVIDUAL TESTS                                             #

####################################################################################################################



# if (True) :
if (False) :
    # Creation of the grid_parameters dictionnary by hand
    xL, xR = 0., 1.
    yL, yR = 0., 1.
    Nx, Ny = 100, 100
    nGhost = 2
    spacePolynOrder = 3
    NVis = spacePolynOrder
    iTest = 2
    alpha = 0.5


    grid_params = {"domain_parameters" : {"xL" : xL, "xR" : xR, "yL" : yL, "yR" : yR}, "mesh_parameters" : {"Nx" : Nx, "Ny" : Ny, "nGhost" : nGhost}}
    params = {"grid_parameters" : grid_params, "order" : {"space_polyn_order" : spacePolynOrder}, "number_visualisation_points" : NVis}

    gridOp = grid_mod.gridOperator(params)
    iMin, iMax, jMin, jMax = gridOp.validGridIndices
    dx, dy = gridOp.steps
    h = gridOp.characteristic_size
    basisChoice = "basic"
    modificationChoice = ""
    evolCoeffs = get_arbitrar_order_SUPG_weighted_evolution_operator_coeffs(basisChoice, modificationChoice, spacePolynOrder, dx, dy, alpha, h)

    Id = np.array([ np.zeros((spacePolynOrder, spacePolynOrder)), np.eye(spacePolynOrder), np.zeros((spacePolynOrder, spacePolynOrder)) ])

    if iTest == 1: ### TEST Identity :
        for i in range(20):
            evolCoeffs[i] = Id
    elif iTest == 2: ### TEST : x operators (y operators = Id)
        for i in range(10):
            evolCoeffs[2 * i + 1] = Id
    elif iTest == 3: ### TEST : y operators (x operators = Id)
        for i in range(10):
            evolCoeffs[2 * i] = Id
    


    # SLICES DE REFERENCE
    # INJECTION MAILLAGE CALCUL (x_i^k) DANS MAILLAGE RAFFINE (ksi_alpha^gamma) :
    # x_i^k = ksi_{i * n_raffinement + k * n_raffinement // ordre}^{k * n_raffinement % ordre}
    n_raffinement = 10

    ### SLICE HORIZONTAL
    NxRef = n_raffinement * Nx * spacePolynOrder
    Xref = np.linspace(xL, xR, NxRef+1)
    j_ref, l_ref = int(Ny / 2.), 0

    ### SLICE VERTICAL
    NyRef = n_raffinement * Ny * spacePolynOrder
    Yref = np.linspace(yL, yR, NyRef+1)
    i_ref, k_ref = int(Nx / 2.), 0

    vertical_index_ref_mesh = (j_ref * n_raffinement + (l_ref * n_raffinement) // spacePolynOrder) * spacePolynOrder + (l_ref * n_raffinement) % spacePolynOrder
    y_ref = Yref[vertical_index_ref_mesh]
    horizontal_index_ref_mesh = (i_ref * n_raffinement + (k_ref * n_raffinement) // spacePolynOrder) * spacePolynOrder + (k_ref * n_raffinement) % spacePolynOrder
    x_ref = Xref[horizontal_index_ref_mesh]

    XXref, YYref = np.meshgrid(Xref, Yref)
    XXref = np.transpose(XXref)
    YYref = np.transpose(YYref)

    paramRef = params


    testFonction = test6

    f_HO_mesh, f_x_HO_mesh, f_y_HO_mesh, f_xx_HO_mesh, f_yy_HO_mesh, f_xy_HO_mesh = getTestFonctionLagrangeArbitrarOrderInterpolation(xL, xR, yL, yR, Nx, Ny, nGhost, spacePolynOrder, testFonction)
    f_ref_mesh, f_x_ref_mesh, f_y_ref_mesh, f_xx_ref_mesh, f_yy_ref_mesh, f_xy_ref_mesh = \
        testFonction(XXref, YYref)

    # # test periodicité de f :
    # print("test périodicité de f : ")
    # print("écarts bords verticaux : ", np.max(abs(f_HO_mesh[iMin, jMin:jMax+1, 0, :] - f_HO_mesh[iMax+1, jMin:jMax+1, 0, :])))
    # print("écarts bords verticaux : ", np.max(abs(f_HO_mesh[iMin:iMax+1, jMin, :, 0] - f_HO_mesh[iMin:iMax+1, jMax+1, :, 0])), "\n")

    f_HO_mesh_flattened = getVisualisationData(f_HO_mesh, params)
    f_y_HO_mesh_flattened = getVisualisationData(f_y_HO_mesh, params)


    q1, q2, q3 = np.zeros((3, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder, spacePolynOrder, 3))
    q1[:, :, :, :, 0] = f_HO_mesh
    q2[:, :, :, :, 1] = f_HO_mesh
    q3[:, :, :, :, 2] = f_HO_mesh


    divFlux1 = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q1, iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs)
    divFlux2 = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q2, iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs)
    divFlux3 = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q3, iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs)

    # print("test périodicité de divFlux1 : ")
    # print("écarts bords verticaux : ", np.max(abs(divFlux1[iMin, jMin:jMax+1, 0, :] - divFlux1[iMax+1, jMin:jMax+1, 0, :])))
    # print("écarts bords verticaux : ", np.max(abs(divFlux1[iMin:iMax+1, jMin, :, 0] - divFlux1[iMin:iMax+1, jMax+1, :, 0])), "\n")

    # Mise à plat des données
    divFlux1_2D = getVisualisationData(divFlux1, params)
    divFlux2_2D = getVisualisationData(divFlux2, params)
    divFlux3_2D = getVisualisationData(divFlux3, params)


    f = f_ref_mesh[::n_raffinement, ::n_raffinement]
    f_x = f_x_ref_mesh[::n_raffinement, ::n_raffinement]
    f_xx = f_xx_ref_mesh[::n_raffinement, ::n_raffinement]
    f_y = f_y_ref_mesh[::n_raffinement, ::n_raffinement]
    f_yy = f_yy_ref_mesh[::n_raffinement, ::n_raffinement]
    f_xy = f_xy_ref_mesh[::n_raffinement, ::n_raffinement]


    # print("test projection de maillage pour f : ", np.max(abs(f - f_HO_mesh_flattened)))
    # print("test projection de maillage pour f_y : ", np.max(abs(f_y - f_y_HO_mesh_flattened)))

    var = [("u", XVEL), ("v", YVEL), ("p", PRES)]

    if iTest == 1: # Test avec identité : indices de stencil cohérents : toutes les sorties de divFlux sont f
        print("="*50, "\n"*2)
        print("Test 1 : tous les opérateurs sont l'identité", "\n"*2)
        print("="*50, "\n"*2)

        exact_operators_names = [["", "", ""], 
                                 ["", "", ""], \
                                 ["", "", ""]   ]
        exact_operators_ref = np.array([[f_ref_mesh, f_ref_mesh, f_ref_mesh], \
                                        [f_ref_mesh, f_ref_mesh, f_ref_mesh], \
                                        [f_ref_mesh, f_ref_mesh, f_ref_mesh]])
        exact_operators = np.array([[f, f, f], \
                                    [f, f, f], \
                                    [f, f, f]])
        
        divFluxes = np.array([[divFlux1_2D[:, :, XVEL], divFlux2_2D[:, :, XVEL], divFlux3_2D[:, :, XVEL]      ], \
                              [divFlux1_2D[:, :, YVEL], divFlux2_2D[:, :, YVEL], divFlux3_2D[:, :, YVEL]      ], \
                              [divFlux1_2D[:, :, PRES], divFlux2_2D[:, :, PRES], 0.5 * divFlux3_2D[:, :, PRES]]])


    if iTest == 2: # Test x opérateurs (y operateurs = Id)
        print("="*50, "\n"*2)
        print("Test 2 : tous les opérateurs en y sont l'identité", "\n"*2)
        print("="*50, "\n"*2)

        exact_operators_names = [["$\\partial_x^2$", "$\\partial_x$", "$\\partial_x$"], 
                                 ["$\\partial_x$", "", ""], \
                                 ["$\\partial_x$", "", "$(- \\Delta x \\partial_x^2 + Id)$"]   ]
        exact_operators_ref = np.array([[f_xx_ref_mesh, f_x_ref_mesh, f_x_ref_mesh], \
                                        [f_x_ref_mesh, f_ref_mesh, f_ref_mesh], \
                                        [f_x_ref_mesh, f_ref_mesh, - dx * f_xx_ref_mesh + f_ref_mesh]])
        exact_operators = np.array([[f_xx, f_x, f_x], \
                                    [f_x , f  , f  ], \
                                    [f_x , f  , f_xx - f / (alpha * h)]])
        
        divFluxes = np.array([[ - divFlux1_2D[:, :, XVEL] / (alpha * h), - divFlux2_2D[:, :, XVEL] / (alpha * h),   divFlux3_2D[:, :, XVEL]              ], \
                              [   divFlux1_2D[:, :, YVEL]              ,   divFlux2_2D[:, :, YVEL]              ,   divFlux3_2D[:, :, YVEL]              ], \
                              [   divFlux1_2D[:, :, PRES]              ,   divFlux2_2D[:, :, PRES]              , - divFlux3_2D[:, :, PRES] / (alpha * h)]])
            


    if iTest == 3: # Test y opérateurs (x operateurs = Id)
        print("="*50, "\n"*2)
        print("Test 3 : tous les opérateurs en x sont l'identité", "\n"*2)
        print("="*50, "\n"*2)

        exact_operators_names = [["", "$\\partial_y$", ""], 
                                 ["$\\partial_y$", "$\\partial^2_y$", "$\\partial_y$"], \
                                 ["", "$\\partial_y$", "$(- \\Delta y \\partial_y^2 + Id)$"]   ]
        exact_operators_ref = np.array([[f_ref_mesh, f_y_ref_mesh, f_ref_mesh], \
                                        [f_y_ref_mesh, f_yy_ref_mesh, f_y_ref_mesh], \
                                        [f_ref_mesh, f_y_ref_mesh, - dy * f_yy_ref_mesh + f_ref_mesh]])
        exact_operators = np.array([[f  , f_y , f       ], \
                                    [f_y, f_yy, f_y     ], \
                                    [f  , f_y , f_yy - f / (alpha * h)]])
        
        divFluxes = np.array([[   divFlux1_2D[:, :, XVEL]              ,   divFlux2_2D[:, :, XVEL]              ,   divFlux3_2D[:, :, XVEL]              ], \
                              [ - divFlux1_2D[:, :, YVEL] / (alpha * h), - divFlux2_2D[:, :, YVEL] / (alpha * h),   divFlux3_2D[:, :, YVEL]              ], \
                              [   divFlux1_2D[:, :, PRES]              ,   divFlux2_2D[:, :, PRES]              , - divFlux3_2D[:, :, PRES] / (alpha * h)]])


    elif iTest == 4:
        print("="*50, "\n"*2)
        print("Test 4 : tous les opérateurs sont laissés tels quels", "\n"*2)
        print("="*50, "\n"*2)

        exact_operators_names = [["$\\partial^2_x$", "$\\partial^2_{{xy}}$", "$\\partial_x$"], 
                                 ["$\\partial^2_{{xy}}$", "$\\partial^2_y$", "$\\partial_y$"], \
                                 ["$\\partial_x$", "$\\partial_y$", "$\\frac{{\\Delta x}}{{h}} \\partial_x^2 + \\frac{{\\Delta y}}{{h}} \\partial_y^2$"]   ]
        exact_operators_ref = np.array([[f_xx_ref_mesh, f_xy_ref_mesh, f_x_ref_mesh], \
                                        [f_xy_ref_mesh, f_yy_ref_mesh, f_y_ref_mesh], \
                                        [f_x_ref_mesh, f_y_ref_mesh, f_xx_ref_mesh + f_yy_ref_mesh]])
        exact_operators = np.array([[f_xx, f_xy, f_x], \
                                    [f_xy, f_yy, f_y], \
                                    [f_x , f_y , f_xx + f_yy]])
        
        divFluxes = np.array([[ - divFlux1_2D[:, :, XVEL] / (alpha * h), - divFlux2_2D[:, :, XVEL] / (alpha * h),   divFlux3_2D[:, :, XVEL]              ], \
                              [ - divFlux1_2D[:, :, YVEL] / (alpha * h), - divFlux2_2D[:, :, YVEL] / (alpha * h),   divFlux3_2D[:, :, YVEL]              ], \
                              [   divFlux1_2D[:, :, PRES]              ,   divFlux2_2D[:, :, PRES]              , - divFlux3_2D[:, :, PRES] / (alpha * h)]])



    # ECARTS 
    for equationName, equation in var:
        print(f"Opérations de l'équation sur {equationName} : ")
        for termName, term in var:
            print(f"   sur {termName} : ", np.max(abs(divFluxes[equation, term] - exact_operators[equation, term])))
        print("")


    # PLOTS
    if True :
    # if False :
        vertical_index_ref_mesh = (j_ref * n_raffinement + l_ref * n_raffinement // spacePolynOrder) * spacePolynOrder + (l_ref * n_raffinement) % spacePolynOrder
        y_ref = Yref[vertical_index_ref_mesh]
        horizontal_index_ref_mesh = (i_ref * n_raffinement + k_ref * n_raffinement // spacePolynOrder) * spacePolynOrder + (k_ref * n_raffinement) % spacePolynOrder
        x_ref = Xref[horizontal_index_ref_mesh]

        for equationName, equation in var:
            with PdfPages(f"Plots/divF/1D_slices_{equationName}_equation_order_{spacePolynOrder}") as pdf:
                for termName, term in var:
                    # exact operator 1D slices
                    exact_operator = exact_operators_ref[equation, term]
                    exact_operator_name = exact_operators_names[equation][term]
                    horizontal_slice = exact_operator[:, vertical_index_ref_mesh]
                    vertical_slice = exact_operator[horizontal_index_ref_mesh, :]

                    # approximate operator
                    approxOperator = divFluxes[equation, term]

                    # Slice horizontal
                    plt.figure()
                    plt.title(f"Slice horizontal [0, 1]x{{{y_ref}}}: {exact_operator_name}{termName} dans l'équation en {equationName}")
                    plt.plot(Xref, horizontal_slice, label="exact")
                    plt.plot(Xref[::n_raffinement], approxOperator[:, j_ref*spacePolynOrder + l_ref], label="approx")
                    plt.legend()
                    pdf.savefig()
                    plt.close()
        
                    # Slice vertical
                    plt.figure()
                    plt.title(f"Slice vertical {{{x_ref}}}x[0, 1]: {exact_operator_name}{termName} dans l'équation en {equationName}")
                    plt.plot(Yref, vertical_slice, label="exact")
                    plt.plot(Yref[::n_raffinement], approxOperator[i_ref*spacePolynOrder + k_ref, :], label="approx")
                    plt.legend()
                    pdf.savefig()
                    plt.close()




####################################################################################################################

#                                      EVOL OPERATOR : CONVERGENCE TESTS                                           #

####################################################################################################################



# if (True) :
if (False) :
    # Creation of the grid_parameters dictionnary by hand
    xL, xR = 0., 1.
    yL, yR = 0., 1.
    nGhost = 2
    spacePolynOrder = 2
    grid_params = {"domain_parameters" : {"xL" : xL, "xR" : xR, "yL" : yL, "yR" : yR}, "mesh_parameters" : {"Nx" : 0, "Ny" : 0, "nGhost" : nGhost}}
    params = {"grid_parameters" : grid_params, "order" : {"space_polyn_order" : spacePolynOrder}}
    alpha = 0.5
    
    
    nList = [100, 200, 400, 800]
    hList = np.ones(len(nList))
    Error_q1_1, Error_q1_2, Error_q1_3, Error_q2_1, Error_q2_2, Error_q2_3, \
        Error_q3_1, Error_q3_2, Error_q3_3 = np.ones((9, len(nList)))

    k = 0
    for n in nList :

        grid_params["mesh_parameters"]["Nx"] = n
        grid_params["mesh_parameters"]["Ny"] = n
        gridOp = grid_mod.gridOperator(params)
        iMin, iMax, jMin, jMax = gridOp.validGridIndices
        dx, dy = gridOp.steps
        h = gridOp.characteristic_size
        basisChoice = "basic"
        modificationChoice = "highest_accuracy"
        evolCoeffs = get_arbitrar_order_SUPG_weighted_evolution_operator_coeffs(basisChoice, modificationChoice, spacePolynOrder, dx, dy, alpha, h)

        # print("evolCoeffs : \n ", evolCoeffs)

        hList[k] = dx

        testFonction = test5

        f, f_x, f_y, f_xx, f_yy, f_xy = \
            getTestFonctionLagrangeArbitrarOrderInterpolation(xL, xR, yL, yR, n, n, nGhost, spacePolynOrder, testFonction)


        q1, q2, q3 = np.zeros((3, n + 2 * nGhost, n + 2 * nGhost, spacePolynOrder, spacePolynOrder, 3))
        q1[:, :, :, :, 0] = f
        q2[:, :, :, :, 1] = f
        q3[:, :, :, :, 2] = f

        # FIRST COMPONENT OPERATIONS :
        Error_q1_1[k], Error_q1_2[k], Error_q1_3[k] = \
            evol_operator_first_component_operations_errors(q1, iMin, iMax, jMin, jMax, dx, dy, f_x, f_xx, f_xy, \
                                                gridOp, spacePolynOrder, evolCoeffs)
        Error_q2_1[k], Error_q2_2[k], Error_q2_3[k] = \
            evol_operator_second_component_operations_errors(q2, iMin, iMax, jMin, jMax, dx, dy, f_y, f_yy, f_xy, \
                                               gridOp, spacePolynOrder, evolCoeffs)
        Error_q3_1[k], Error_q3_2[k], Error_q3_3[k] = \
            evol_operator_third_component_operations_errors(q3, iMin, iMax, jMin, jMax, dx, dy, f_x, f_y, f_xx, f_yy, \
                                              gridOp, spacePolynOrder, evolCoeffs)

        print("")
        print("nombre de points par côté : ", n)
        print("Erreurs opérations sur 1e compsante : ", Error_q1_1[k], Error_q1_2[k], Error_q1_3[k])
        print("Erreurs opérations sur 2nd compsante : ", Error_q2_1[k], Error_q2_2[k], Error_q2_3[k])
        print("Erreurs opérations sur 3e compsante : ", Error_q3_1[k], Error_q3_2[k], Error_q3_3[k])
        print("")
        k += 1

    cross_term_u = "$\\partial_y \\partial_x u$"
    cross_term_v = "$\\partial_x \\partial_y v$"
    
    with PdfPages(f"Plots/divF/test_P{spacePolynOrder}_{basisChoice}_{modificationChoice}_operators_") as pdf:
        # Résultats des opérations sur la première composante
        plt.figure()
        plt.title(f"Erreurs en norme $L^{{\\infty}}$ sur les différentes composantes de $\\widetilde{{\\nabla}} f$ \n \
                  P{spacePolynOrder} {basisChoice} {modificationChoice} avec v = p = 0")
        plt.plot(hList, Error_q1_1, label="$\\partial^2_x u$", marker="o")
        plt.plot(hList, Error_q1_2, label=cross_term_u, marker="^")
        plt.plot(hList, Error_q1_3, label="$\\partial_x u$", marker="s")
        plt.plot(hList, ( max(Error_q1_1[0], Error_q1_2[0], Error_q1_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
        plt.plot(hList, ( max(Error_q1_1[0], Error_q1_2[0], Error_q1_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
        plt.xlabel("h")
        plt.ylabel("error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()

        # Résultats des opérations sur la seonde composante
        plt.figure()
        plt.title(f"Erreurs en norme $L^{{\\infty}}$ sur les différentes composantes de $\\widetilde{{\\nabla}} f$ \n \
                  P{spacePolynOrder} {basisChoice} {modificationChoice} avec u = p = 0")
        plt.plot(hList, Error_q2_1, label=cross_term_v, marker="o")
        plt.plot(hList, Error_q2_2, label="$\\partial^2_y v$", marker="^")
        plt.plot(hList, Error_q2_3, label="$\\partial_y v$", marker="s")
        plt.plot(hList, ( max(Error_q2_1[0], Error_q2_2[0], Error_q2_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
        plt.plot(hList, ( max(Error_q2_1[0], Error_q2_2[0], Error_q2_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
        plt.xlabel("h")
        plt.ylabel("error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()

        # Résultats des opérations sur la troisième composante
        plt.figure()
        plt.title(f"Erreurs en norme $L^{{\\infty}}$ sur les différentes composantes de $\\widetilde{{\\nabla}} f$ \n \
                  P{spacePolynOrder} {basisChoice} {modificationChoice} avec u = v = 0")
        plt.plot(hList, Error_q3_1, label="$\\partial_x p$", marker="o")
        plt.plot(hList, Error_q3_2, label="$\\partial_y p$", marker="^")
        plt.plot(hList, Error_q3_3, label="$\\Delta p$", marker="s")
        plt.plot(hList, ( max(Error_q3_1[0], Error_q3_2[0], Error_q3_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
        plt.plot(hList, ( max(Error_q3_1[0], Error_q3_2[0], Error_q3_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
        plt.xlabel("h")
        plt.ylabel("error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()





####################################################################################################################

#                                     MASS OPERATOR : INDIVIDUAL TESTS                                             #

####################################################################################################################





# if (True) :
if (False) :
    # Creation of the grid_parameters dictionnary by hand
    xL, xR = 0., 1.
    yL, yR = 0., 1.
    Nx, Ny = 100, 100
    nGhost = 2
    spacePolynOrder = 1
    NVis = spacePolynOrder
    iTest = 4
    alpha = 0.5


    grid_params = {"domain_parameters" : {"xL" : xL, "xR" : xR, "yL" : yL, "yR" : yR}, "mesh_parameters" : {"Nx" : Nx, "Ny" : Ny, "nGhost" : nGhost}}
    params = {"grid_parameters" : grid_params, "order" : {"space_polyn_order" : spacePolynOrder}, "number_visualisation_points" : NVis}

    gridOp = grid_mod.gridOperator(params)
    iMin, iMax, jMin, jMax = gridOp.validGridIndices
    dx, dy = gridOp.steps
    h = gridOp.characteristic_size
    basisChoice = "basic"
    modificationChoice = ""
    massCoeffs = get_arbitrar_order_SUPG_weighted_mass_operator_coeffs(basisChoice, modificationChoice, spacePolynOrder, dx, dy, alpha, h)

    Id = np.array([ np.zeros((spacePolynOrder, spacePolynOrder)), np.eye(spacePolynOrder), np.zeros((spacePolynOrder, spacePolynOrder)) ])

    if iTest == 1: ### TEST Identity :
        for i in range(20):
            massCoeffs[i] = Id
    elif iTest == 2: ### TEST : x operators (y operators = Id)
        for i in range(10):
            massCoeffs[2 * i + 1] = Id
    elif iTest == 3: ### TEST : y operators (x operators = Id)
        for i in range(10):
            massCoeffs[2 * i] = Id
    


    # SLICES DE REFERENCE
    # INJECTION MAILLAGE CALCUL (x_i^k) DANS MAILLAGE RAFFINE (ksi_alpha^gamma) :
    # x_i^k = ksi_{i * n_raffinement + k * n_raffinement // ordre}^{k * n_raffinement % ordre}
    n_raffinement = 10

    ### SLICE HORIZONTAL
    NxRef = n_raffinement * Nx * spacePolynOrder
    Xref = np.linspace(xL, xR, NxRef+1)
    j_ref, l_ref = int(Ny / 2.), 0

    ### SLICE VERTICAL
    NyRef = n_raffinement * Ny * spacePolynOrder
    Yref = np.linspace(yL, yR, NyRef+1)
    i_ref, k_ref = int(Nx / 2.), 0

    vertical_index_ref_mesh = (j_ref * n_raffinement + (l_ref * n_raffinement) // spacePolynOrder) * spacePolynOrder + (l_ref * n_raffinement) % spacePolynOrder
    y_ref = Yref[vertical_index_ref_mesh]
    horizontal_index_ref_mesh = (i_ref * n_raffinement + (k_ref * n_raffinement) // spacePolynOrder) * spacePolynOrder + (k_ref * n_raffinement) % spacePolynOrder
    x_ref = Xref[horizontal_index_ref_mesh]

    XXref, YYref = np.meshgrid(Xref, Yref, indexing="ij")

    paramRef = params


    testFonction = test6

    f_HO_mesh, f_x_HO_mesh, f_y_HO_mesh, f_xx_HO_mesh, f_yy_HO_mesh, f_xy_HO_mesh = getTestFonctionLagrangeArbitrarOrderInterpolation(xL, xR, yL, yR, Nx, Ny, nGhost, spacePolynOrder, testFonction)
    f_ref_mesh, f_x_ref_mesh, f_y_ref_mesh, f_xx_ref_mesh, f_yy_ref_mesh, f_xy_ref_mesh = \
        testFonction(XXref, YYref)
    zeros_ref_mesh = np.zeros_like(XXref)

    # # test periodicité de f :
    # print("test périodicité de f : ")
    # print("écarts bords verticaux : ", np.max(abs(f_HO_mesh[iMin, jMin:jMax+1, 0, :] - f_HO_mesh[iMax+1, jMin:jMax+1, 0, :])))
    # print("écarts bords verticaux : ", np.max(abs(f_HO_mesh[iMin:iMax+1, jMin, :, 0] - f_HO_mesh[iMin:iMax+1, jMax+1, :, 0])), "\n")

    f_HO_mesh_flattened = getVisualisationData(f_HO_mesh, params)
    f_y_HO_mesh_flattened = getVisualisationData(f_y_HO_mesh, params)


    q1, q2, q3 = np.zeros((3, Nx + 2*nGhost, Ny + 2*nGhost, spacePolynOrder, spacePolynOrder, 3))
    q1[:, :, :, :, 0] = f_HO_mesh
    q2[:, :, :, :, 1] = f_HO_mesh
    q3[:, :, :, :, 2] = f_HO_mesh


    mass1 = get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator(q1, iMin, iMax, jMin, jMax, spacePolynOrder, massCoeffs)
    mass2 = get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator(q2, iMin, iMax, jMin, jMax, spacePolynOrder, massCoeffs)
    mass3 = get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator(q3, iMin, iMax, jMin, jMax, spacePolynOrder, massCoeffs)

    # print("test périodicité de divFlux1 : ")
    # print("écarts bords verticaux : ", np.max(abs(divFlux1[iMin, jMin:jMax+1, 0, :] - divFlux1[iMax+1, jMin:jMax+1, 0, :])))
    # print("écarts bords verticaux : ", np.max(abs(divFlux1[iMin:iMax+1, jMin, :, 0] - divFlux1[iMin:iMax+1, jMax+1, :, 0])), "\n")

    # Mise à plat des données
    mass1_2D = getVisualisationData(mass1[iMin:iMax+2, jMin:jMax+2], params)
    mass2_2D = getVisualisationData(mass2[iMin:iMax+2, jMin:jMax+2], params)
    mass3_2D = getVisualisationData(mass3[iMin:iMax+2, jMin:jMax+2], params)

    zeros = zeros_ref_mesh[::n_raffinement, ::n_raffinement]
    f = f_ref_mesh[::n_raffinement, ::n_raffinement]
    f_x = f_x_ref_mesh[::n_raffinement, ::n_raffinement]
    f_xx = f_xx_ref_mesh[::n_raffinement, ::n_raffinement]
    f_y = f_y_ref_mesh[::n_raffinement, ::n_raffinement]
    f_yy = f_yy_ref_mesh[::n_raffinement, ::n_raffinement]
    f_xy = f_xy_ref_mesh[::n_raffinement, ::n_raffinement]


    # print("test projection de maillage pour f : ", np.max(abs(f - f_HO_mesh_flattened)))
    # print("test projection de maillage pour f_y : ", np.max(abs(f_y - f_y_HO_mesh_flattened)))

    var = [("u", XVEL), ("v", YVEL), ("p", PRES)]

    if iTest == 1: # Test avec identité : indices de stencil cohérents : toutes les sorties de mass sont f
        print("="*50, "\n"*2)
        print("Test 1 : tous les opérateurs sont l'identité", "\n"*2)
        print("="*50, "\n"*2)

        exact_operators_names = [["", "", ""], 
                                 ["", "", ""], \
                                 ["", "", ""]   ]
        exact_operators_ref = np.array([[f_ref_mesh, f_ref_mesh, f_ref_mesh], \
                                        [f_ref_mesh, f_ref_mesh, f_ref_mesh], \
                                        [f_ref_mesh, f_ref_mesh, f_ref_mesh]])
        exact_operators = np.array([[f, f, f], \
                                    [f, f, f], \
                                    [f, f, f]])
        
        masses = np.array([   [mass1_2D[:, :, XVEL], mass2_2D[:, :, XVEL], mass3_2D[:, :, XVEL]], \
                              [mass1_2D[:, :, YVEL], mass2_2D[:, :, YVEL], mass3_2D[:, :, YVEL]], \
                              [mass1_2D[:, :, PRES], mass2_2D[:, :, PRES], mass3_2D[:, :, PRES]]])


    if iTest == 2: # Test x opérateurs (y operateurs = Id)
        print("="*50, "\n"*2)
        print("Test 2 : tous les opérateurs en y sont l'identité", "\n"*2)
        print("="*50, "\n"*2)

        exact_operators_names = [["$\\partial_x^2$", "$\\partial_x$", "$\\partial_x$"], 
                                 ["$\\partial_x$", "", ""], \
                                 ["$\\partial_x$", "", "$(- \\Delta x \\partial_x^2 + Id)$"]   ]
        exact_operators_ref = np.array([[f_xx_ref_mesh, f_x_ref_mesh, f_x_ref_mesh], \
                                        [f_x_ref_mesh, f_ref_mesh, f_ref_mesh], \
                                        [f_x_ref_mesh, f_ref_mesh, - dx * f_xx_ref_mesh + f_ref_mesh]])
        exact_operators = np.array([[f_xx, f_x, f_x], \
                                    [f_x, f, f], \
                                    [f_x, f, - dx * f_xx + f]])
        
        masses = np.array([ [  mass1_2D[:, :, XVEL]              , mass2_2D[:, :, XVEL], - mass3_2D[:, :, XVEL] / (alpha * h) ], \
                            [  mass1_2D[:, :, YVEL]              , mass2_2D[:, :, YVEL],   mass3_2D[:, :, YVEL]               ], \
                            [- mass1_2D[:, :, PRES] / (alpha * h), mass2_2D[:, :, PRES],   mass3_2D[:, :, PRES]               ]])
            


    if iTest == 3: # Test y opérateurs (x operateurs = Id)
        print("="*50, "\n"*2)
        print("Test 3 : tous les opérateurs en x sont l'identité", "\n"*2)
        print("="*50, "\n"*2)

        exact_operators_names = [["", "$\\partial_y$", ""], 
                                 ["$\\partial_y$", "$\\partial^2_y$", "$\\partial_y$"], \
                                 ["", "$\\partial_y$", "$(- \\Delta y \\partial_y^2 + Id)$"]   ]
        exact_operators_ref = np.array([[f_ref_mesh, f_y_ref_mesh, f_ref_mesh], \
                                        [f_y_ref_mesh, f_yy_ref_mesh, f_y_ref_mesh], \
                                        [f_ref_mesh, f_y_ref_mesh, - dy * f_yy_ref_mesh + f_ref_mesh]])
        exact_operators = np.array([[f, f_y, f], \
                                    [f_y, f_yy, f_y], \
                                    [f, f_y, - dy * f_yy + f]])
        
        masses = np.array([ [mass1_2D[:, :, XVEL],   mass2_2D[:, :, XVEL]              ,   mass3_2D[:, :, XVEL]               ], \
                            [mass1_2D[:, :, YVEL],   mass2_2D[:, :, YVEL]              , - mass3_2D[:, :, YVEL] / (alpha * h) ], \
                            [mass1_2D[:, :, PRES], - mass2_2D[:, :, PRES] / (alpha * h),   mass3_2D[:, :, PRES]               ]])


    elif iTest == 4:
        print("="*50, "\n"*2)
        print("Test 4 : tous les opérateurs sont laissés tels quels", "\n"*2)
        print("="*50, "\n"*2)

        exact_operators_names = [["$\\mathsf{{Id}}$", "$0$", "$\\partial_x$"], 
                                 ["0", "$\\mathsf{{Id}}$", "$\\partial_y$"], \
                                 ["$\\partial_x$", "$\\partial_y$", "$\\mathsf{{Id}}$"]   ]
        exact_operators_ref = np.array([[f_ref_mesh, zeros_ref_mesh, f_x_ref_mesh], \
                                        [zeros_ref_mesh, f_ref_mesh, f_y_ref_mesh], \
                                        [f_x_ref_mesh, f_y_ref_mesh, f_ref_mesh]])
        exact_operators = np.array([[f, zeros, f_x], \
                                    [zeros, f, f_y], \
                                    [f_x, f_y, f]])
        
        masses = np.array([ [  mass1_2D[:, :, XVEL]              ,   mass2_2D[:, :, XVEL]              , - mass3_2D[:, :, XVEL] / (alpha * h) ], \
                            [  mass1_2D[:, :, YVEL]              ,   mass2_2D[:, :, YVEL]              , - mass3_2D[:, :, YVEL] / (alpha * h) ], \
                            [- mass1_2D[:, :, PRES] / (alpha * h), - mass2_2D[:, :, PRES] / (alpha * h),   mass3_2D[:, :, PRES]               ]])



    # ECARTS 
    for equationName, equation in var:
        print(f"Opérations de l'équation sur {equationName} : ")
        for termName, term in var:
            print(f"   sur {termName} : ", np.max(abs(masses[equation, term] - exact_operators[equation, term])))
        print("")


    # PLOTS
    if True :
    # if False :
        vertical_index_ref_mesh = (j_ref * n_raffinement + l_ref * n_raffinement // spacePolynOrder) * spacePolynOrder + (l_ref * n_raffinement) % spacePolynOrder
        y_ref = Yref[vertical_index_ref_mesh]
        horizontal_index_ref_mesh = (i_ref * n_raffinement + k_ref * n_raffinement // spacePolynOrder) * spacePolynOrder + (k_ref * n_raffinement) % spacePolynOrder
        x_ref = Xref[horizontal_index_ref_mesh]

        for equationName, equation in var:
            with PdfPages(f"Plots/mass/1D_slices_{equationName}_equation_order_{spacePolynOrder}") as pdf:
                for termName, term in var:
                    # exact operator 1D slices
                    exact_operator = exact_operators_ref[equation, term]
                    exact_operator_name = exact_operators_names[equation][term]
                    horizontal_slice = exact_operator[:, vertical_index_ref_mesh]
                    vertical_slice = exact_operator[horizontal_index_ref_mesh, :]

                    # approximate operator
                    approxOperator = masses[equation, term]

                    # Slice horizontal
                    plt.figure()
                    plt.title(f"Slice horizontal [0, 1]x{{{y_ref}}}: {exact_operator_name}{termName} dans l'équation en {equationName}")
                    plt.plot(Xref, horizontal_slice, label="exact")
                    plt.plot(Xref[::n_raffinement], approxOperator[:, j_ref*spacePolynOrder + l_ref], label="approx")
                    plt.legend()
                    pdf.savefig()
                    plt.close()
        
                    # Slice vertical
                    plt.figure()
                    plt.title(f"Slice vertical {{{x_ref}}}x[0, 1]: {exact_operator_name}{termName} dans l'équation en {equationName}")
                    plt.plot(Yref, vertical_slice, label="exact")
                    plt.plot(Yref[::n_raffinement], approxOperator[i_ref*spacePolynOrder + k_ref, :], label="approx")
                    plt.legend()
                    pdf.savefig()
                    plt.close()




####################################################################################################################

#                                      MASS OPERATOR : CONVERGENCE TESTS                                           #

####################################################################################################################



# if (True) :
if (False) :
    # Creation of the grid_parameters dictionnary by hand
    xL, xR = 0., 1.
    yL, yR = 0., 1.
    nGhost = 2
    spacePolynOrder = 2
    grid_params = {"domain_parameters" : {"xL" : xL, "xR" : xR, "yL" : yL, "yR" : yR}, "mesh_parameters" : {"Nx" : 0, "Ny" : 0, "nGhost" : nGhost}}
    params = {"grid_parameters" : grid_params, "order" : {"space_polyn_order" : spacePolynOrder}}
    
    
    nList = [100, 200, 400, 800]
    hList = np.ones(len(nList))
    Error_q1_1, Error_q1_2, Error_q1_3, Error_q2_1, Error_q2_2, Error_q2_3, \
        Error_q3_1, Error_q3_2, Error_q3_3 = np.ones((9, len(nList)))

    k = 0
    for n in nList :

        grid_params["mesh_parameters"]["Nx"] = n
        grid_params["mesh_parameters"]["Ny"] = n
        gridOp = grid_mod.gridOperator(params)
        iMin, iMax, jMin, jMax = gridOp.validGridIndices
        dx, dy = gridOp.steps
        basisChoice = "basic"
        modificationChoice = ""
        massCoeffs = get_arbitrar_order_SUPG_weighted_mass_operator_coeffs(basisChoice, modificationChoice, spacePolynOrder, dx, dy, alpha, h)

        # print("massCoeffs : \n ", massCoeffs)

        hList[k] = dx

        testFonction = test5

        f, f_x, f_y = \
            getTestFonctionLagrangeArbitrarOrderInterpolation(xL, xR, yL, yR, n, n, nGhost, spacePolynOrder, testFonction)[:3]


        q1, q2, q3 = np.zeros((3, n + 2 * nGhost, n + 2 * nGhost, spacePolynOrder, spacePolynOrder, 3))
        q1[:, :, :, :, 0] = f
        q2[:, :, :, :, 1] = f
        q3[:, :, :, :, 2] = f

        # FIRST COMPONENT OPERATIONS :
        Error_q1_1[k], Error_q1_2[k], Error_q1_3[k] = \
            mass_operator_first_component_operations_errors(q1, iMin, iMax, jMin, jMax, dx, dy, f, f_x, gridOp, spacePolynOrder, massCoeffs)
        Error_q2_1[k], Error_q2_2[k], Error_q2_3[k] = \
            mass_operator_second_component_operations_errors(q2, iMin, iMax, jMin, jMax, dx, dy, f, f_y, gridOp, spacePolynOrder, massCoeffs)
        Error_q3_1[k], Error_q3_2[k], Error_q3_3[k] = \
            mass_operator_third_component_operations_errors(q3, iMin, iMax, jMin, jMax, dx, dy, f, f_x, f_y, gridOp, spacePolynOrder, massCoeffs)

        print("")
        print("nombre de points par côté : ", n)
        print("Erreurs opérations sur 1e compsante : ", Error_q1_1[k], Error_q1_2[k], Error_q1_3[k])
        print("Erreurs opérations sur 2nd compsante : ", Error_q2_1[k], Error_q2_2[k], Error_q2_3[k])
        print("Erreurs opérations sur 3e compsante : ", Error_q3_1[k], Error_q3_2[k], Error_q3_3[k])
        print("")
        k += 1
    
    with PdfPages(f"Plots/mass/test_operators_order_{spacePolynOrder}") as pdf:
        # Résultats des opérations sur la première composante   
        plt.figure()
        plt.title(f"Erreurs en norme $L^{{\\infty}}$ sur les différentes composantes de $\\mathcal{{A}} f$ \n pour l'ordre {spacePolynOrder} avec v = p = 0")
        plt.plot(hList, Error_q1_1, label="$u$", marker="o")
        plt.plot(hList, Error_q1_2, label="0", marker="^")
        plt.plot(hList, Error_q1_3, label="$\\partial_x u$", marker="s")
        plt.plot(hList, ( max(Error_q1_1[0], Error_q1_2[0], Error_q1_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
        plt.plot(hList, ( max(Error_q1_1[0], Error_q1_2[0], Error_q1_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
        plt.xlabel("h")
        plt.ylabel("error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()

        # Résultats des opérations sur la seonde composante
        plt.figure()
        plt.title(f"Erreurs en norme $L^{{\\infty}}$ sur les différentes composantes de $\\mathcal{{A}} f$ \n pour l'ordre {spacePolynOrder} avec u = p = 0")
        plt.plot(hList, Error_q2_1, label="0", marker="o")
        plt.plot(hList, Error_q2_2, label="$v$", marker="^")
        plt.plot(hList, Error_q2_3, label="$\\partial_y v$", marker="s")
        plt.plot(hList, ( max(Error_q2_1[0], Error_q2_2[0], Error_q2_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
        plt.plot(hList, ( max(Error_q2_1[0], Error_q2_2[0], Error_q2_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
        plt.xlabel("h")
        plt.ylabel("error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()

        # Résultats des opérations sur la troisième composante
        plt.figure()
        plt.title(f"Erreurs en norme $L^{{\\infty}}$ sur les différentes composantes de $\\mathcal{{A}} f$ \n pour l'ordre {spacePolynOrder} avec u = v = 0")
        plt.plot(hList, Error_q3_1, label="$\\partial_x p$", marker="o")
        plt.plot(hList, Error_q3_2, label="$\\partial_y p$", marker="^")
        plt.plot(hList, Error_q3_3, label="$p$", marker="s")
        plt.plot(hList, ( max(Error_q3_1[0], Error_q3_2[0], Error_q3_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
        plt.plot(hList, ( max(Error_q3_1[0], Error_q3_2[0], Error_q3_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
        plt.xlabel("h")
        plt.ylabel("error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()





####################################################################################################################

#                                        MASS INVERSION : "TRUE ORDERS"                                            #

####################################################################################################################


# INDEX PATH (flattening 4D array to 1D) : "(i, j, k, l)" to "alpha = i * nDOF + k + (j * nDOF + l) * nXTotalCells * nDOF"
# THEN : crossing the alpha indices by ascending order 
# alpha = i * nDOF + k + (j * nDOF + l) * nXTotalCells * nDOF 
#       = k + i * nDOF + l * nXTotalCells * nDOF + j * nXTotalCells * nDOF ** 2
# so k in the inner-most index, then i, then...
def getPeriodicGhostRowsCOOEntries(position, nXValidCells, nYValidCells, nDOF, nGhost):

    nXValidDOFs = nXValidCells * nDOF
    nXTotalCells = nXValidCells + 2 * nGhost
    nXTotalDOFs = nXTotalCells*nDOF
    nGhostDOFs = nGhost*nDOF
    xPeriodicShift = nXValidCells * nDOF
    
    if position == "lower" :
        first_row_index = 0
        yPeriodicShift = nYValidCells * nXTotalCells * nDOF**2
    elif position == "upper" :
        first_row_index = (nGhost + nYValidCells) * nDOF * nXTotalDOFs
        yPeriodicShift = -nYValidCells * nXTotalCells * nDOF**2
    else :
        raise ValueError(f"position must be 'lower' or 'upper', got {position!r}")

    bloc_size = nXTotalDOFs * nGhostDOFs
    diag = first_row_index + np.arange(bloc_size)                       # colonne de l'entrée +1

    # décalage vers l'image périodique, pour une ligne de DOFs (gauche | valide | droite)
    one_row_shifts = np.concatenate((np.full(nGhostDOFs, xPeriodicShift), \
                                     np.zeros(nXValidDOFs, dtype=int), \
                                     np.full(nGhostDOFs, -xPeriodicShift))) + yPeriodicShift

    periodic_image = diag + np.tile(one_row_shifts, nGhostDOFs)         # colonne de l'entrée -1

    rows = np.repeat(diag, 2)
    cols = np.stack((diag, periodic_image), axis=-1).ravel()
    values = np.tile([1, -1], bloc_size)
    
    return rows, cols, values



def getBasicMassOtimesIdentityC00(nXValidCells, nYValidCells, nDOF, nGhost):

    # Mass : [0, 0] has 3 coeffs, [i, 0] and [0, j] for i,j>0 have 2, the other have 1
    nb_mass_coeffs = (nDOF + 2)*nDOF       # = (3 + 2 * (2 * (nDOF - 1)) + 1 * (nDOF - 1)**2)
    nXValidDOFs = nXValidCells * nDOF
    nYValidDOFs = nYValidCells * nDOF
    nXTotalCells = nXValidCells + 2 * nGhost
    nXTotalDOFs = nXTotalCells*nDOF
    nGhostDOFs = nGhost*nDOF
    xPeriodicShift = nXValidCells * nDOF
    mesh_interm_row_nb_of_coeffs = 4 * nGhost * nDOF + nXValidCells * nb_mass_coeffs

    if nDOF == 1 :
        mass_entries = np.array([1, 4, 1], dtype=float) / 6.
        mass_entries_row_index = np.zeros(3)
        mass_rows_neighbors = np.array([-1, 0, 1])
        
    if nDOF == 2 :
        mass_entries = np.array([-1, 2, 8, 2, -1, 2, 16, 2], dtype=float) / 30.
        mass_entries_row_index = np.concatenate((np.full(5, 0), np.full(3, 1)))
        mass_rows_neighbors = np.array([-2, -1, 0, 1, 2, -1, 0, 1])



    # FIRST ROWS OF GHOST DOFs ; j < nGhost
    First_bloc_rows, First_bloc_cols, First_bloc_values = getPeriodicGhostRowsCOOEntries("lower", nXValidCells, nYValidCells, nDOF, nGhost)



    # INTERMEDIATE ROWS OF DOFs : nGhost <= j <= nGhost + nx - 1
    first_row_index = nGhostDOFs * nXTotalDOFs

    # FIRST INTERMEDIATE ROW OF THE MESH :
    # Rows
    interm_row_valid_DOFs_rows = np.tile(mass_entries_row_index, nXValidCells) \
                             + np.repeat(np.arange(nXValidCells)*nDOF, nb_mass_coeffs) \
                             + nGhostDOFs
    interm_row_DOFs_rows = np.concatenate((np.repeat(np.arange(nGhostDOFs), 2), \
                                           interm_row_valid_DOFs_rows, \
                                           np.repeat(np.arange(nGhostDOFs), 2) + (nXValidCells+nGhost)*nDOF)) \
                           + first_row_index

    # Cols
    interm_row_valid_DOFs_relativ_cols = np.tile(mass_rows_neighbors, nXValidCells)
    interm_row_relativ_cols = np.concatenate((np.tile([0, xPeriodicShift], nGhostDOFs), \
                                               interm_row_valid_DOFs_relativ_cols, \
                                               np.tile([0, -xPeriodicShift], nGhostDOFs)))
    interm_row_DOFs_cols = interm_row_relativ_cols + interm_row_DOFs_rows
    
    # Values
    interm_row_DOFs_values = np.concatenate((np.tile([1, -1], nGhostDOFs), \
                                                np.tile(mass_entries, nXValidCells), \
                                                np.tile([1, -1], nGhostDOFs)))

    # ALL INTERMEDIATE ROWS OF THE MESH
    interm_rows_DOFs_rows = np.tile(interm_row_DOFs_rows, nYValidDOFs) \
                                + np.repeat(np.arange(nYValidDOFs)*nXTotalDOFs, mesh_interm_row_nb_of_coeffs)
    interm_rows_DOFs_cols = np.tile(interm_row_DOFs_cols, nYValidDOFs) \
                            + np.repeat(np.arange(nYValidDOFs)*nXTotalDOFs, mesh_interm_row_nb_of_coeffs) 
    interm_rows_DOFs_values = np.tile(interm_row_DOFs_values, nYValidDOFs)



    # LAST ROWS OF GHOST DOFs ; nGhost + nx < j
    Last_bloc_rows, Last_bloc_cols, Last_bloc_values = getPeriodicGhostRowsCOOEntries("upper", nXValidCells, nYValidCells, nDOF, nGhost)



    # FINAL ASSEMBLING
    IM = np.concatenate((First_bloc_rows, interm_rows_DOFs_rows, Last_bloc_rows))
    JM = np.concatenate((First_bloc_cols, interm_rows_DOFs_cols, Last_bloc_cols))
    M = np.concatenate((First_bloc_values, interm_rows_DOFs_values, Last_bloc_values))

    return IM, JM, M



# q_ex and q_app are 1D arrays
def make1DSlicesPlots(grid, q_ex, q_app, approx_type, basisChoice, spacePolynOrder, modificationChoice, i_slice, j_slice, k_slice, l_slice):
    iMin, iMax, jMin, jMax = grid.validGridIndices
    X, Y = grid.validGrid
    nx = iMax - iMin + 1
    ny = jMax - jMin + 1
    nGhost = iMin
    nDOF = spacePolynOrder

    # extracting the horizontal slices
    horizontal_slice_start = nGhost * nDOF + 0 + (j_slice * nDOF + l_slice) * (nx+2*nGhost) * nDOF    # (nG, 0, j_slice, l_slice)
    horizontal_slice_end = horizontal_slice_start + nx * nDOF + 1                                     # (nG+nx, 0, j_slice, l_slice)
    horizontal_slice_step = 1
    q_ex_horizontal_slice = q_ex[horizontal_slice_start : horizontal_slice_end : horizontal_slice_step]
    q_app_horizontal_slice = q_app[horizontal_slice_start : horizontal_slice_end : horizontal_slice_step]
    X_horizontal_slice = np.zeros(nx*nGhost+1)
    for i in range(nx*nDOF+1):
        X_horizontal_slice[i] = X[i // nDOF, 0, i % nDOF, 0]

    # extracting the vertical slices
    vertical_slice_start = i_slice * nDOF + k_slice + (nGhost * nDOF + 0) * (nx+2*nGhost) * nDOF    # (nG, 0, j_slice, l_slice)
    vertical_slice_end = vertical_slice_start + ny * nDOF * (nx+2*nGhost) * nDOF + 1                # (nG+nx, 0, j_slice, l_slice)
    vertical_slice_step = (nx+2*nGhost) * nDOF
    q_ex_vertical_slice = q_ex[vertical_slice_start : vertical_slice_end : vertical_slice_step]
    q_app_vertical_slice = q_app[vertical_slice_start : vertical_slice_end : vertical_slice_step]
    Y_vertical_slice = np.zeros(ny*nGhost+1)
    for j in range(ny*nDOF+1):
        Y_vertical_slice[j] = Y[0, j // nDOF, 0, j % nDOF]

    x_slice = X[i_slice-nGhost, 0, k_slice, 0]
    y_slice = Y[0, j_slice-nGhost, 0, l_slice]

    with PdfPages(f"Plots/divF/{approx_type}_{nx}_1Dslices_{basisChoice}_P{spacePolynOrder}_{modificationChoice}") as pdf:
        # Slice horizontal
        plt.figure()
        plt.title(f"Slice horizontal [0, 1]x{{{y_slice}}} : {approx_type}")
        plt.plot(X_horizontal_slice, q_ex_horizontal_slice, label="exact")
        plt.plot(X_horizontal_slice, q_app_horizontal_slice, label="approx")
        plt.legend()
        pdf.savefig()
        plt.close()

        # Slice vertical
        plt.figure()
        plt.title(f"Slice vertical {{{x_slice}}}x[0, 1] : {approx_type}")
        plt.plot(Y_vertical_slice, q_ex_vertical_slice, label="exact")
        plt.plot(Y_vertical_slice, q_app_vertical_slice, label="approx")
        plt.legend()
        pdf.savefig()
        plt.close()

if True :
# if False :
    # Creation of the grid_parameters dictionnary by hand
    xL, xR = 0., 1.
    yL, yR = 0., 1.
    nGhost = 2
    spacePolynOrder = 1
    grid_params = {"domain_parameters" : {"xL" : xL, "xR" : xR, "yL" : yL, "yR" : yR}, "mesh_parameters" : {"Nx" : 0, "Ny" : 0, "nGhost" : nGhost}}
    params = {"grid_parameters" : grid_params, "order" : {"space_polyn_order" : spacePolynOrder}}
    alpha = 0.5
    functionChoice = 5
    doSlicesPlots = False

    nList = [100, 200, 400, 800]
    hList = np.ones(len(nList))
    Error_fx, Error_fxx = np.ones((2, len(nList)))

    simuIndex = 0
    for n in nList :

        grid_params["mesh_parameters"]["Nx"] = n
        grid_params["mesh_parameters"]["Ny"] = n
        gridOp = grid_mod.gridOperator(params)
        iMin, iMax, jMin, jMax = gridOp.validGridIndices
        dx, dy = gridOp.steps
        h = gridOp.characteristic_size
        basisChoice = "basic"
        modificationChoice = ""
        IM, JM, M = getBasicMassOtimesIdentityC00(n, n, spacePolynOrder, nGhost)
        nTotalDofs = (n + 2 * nGhost) * (n + 2 * nGhost) * spacePolynOrder**2
        
        M_coo = sp.sparse.coo_matrix((M, (IM, JM)), shape=(nTotalDofs, nTotalDofs))
        M_csr = M_coo.tocsr()
        evolCoeffs = get_arbitrar_order_SUPG_evolution_operator_coeffs(basisChoice, modificationChoice, spacePolynOrder, dx, dy, alpha, h)
        # Modifications of the coeffs :
        Id = np.array([np.zeros((spacePolynOrder, spacePolynOrder)), np.eye(spacePolynOrder), np.zeros((spacePolynOrder, spacePolynOrder))])
        evolCoeffs[1] = Id 
        evolCoeffs[13] = Id

        # function and parameters choices
        if functionChoice == 1 : # f(x, y) = cos(2 kx pi x) + sin(2 ky pi y) 
            testFonction = test1
            f_params = 2, 5 # kx, ky
            functionDef = f"$f(x, y) = \\cos(2 \\times {f_params[0]} \\pi x) + \\sin(2 \\times {f_params[1]} \\pi y)$"

        if functionChoice == 2 : # f(x, y) = a x^2 + b y^2 + c x + d y + e
            testFonction = test2
            f_params = np.pi, np.exp(1), 13., np.sqrt(2.), 3. # a, b, c, d, e
            functionDef = f"$f(x, y) = {f_params[0]} x^2 + {f_params[1]} y^2 + {f_params[2]} x + {f_params[3]} y + {f_params[4]}$"

        if functionChoice == 3 : # f(x, y) = cos(2 kx pi x) * cos(2 ky pi y)
            testFonction = test3
            f_params = 2, 5 # kx, ky
            functionDef = f"$f(x, y) = \\cos(2 \\times {f_params[0]} \\pi x) \\cos(2 \\times {f_params[1]} \\pi y)$"

        if functionChoice == 4 : # f(x, y) = a x^2 + b y^2 + c x*y + d x + e y + f
            testFonction = test4
            f_params = np.pi, -np.exp(1), -13., np.sqrt(2.), 3., -7./3. # a, b, c, d, e
            functionDef = f"$f(x, y) = {f_params[0]} x^2 + {f_params[1]} y^2 + {f_params[2]} xy + {f_params[3]} x + {f_params[4]} y + {f_params[5]}$"

        if functionChoice == 5 : # f(x, y) = sin(2 kx pi x) * sin(2 ky pi y)
            testFonction = test5
            f_params = 5, 5 # kx, ky
            functionDef = f"$f(x, y) = \\sin(2 \\times {f_params[0]} \\pi x) \\sin(2 \\times {f_params[1]} \\pi y)$"

        if functionChoice == 6 : # f(x, y) = x^a . (1-x)^b . y^c . (1-y)^d
            testFonction = test6
            f_params = 1., 1., 1., 1.
            functionDef = f"$f(x, y) = x^{{f_params[0]}} (1-x)^{{f_params[1]}} y^{{f_params[2]}} (1-y)^{{f_params[3]}}$"


        

        f, f_x, _, f_xx = \
            getTestFonctionLagrangeArbitrarOrderInterpolation(xL, xR, yL, yR, n, n, nGhost, spacePolynOrder, testFonction, f_params)[:4]


        q = np.zeros((n + 2 * nGhost, n + 2 * nGhost, spacePolynOrder, spacePolynOrder, 3))
        q[:, :, :, :, 0] = f

        divFlux = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q, iMin, iMax, jMin, jMax, spacePolynOrder, evolCoeffs)
        divFlux[:nGhost] = 0.
        divFlux[:, :nGhost] = 0.
        divFlux[nGhost+n:] = 0.
        divFlux[:, nGhost+n:] = 0.

        # Extraction et applatissement de f_x_tilde et f_xx_tilde, aplatissement de f_x et f_xx
        f_x_tilde, f_xx_tilde, f_x_flattened, f_xx_flattened, massed_f_x_tilde, massed_f_xx_tilde = np.zeros((6, (n+2*nGhost)**2 * spacePolynOrder**2))
        for j in range(n+2*nGhost):
          for l in range(spacePolynOrder):
            for i in range(n+2*nGhost):
              for k in range(spacePolynOrder):
                index1D = i * spacePolynOrder + k + (j * spacePolynOrder + l) * (n + 2 * nGhost) * spacePolynOrder
                f_x_tilde[index1D] = divFlux[i, j, k, l, 2]
                f_xx_tilde[index1D] = - divFlux[i, j, k, l, 0] / (alpha * h)
                f_x_flattened[index1D] = f_x[i, j, k, l]
                f_xx_flattened[index1D] = f_xx[i, j, k, l]

        massed_f_x_tilde = sp.sparse.linalg.spsolve(M_csr, f_x_tilde)
        massed_f_xx_tilde = sp.sparse.linalg.spsolve(M_csr, f_xx_tilde)

        if doSlicesPlots :
            # 1D slices plots
            i_slice = int(n/4)+nGhost
            j_slice = int(3*n/4)+nGhost
            k_slice = 0
            l_slice = 0
            make1DSlicesPlots(gridOp, f_x_flattened, massed_f_x_tilde, "der", basisChoice, spacePolynOrder, modificationChoice, i_slice, j_slice, k_slice, l_slice)
            make1DSlicesPlots(gridOp, f_xx_flattened, massed_f_xx_tilde, "second_der", basisChoice, spacePolynOrder, modificationChoice, i_slice, j_slice, k_slice, l_slice)

        hList[simuIndex] = dx
        Error_fx[simuIndex] = np.max(abs(massed_f_x_tilde - f_x_flattened))
        Error_fxx[simuIndex] = np.max(abs(massed_f_xx_tilde - f_xx_flattened))


        if simuIndex > 0 :
            print("")
            print("nombre de points par côté : ", n)
            print("Erreurs opérations sur la dérivée : ", Error_fx[simuIndex], " || pente : ", (np.log(Error_fx[simuIndex]) - np.log(Error_fx[simuIndex-1])) / (np.log(hList[simuIndex]) - np.log(hList[simuIndex-1])))
            print("Erreurs opérations sur la dérivée seconde : ", Error_fxx[simuIndex], " || pente : ", (np.log(Error_fxx[simuIndex]) - np.log(Error_fxx[simuIndex-1])) / (np.log(hList[simuIndex]) - np.log(hList[simuIndex-1])))
            print("")
        else :
            print("")
            print("nombre de points par côté : ", n)
            print("Erreurs opérations sur la dérivée : ", Error_fx[simuIndex])
            print("Erreurs opérations sur la dérivée seconde : ", Error_fxx[simuIndex])
            print("")
        
        simuIndex += 1

    # regression linéaire sur les droites de convergence :
    # erreurs sur f_x
    X_fx = np.ones((len(nList), 2))
    X_fx[:, 1] = np.log(hList)
    X_fx_T = np.transpose(X_fx)
    A_fx = np.linalg.solve(X_fx_T @ X_fx, X_fx_T @ np.log(Error_fx))

    # erreurs sur f_xx
    X_fxx = np.ones((len(nList), 2))
    X_fxx[:, 1] = np.log(hList)
    X_fxx_T = np.transpose(X_fxx)
    A_fxx = np.linalg.solve(X_fxx_T @ X_fxx, X_fxx_T @ np.log(Error_fxx))

    print("Regressions linéaires sur les données : ")
    print(" - pente pour la dérivée : ", A_fx[1])
    print(" - pente pour la dérivée seconde : ", A_fxx[1])
    
    with PdfPages(f"Plots/divF/Test_{functionChoice}_Conv_{basisChoice}_P{spacePolynOrder}_{modificationChoice}") as pdf:
        # Résultats des opérations sur la première composante   
        plt.figure()
        plt.title(f"Erreurs en norme $L^{{\\infty}}$ sur $\\partial_x$ pour avec {basisChoice} P{spacePolynOrder} {modificationChoice} \n \
                  et {functionDef}")
        plt.plot(hList, Error_fx, label="$\\left\\| \\partial_x f - \\widetilde{\\partial_x} f \\right\\|_{{\\infty}}$",marker="o")
        plt.plot(hList, np.exp(A_fx[0]) * hList**A_fx[1], label=f"regression de pente {round(A_fx[1], 3)}", linestyle="--", alpha=0.5)
        plt.xlabel("h")
        plt.ylabel("error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()

        # Résultats des opérations sur la seonde composante
        plt.figure()
        plt.title(f"Erreurs en norme $L^{{\\infty}}$ sur $\\partial_x^2$ pour avec {basisChoice} P{spacePolynOrder} {modificationChoice}  \n \
                  et {functionDef}")
        plt.plot(hList, Error_fxx, label="$\\left\\| \\partial_x^2 f - \\widetilde{\\partial_x^2} f \\right\\|_{{\\infty}}$", marker="o")
        plt.plot(hList, np.exp(A_fxx[0]) * hList**A_fxx[1], label=f"regression de pente {round(A_fxx[1])}", linestyle="--", alpha=0.5)
        plt.xlabel("h")
        plt.ylabel("error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()

