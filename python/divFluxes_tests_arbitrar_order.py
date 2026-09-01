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
from python.spatial_operators import getApproxDivFlux, get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator, get_arbitrar_order_SUPG_evolution_operator_coeffs
from plots import getVisualisationData



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
    # a, b, c, d, e = 0., 0., 0., 0., 3.

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


def test5(X, Y): # f(x, y) = sin(2 kx pi x) * sin(2 ky pi y)
    kx = 5.
    ky = 5.

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


def test6(X, Y): # x^a . (1-x)^b . y^c . (1-y)^d
    a, b, c, d = 1, 1, 1, 1

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



def getTestFonctionLagrangeArbitrarOrderInterpolation(xL, xR, yL, yR, Nx, Ny, nGhost, order, testFonction):
    dx, dy = (xR - xL) / (1. * Nx), (yR - yL) / (1. * Ny)

    fTab = np.zeros((6, Nx + 2 * nGhost, Ny + 2 * nGhost, order, order))
    iMin, jMin = nGhost, nGhost

    for i in range(Nx + 2 * nGhost):
        for j in range(Ny + 2 * nGhost):
            for k in range(order):
                for l in range(order):
                    x_ik = xL + (i - iMin + (1. * k) / order) * dx
                    y_jl = yL + (j - jMin + (1. * l) / order) * dy
                    fTab[:, i, j, k, l] = testFonction(x_ik, y_jl)

    return fTab

####################################################################################################################

#                                TESTING FUNCTIONS FOR EACH INDIVIDUAL COMPONENT                                   #

####################################################################################################################

# U has x-first-derivative on the last component and x-second-derivative on the first component

def first_component_operations_errors(q, iMin, iMax, jMin, jMax, dx, dy, x_der, x_second_der, xy_der, \
                                      grid, order, coeffs):
    
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx = grid.steps[0]

    # divFlux = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q, iMin, iMax, jMin, jMax, order, coeffs)

    divFlux = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q, iMin, iMax, jMin, jMax, order, coeffs)
    # print("test periodicité divFlux", grid.isPeriodic(divFlux))
    # print("ecart divFlux bandes horizontales : ", np.max(divFlux[iMin, jMin:jMax+1] - divFlux[iMax, jMin:jMax+1]))
    # print("ecart divFlux bandes verticales : ", np.max(divFlux[iMin:iMax+1, jMin] - divFlux[iMin:iMax+1, jMax]))

    first_component_error = np.max(abs(- 2. * divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 0] / dx - x_second_der[iMin:iMax+1, jMin:jMax+1, :, :]))

    second_component_error = np.max(abs(- 2. * divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 1] / dy - xy_der[iMin:iMax+1, jMin:jMax+1, :, :]))

    third_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 2] - x_der[iMin:iMax+1, jMin:jMax+1, :, :]))

    return first_component_error, second_component_error, third_component_error



def second_component_operations_errors(q, iMin, iMax, jMin, jMax, dx, dy, y_der, y_second_der, xy_der, \
                                       grid, order, coeffs):
    iMin, iMax, jMin, jMax = grid.valid_grid
    dy = grid.steps[1]

    divFlux = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q, iMin, iMax, jMin, jMax, order, coeffs)

    first_component_error = np.max(abs(- 2. * divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 0] / dx - xy_der[iMin:iMax+1, jMin:jMax+1, :, :]))

    second_component_error = np.max(abs(- 2. * divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 1] / dy - y_second_der[iMin:iMax+1, jMin:jMax+1, :, :]))

    third_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 2] - y_der[iMin:iMax+1, jMin:jMax+1, :, :]))

    return first_component_error, second_component_error, third_component_error



def third_component_operations_errors(q, iMin, iMax, jMin, jMax, dx, dy, x_der, y_der, x_second_der, y_second_der, \
                                      grid, order, coeffs):
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx, dy = grid.steps

    divFlux = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q, iMin, iMax, jMin, jMax, order, coeffs)

    first_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 0] - x_der[iMin:iMax+1, jMin:jMax+1, :, :]))

    second_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 1] - y_der[iMin:iMax+1, jMin:jMax+1, :, :]))

    third_component_error = np.max(abs( (- 2. * divFlux[iMin:iMax+1, jMin:jMax+1, :, :, 2] / max(dx, dy) \
                                         - x_second_der[iMin:iMax+1, jMin:jMax+1, :, :] \
                                         - y_second_der[iMin:iMax+1, jMin:jMax+1, :, :]) ))

    return first_component_error, second_component_error, third_component_error



####################################################################################################################

#                                               INDIVIDUAL TESTS                                                   #

####################################################################################################################



if (True) :
# if (False) :
    # Creation of the grid_parameters dictionnary by hand
    xL, xR = 0., 1.
    yL, yR = 0., 1.
    Nx, Ny = 100, 100
    nGhost = 2
    schemeChoice = 1
    order = 3
    NVis = order
    iTest = 2


    grid_params = {"domain_parameters" : {"xL" : xL, "xR" : xR, "yL" : yL, "yR" : yR}, "mesh_parameters" : {"Nx" : Nx, "Ny" : Ny, "nGhost" : nGhost}}
    params = {"grid_parameters" : grid_params, "order" : order, "number_visualisation_points" : NVis}

    gridOp = grid_mod.gridOperator(params)
    iMin, iMax, jMin, jMax = gridOp.valid_grid
    dx, dy = gridOp.steps
    coeffs = get_arbitrar_order_SUPG_evolution_operator_coeffs(order, dx, dy)

    Id = np.array([ np.zeros((order, order)), np.eye(order), np.zeros((order, order)) ])

    if iTest == 1: ### TEST Identity :
        for i in range(20):
            coeffs[i] = Id
    elif iTest == 2: ### TEST : x operators (y operators = Id)
        for i in range(10):
            coeffs[2 * i + 1] = Id
    elif iTest == 3: ### TEST : y operators (x operators = Id)
        for i in range(10):
            coeffs[2 * i] = Id
    


    # SLICES DE REFERENCE
    # INJECTION MAILLAGE CALCUL (x_i^k) DANS MAILLAGE RAFFINE (ksi_alpha^gamma) :
    # x_i^k = ksi_{i * n_raffinement + k * n_raffinement // ordre}^{k * n_raffinement % ordre}
    n_raffinement = 10

    ### SLICE HORIZONTAL
    NxRef = n_raffinement * Nx * order
    Xref = np.linspace(xL, xR, NxRef+1)
    j_ref, l_ref = int(Ny / 2.), 0

    ### SLICE VERTICAL
    NyRef = n_raffinement * Ny * order
    Yref = np.linspace(yL, yR, NyRef+1)
    i_ref, k_ref = int(Nx / 2.), 0

    vertical_index_ref_mesh = (j_ref * n_raffinement + (l_ref * n_raffinement) // order) * order + (l_ref * n_raffinement) % order
    y_ref = Yref[vertical_index_ref_mesh]
    horizontal_index_ref_mesh = (i_ref * n_raffinement + (k_ref * n_raffinement) // order) * order + (k_ref * n_raffinement) % order
    x_ref = Xref[horizontal_index_ref_mesh]

    XXref, YYref = np.meshgrid(Xref, Yref)
    XXref = np.transpose(XXref)
    YYref = np.transpose(YYref)

    paramRef = params


    testFonction = test6

    f_HO_mesh, f_x_HO_mesh, f_y_HO_mesh, f_xx_HO_mesh, f_yy_HO_mesh, f_xy_HO_mesh = getTestFonctionLagrangeArbitrarOrderInterpolation(xL, xR, yL, yR, Nx, Ny, nGhost, order, testFonction)
    f_ref_mesh, f_x_ref_mesh, f_y_ref_mesh, f_xx_ref_mesh, f_yy_ref_mesh, f_xy_ref_mesh = \
        testFonction(XXref, YYref)

    # # test periodicité de f :
    # print("test périodicité de f : ")
    # print("écarts bords verticaux : ", np.max(abs(f_HO_mesh[iMin, jMin:jMax+1, 0, :] - f_HO_mesh[iMax+1, jMin:jMax+1, 0, :])))
    # print("écarts bords verticaux : ", np.max(abs(f_HO_mesh[iMin:iMax+1, jMin, :, 0] - f_HO_mesh[iMin:iMax+1, jMax+1, :, 0])), "\n")

    f_HO_mesh_flattened = getVisualisationData(f_HO_mesh, params)[-1]
    f_y_HO_mesh_flattened = getVisualisationData(f_y_HO_mesh, params)[-1]


    q1, q2, q3 = np.zeros((3, Nx + 2*nGhost, Ny + 2*nGhost, order, order, 3))
    q1[:, :, :, :, 0] = f_HO_mesh
    q2[:, :, :, :, 1] = f_HO_mesh
    q3[:, :, :, :, 2] = f_HO_mesh


    divFlux1 = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q1, iMin, iMax, jMin, jMax, order, coeffs)
    divFlux2 = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q2, iMin, iMax, jMin, jMax, order, coeffs)
    divFlux3 = get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q3, iMin, iMax, jMin, jMax, order, coeffs)

    # print("test périodicité de divFlux1 : ")
    # print("écarts bords verticaux : ", np.max(abs(divFlux1[iMin, jMin:jMax+1, 0, :] - divFlux1[iMax+1, jMin:jMax+1, 0, :])))
    # print("écarts bords verticaux : ", np.max(abs(divFlux1[iMin:iMax+1, jMin, :, 0] - divFlux1[iMin:iMax+1, jMax+1, :, 0])), "\n")

    # Mise à plat des données
    divFlux1_2D = getVisualisationData(divFlux1, params)[-1]
    divFlux2_2D = getVisualisationData(divFlux2, params)[-1]
    divFlux3_2D = getVisualisationData(divFlux3, params)[-1]


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
        
        divFluxes = np.array([[2. * divFlux1_2D[:, :, XVEL], 2. * divFlux2_2D[:, :, XVEL], divFlux3_2D[:, :, XVEL]], \
                              [2. * divFlux1_2D[:, :, YVEL], 2. * divFlux2_2D[:, :, YVEL], divFlux3_2D[:, :, YVEL]], \
                              [divFlux1_2D[:, :, PRES], divFlux2_2D[:, :, PRES], divFlux3_2D[:, :, PRES]]])


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
        
        divFluxes = np.array([[- 2. * divFlux1_2D[:, :, XVEL] / dx, - 2. * divFlux2_2D[:, :, XVEL] / dx,      divFlux3_2D[:, :, XVEL]], \
                              [  2. * divFlux1_2D[:, :, YVEL]     ,   2. * divFlux2_2D[:, :, YVEL]     ,      divFlux3_2D[:, :, YVEL]], \
                              [       divFlux1_2D[:, :, PRES]     ,        divFlux2_2D[:, :, PRES]     , 2. * divFlux3_2D[:, :, PRES]]])
            


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
        
        divFluxes = np.array([[  2. * divFlux1_2D[:, :, XVEL]     ,   2. * divFlux2_2D[:, :, XVEL]     ,      divFlux3_2D[:, :, XVEL]], \
                              [- 2. * divFlux1_2D[:, :, YVEL] / dy, - 2. * divFlux2_2D[:, :, YVEL] / dy,      divFlux3_2D[:, :, YVEL]], \
                              [       divFlux1_2D[:, :, PRES]     ,        divFlux2_2D[:, :, PRES]     , 2. * divFlux3_2D[:, :, PRES]]])


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
                                    [f_x, f_y, f_xx + f_yy]])
        
        divFluxes = np.array([[- 2. * divFlux1_2D[:, :, XVEL] / dx, - 2. * divFlux2_2D[:, :, XVEL] / dx,      divFlux3_2D[:, :, XVEL]], \
                              [- 2. * divFlux1_2D[:, :, YVEL] / dy, - 2. * divFlux2_2D[:, :, YVEL] / dy,      divFlux3_2D[:, :, YVEL]], \
                              [       divFlux1_2D[:, :, PRES]     ,        divFlux2_2D[:, :, PRES]     , - 2. * divFlux3_2D[:, :, PRES] / max(dx, dy)]])



    # ECARTS 
    for equationName, equation in var:
        print(f"Opérations de l'équation sur {equationName} : ")
        for termName, term in var:
            print(f"   sur {termName} : ", np.max(abs(divFluxes[equation, term] - exact_operators[equation, term])))
        print("")


    # PLOTS
    if True :
    # if False :
        vertical_index_ref_mesh = (j_ref * n_raffinement + l_ref * n_raffinement // order) * order + (l_ref * n_raffinement) % order
        y_ref = Yref[vertical_index_ref_mesh]
        horizontal_index_ref_mesh = (i_ref * n_raffinement + k_ref * n_raffinement // order) * order + (k_ref * n_raffinement) % order
        x_ref = Xref[horizontal_index_ref_mesh]

        for equationName, equation in var:
            with PdfPages(f"Plots/1D_slices_{equationName}_equation_order_{order}") as pdf:
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
                    plt.plot(Xref[::n_raffinement], approxOperator[:, j_ref*order + l_ref], label="approx")
                    plt.legend()
                    pdf.savefig()
                    plt.close()
        
                    # Slice vertical
                    plt.figure()
                    plt.title(f"Slice vertical {{{x_ref}}}x[0, 1]: {exact_operator_name}{termName} dans l'équation en {equationName}")
                    plt.plot(Yref, vertical_slice, label="exact")
                    plt.plot(Yref[::n_raffinement], approxOperator[i_ref*order + k_ref, :], label="approx")
                    plt.legend()
                    pdf.savefig()
                    plt.close()




####################################################################################################################

#                                              CONVERGENCE TESTS                                                   #

####################################################################################################################



# if (True) :
# if (False) :
    # Creation of the grid_parameters dictionnary by hand
    xL, xR = 0., 1.
    yL, yR = 0., 1.
    nGhost = 2
    order = 3
    grid_params = {"domain_parameters" : {"xL" : xL, "xR" : xR, "yL" : yL, "yR" : yR}, "mesh_parameters" : {"Nx" : 0, "Ny" : 0, "nGhost" : nGhost}}
    params = {"grid_parameters" : grid_params, "order" : order}
    
    
    nList = [100, 200, 400, 800]
    hList = np.ones(len(nList))
    Error_q1_1, Error_q1_2, Error_q1_3, Error_q2_1, Error_q2_2, Error_q2_3, \
        Error_q3_1, Error_q3_2, Error_q3_3 = np.ones((9, len(nList)))

    k = 0
    for n in nList :

        grid_params["mesh_parameters"]["Nx"] = n
        grid_params["mesh_parameters"]["Ny"] = n
        gridOp = grid_mod.gridOperator(params)
        iMin, iMax, jMin, jMax = gridOp.valid_grid
        dx, dy = gridOp.steps
        coeffs = get_arbitrar_order_SUPG_evolution_operator_coeffs(order, dx, dy)

        # print("coeffs : \n ", coeffs)

        hList[k] = dx

        testFonction = test5

        f, f_x, f_y, f_xx, f_yy, f_xy = \
            getTestFonctionLagrangeArbitrarOrderInterpolation(xL, xR, yL, yR, n, n, nGhost, order, testFonction)


        q1, q2, q3 = np.zeros((3, iMax+nGhost+1, jMax+nGhost+1, order, order, 3))
        q1[:, :, :, :, 0] = f
        q2[:, :, :, :, 1] = f
        q3[:, :, :, :, 2] = f

        # FIRST COMPONENT OPERATIONS :
        Error_q1_1[k], Error_q1_2[k], Error_q1_3[k] = \
            first_component_operations_errors(q1, iMin, iMax, jMin, jMax, dx, dy, f_x, f_xx, f_xy, \
                                                gridOp, order, coeffs)
        Error_q2_1[k], Error_q2_2[k], Error_q2_3[k] = \
            second_component_operations_errors(q2, iMin, iMax, jMin, jMax, dx, dy, f_y, f_yy, f_xy, \
                                               gridOp, order, coeffs)
        Error_q3_1[k], Error_q3_2[k], Error_q3_3[k] = \
            third_component_operations_errors(q3, iMin, iMax, jMin, jMax, dx, dy, f_x, f_y, f_xx, f_yy, \
                                              gridOp, order, coeffs)

        print("")
        print("nombre de points par côté : ", n)
        print("Erreurs opérations sur 1e compsante : ", Error_q1_1[k], Error_q1_2[k], Error_q1_3[k])
        print("Erreurs opérations sur 2nd compsante : ", Error_q2_1[k], Error_q2_2[k], Error_q2_3[k])
        print("Erreurs opérations sur 3e compsante : ", Error_q3_1[k], Error_q3_2[k], Error_q3_3[k])
        print("")
        k += 1

    cross_term_u = "$\\partial_y \\partial_x u$"
    cross_term_v = "$\\partial_x \\partial_y v$"
    

    # Résultats des opérations sur la première composante
    plt.figure()
    plt.title(f"Erreurs en norme $L^{{\\infty}}$ sur les différentes composantes de $\\widetilde{{\\nabla}} f$ \n pour l'ordre {order} avec v = p = 0")
    plt.plot(hList, Error_q1_1, label="$\\partial^2_x u$", marker="o")
    plt.plot(hList, Error_q1_2, label=cross_term_u, marker="^")
    plt.plot(hList, Error_q1_3, label="$\\partial_x u$", marker="s")
    plt.plot(hList, ( max(Error_q1_1[0], Error_q1_2[0], Error_q1_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
    plt.plot(hList, ( max(Error_q1_1[0], Error_q1_2[0], Error_q1_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"Plots/test_operators_1_ordrer_{order}")

    # Résultats des opérations sur la seonde composante
    plt.figure()
    plt.title(f"Erreurs en norme $L^{{\\infty}}$ sur les différentes composantes de $\\widetilde{{\\nabla}} f$ \n pour l'ordre {order} avec u = p = 0")
    plt.plot(hList, Error_q2_1, label=cross_term_v, marker="o")
    plt.plot(hList, Error_q2_2, label="$\\partial^2_y v$", marker="^")
    plt.plot(hList, Error_q2_3, label="$\\partial_y v$", marker="s")
    plt.plot(hList, ( max(Error_q2_1[0], Error_q2_2[0], Error_q2_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
    plt.plot(hList, ( max(Error_q2_1[0], Error_q2_2[0], Error_q2_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"Plots/test_operators_2_ordrer_{order}")

    # Résultats des opérations sur la troisième composante
    plt.figure()
    plt.title(f"Erreurs en norme $L^{{\\infty}}$ sur les différentes composantes de $\\widetilde{{\\nabla}} f$ \n pour l'ordre {order} avec u = v = 0")
    plt.plot(hList, Error_q3_1, label="$\\partial_x p$", marker="o")
    plt.plot(hList, Error_q3_2, label="$\\partial_y p$", marker="^")
    plt.plot(hList, Error_q3_3, label="$\\Delta p$", marker="s")
    plt.plot(hList, ( max(Error_q3_1[0], Error_q3_2[0], Error_q3_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
    plt.plot(hList, ( max(Error_q3_1[0], Error_q3_2[0], Error_q3_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"Plots/test_operators_3_ordrer_{order}")
