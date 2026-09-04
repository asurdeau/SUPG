# Modules généraux
import numpy as np
import yaml
import math
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

# Modules perso
import config
from config import XVEL, YVEL, PRES
import grid_mod
import spatial_operators
import solutions
from spatial_operators import getApproxDivFlux



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



####################################################################################################################

#                                TESTING FUNCTIONS FOR EACH INDIVIDUAL COMPONENT                                   #

####################################################################################################################

# U has x-first-derivative on the last component and x-second-derivative on the first component

def first_component_operations_errors(q, iMin, iMax, jMin, jMax, dx, dy, divFlux_method, f_x, f_xx, f_xy, \
                                      grid, schemeChoice, operatorsCoeffs):
    
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx = grid.steps[0]

    divFlux = divFlux_method(q, grid, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, operatorsCoeffs)
    print("test periodicité divFlux", grid.isPeriodic(divFlux))
    print("ecart divFlux bandes horizontales : ", np.max(divFlux[iMin, jMin:jMax+1] - divFlux[iMax, jMin:jMax+1]))
    print("ecart divFlux bandes verticales : ", np.max(divFlux[iMin:iMax+1, jMin] - divFlux[iMin:iMax+1, jMax]))

    first_component_error = np.max(abs(- 2. / dx * divFlux[iMin:iMax+1, jMin:jMax+1, 0] - f_xx[iMin:iMax+1, jMin:jMax+1]))

    second_component_error = np.max(abs(- 2. / dx * divFlux[iMin:iMax+1, jMin:jMax+1, 1] - f_xy[iMin:iMax+1, jMin:jMax+1]))

    third_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, 2] - f_x[iMin:iMax+1, jMin:jMax+1]))

    return first_component_error, second_component_error, third_component_error



def second_component_operations_errors(q, iMin, iMax, jMin, jMax, dx, dy, divFlux_method, f_y, f_yy, f_xy, \
                                       grid, schemeChoice, operatorsCoeffs):
    iMin, iMax, jMin, jMax = grid.valid_grid
    dy = grid.steps[1]

    divFlux = divFlux_method(q, grid, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, operatorsCoeffs)

    first_component_error = np.max(abs(- 2. / dy * divFlux[iMin:iMax+1, jMin:jMax+1, 0] - f_xy[iMin:iMax+1, jMin:jMax+1]))

    second_component_error = np.max(abs(- 2. / dy * divFlux[iMin:iMax+1, jMin:jMax+1, 1] - f_yy[iMin:iMax+1, jMin:jMax+1]))

    third_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, 2] - f_y[iMin:iMax+1, jMin:jMax+1]))

    return first_component_error, second_component_error, third_component_error



def third_component_operations_errors(q, iMin, iMax, jMin, jMax, dx, dy, divFlux_method, f_x, f_y, f_xx, f_yy, \
                                      grid, schemeChoice, operatorsCoeffs):
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx, dy = grid.steps

    divFlux = divFlux_method(q, grid, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, operatorsCoeffs)

    first_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, 0] - f_x[iMin:iMax+1, jMin:jMax+1]))

    second_component_error = np.max(abs(divFlux[iMin:iMax+1, jMin:jMax+1, 1] - f_y[iMin:iMax+1, jMin:jMax+1]))

    third_component_error = np.max(abs( (- 2. * divFlux[iMin:iMax+1, jMin:jMax+1, 2] \
                                         - dx * f_xx[iMin:iMax+1, jMin:jMax+1] \
                                         - dy * f_yy[iMin:iMax+1, jMin:jMax+1]) / max(dx, dy) ))

    return first_component_error, second_component_error, third_component_error



####################################################################################################################

#                                               INDIVIDUAL TESTS                                                   #

####################################################################################################################



# if (True) :
if (False) :
    # Creation of the grid_parameters dictionnary by hand
    xL, xR = 0., 1.
    yL, yR = 0., 1.
    Nx, Ny = 200, 200
    nGhost = 2
    schemeChoice = -1
    operators = yaml.load(open("operators.yaml"),Loader=yaml.SafeLoader)
    operatorsCoeffs = []
    iTest = 4

    grid_params = {"domain_parameters" : {"xL" : xL, "xR" : xR, "yL" : yL, "yR" : yR}, "mesh_parameters" : {"Nx" : Nx, "Ny" : Ny, "nGhost" : nGhost}}
    params = {"grid_parameters" : grid_params, "order" : 0.}
    
    gridOp = grid_mod.gridOperator(params)
    iMin, iMax, jMin, jMax = gridOp.valid_grid
    dx, dy = gridOp.steps
    X = gridOp.xValidGrid
    Y = gridOp.yValidGrid

    if schemeChoice == 3:
        operatorsCoeffs = spatial_operators.extract_operators(operators)

        if iTest == 1: ### TEST Identity :
            for i in range(20):
                operatorsCoeffs[i] = [0., 1., 0.]
        elif iTest == 2: ### TEST : x operators (y operators = Id)
            for i in range(10):
                operatorsCoeffs[2 * i + 1] = [0., 1., 0.]
        elif iTest == 3: ### TEST : y operators (x operators = Id)
            for i in range(10):
                operatorsCoeffs[2 * i] = [0., 1., 0.]

    elif schemeChoice == 4:
            operatorsCoeffs = spatial_operators.extract_operators_improved(dx, dy)
    
            if iTest == 1: ### TEST Identity :
                for i in range(20):
                    operatorsCoeffs[i] = [0., 1., 0.]
            elif iTest == 2: ### TEST : x operators (y operators = Id)
                for i in range(10):
                    operatorsCoeffs[2 * i + 1] = [0., 1., 0.]
            elif iTest == 3: ### TEST : y operators (x operators = Id)
                for i in range(10):
                    operatorsCoeffs[2 * i] = [0., 1., 0.]

    else : 
        iTest = 4
    


    # SLICES DE REFERENCE
    # INJECTION MAILLAGE CALCUL (x_i^k) DANS MAILLAGE RAFFINE (ksi_alpha^gamma) :
    # x_i^k = ksi_{i * n_raffinement + k * n_raffinement // ordre}^{k * n_raffinement % ordre}
    n_raffinement = 10

    ### SLICE HORIZONTAL
    NxRef = n_raffinement * Nx
    Xref = np.linspace(xL, xR, NxRef+1)
    j_ref, l_ref = int(Ny / 2.), 0

    ### SLICE VERTICAL
    NyRef = n_raffinement * Ny
    Yref = np.linspace(yL, yR, NyRef+1)
    i_ref, k_ref = int(Nx / 2.), 0

    vertical_index_ref_mesh = j_ref * n_raffinement
    y_ref = Yref[vertical_index_ref_mesh]
    horizontal_index_ref_mesh = i_ref * n_raffinement
    x_ref = Xref[horizontal_index_ref_mesh]

    XXref, YYref = np.meshgrid(Xref, Yref)
    XXref = np.transpose(XXref)
    YYref = np.transpose(YYref)

    paramRef = params


    testFonction = test5
    f = testFonction(X, Y)[0]
    f_ref_mesh, f_x_ref_mesh, f_y_ref_mesh, f_xx_ref_mesh, f_yy_ref_mesh, f_xy_ref_mesh = \
        testFonction(XXref, YYref)

    with PdfPages(f"Plots/ordre 1/fonction_test") as pdf:
        # Slice horizontal
        plt.figure()
        plt.title(f"Slice horizontal [0, 1]x{{{y_ref}}} de la fonction test ")
        plt.plot(X[iMin:iMax+2, j_ref], f[iMin:iMax+2, j_ref])
        pdf.savefig()
        plt.close()

        # Slice vertical
        plt.figure()
        plt.title(f"Slice horizontal [0, 1]x{{{y_ref}}} de la fonction test ")
        plt.plot(Y[i_ref, jMin:jMax+2], f[i_ref, jMin:jMax+2])
        pdf.savefig()
        plt.close()

        # figure 2D
        plt.figure()
        cf = plt.contourf(X, Y, f, levels=10, cmap="plasma")
        plt.contour(X[iMin:iMax+2, jMin:jMax+2], Y[iMin:iMax+2, jMin:jMax+2], f[iMin:iMax+2, jMin:jMax+2], levels=10, colors="k", linewidths=0.3)
        plt.contourf(X[iMin:iMax+2, jMin:jMax+2], Y[iMin:iMax+2, jMin:jMax+2], f[iMin:iMax+2, jMin:jMax+2], levels=10)
        plt.colorbar(cf, format="%.2f")
        plt.title(f"Plot 2D de la fonction test")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        pdf.savefig()
        plt.close()


    # test periodicité de f :
    print("test périodicité de f : ")
    print("écarts bords verticaux : ", np.max(abs(f[iMin, jMin:jMax+1] - f[iMax+1, jMin:jMax+1])))
    print("écarts bords verticaux : ", np.max(abs(f[iMin:iMax+1, jMin] - f[iMin:iMax+1, jMax+1])), "\n")


    q1, q2, q3 = np.zeros((3, Nx + 2*nGhost, Ny + 2*nGhost, 3))
    q1[:, :, 0] = f
    q2[:, :, 1] = f
    q3[:, :, 2] = f


    divFlux1 = getApproxDivFlux(q1, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, operators, operatorsCoeffs)[iMin:iMax+2, jMin:jMax+2]
    divFlux2 = getApproxDivFlux(q2, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, operators, operatorsCoeffs)[iMin:iMax+2, jMin:jMax+2]
    divFlux3 = getApproxDivFlux(q3, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, operators, operatorsCoeffs)[iMin:iMax+2, jMin:jMax+2]


    f = f_ref_mesh[::n_raffinement, ::n_raffinement]
    f_x = f_x_ref_mesh[::n_raffinement, ::n_raffinement]
    f_xx = f_xx_ref_mesh[::n_raffinement, ::n_raffinement]
    f_y = f_y_ref_mesh[::n_raffinement, ::n_raffinement]
    f_yy = f_yy_ref_mesh[::n_raffinement, ::n_raffinement]
    f_xy = f_xy_ref_mesh[::n_raffinement, ::n_raffinement]


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
        
        divFluxes = np.array([[2. * divFlux1[:, :, XVEL], 2. * divFlux2[:, :, XVEL], divFlux3[:, :, XVEL]], \
                                [2. * divFlux1[:, :, YVEL], 2. * divFlux2[:, :, YVEL], divFlux3[:, :, YVEL]], \
                                [divFlux1[:, :, PRES], divFlux2[:, :, PRES], divFlux3[:, :, PRES]]])


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
        
        divFluxes = np.array([[- 2. * divFlux1[:, :, XVEL] / dx, - 2. * divFlux2[:, :, XVEL] / dx,      divFlux3[:, :, XVEL]], \
                                [  2. * divFlux1[:, :, YVEL]     ,   2. * divFlux2[:, :, YVEL]     ,      divFlux3[:, :, YVEL]], \
                                [       divFlux1[:, :, PRES]     ,        divFlux2[:, :, PRES]     , 2. * divFlux3[:, :, PRES]]])
            


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
        
        divFluxes = np.array([[  2. * divFlux1[:, :, XVEL]     ,   2. * divFlux2[:, :, XVEL]     ,      divFlux3[:, :, XVEL]], \
                                [- 2. * divFlux1[:, :, YVEL] / dy, - 2. * divFlux2[:, :, YVEL] / dy,      divFlux3[:, :, YVEL]], \
                                [       divFlux1[:, :, PRES]     ,        divFlux2[:, :, PRES]     , 2. * divFlux3[:, :, PRES]]])


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
        
        divFluxes = np.array([[- 2. * divFlux1[:, :, XVEL] / dx, - 2. * divFlux2[:, :, XVEL] / dx,      divFlux3[:, :, XVEL]], \
                                [- 2. * divFlux1[:, :, YVEL] / dy, - 2. * divFlux2[:, :, YVEL] / dy,      divFlux3[:, :, YVEL]], \
                                [       divFlux1[:, :, PRES]     ,        divFlux2[:, :, PRES]     , - 2. * divFlux3[:, :, PRES] / max(dx, dy)]])



    # ECARTS 
    for equationName, equation in var:
        print(f"Opérations de l'équation sur {equationName} : ")
        for termName, term in var:
            print(f"   sur {termName} : ", np.max(abs(divFluxes[equation, term] - exact_operators[equation, term])))
        print("")


    # PLOTS
    if True :
    # if False :
        vertical_index_ref_mesh = j_ref * n_raffinement
        y_ref = Yref[vertical_index_ref_mesh]
        horizontal_index_ref_mesh = i_ref * n_raffinement
        x_ref = Xref[horizontal_index_ref_mesh]

        for equationName, equation in var:
            with PdfPages(f"Plots/ordre 1/1D_slices_{equationName}_equation_{schemeChoice}") as pdf:
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
                    plt.plot(Xref[::n_raffinement], approxOperator[:, j_ref], label="approx")
                    plt.legend()
                    pdf.savefig()
                    plt.close()
        
                    # Slice vertical
                    plt.figure()
                    plt.title(f"Slice vertical {{{x_ref}}}x[0, 1]: {exact_operator_name}{termName} dans l'équation en {equationName}")
                    plt.plot(Yref, vertical_slice, label="exact")
                    plt.plot(Yref[::n_raffinement], approxOperator[i_ref, :], label="approx")
                    plt.legend()
                    pdf.savefig()
                    plt.close()



####################################################################################################################

#                                              CONVERGENCE TESTS                                                   #

####################################################################################################################



if (True) :
# if (False) :
    # Creation of the grid_parameters dictionnary by hand
    xL, xR = 0., 1.
    yL, yR = 0., 1.
    nGhost = 2
    order = 0
    grid_params = {"domain_parameters" : {"xL" : xL, "xR" : xR, "yL" : yL, "yR" : yR}, "mesh_parameters" : {"Nx" : 0, "Ny" : 0, "nGhost" : nGhost}}
    params = {"grid_parameters" : grid_params, "order" : order}
    schemeChoice = 4
    divFlux_method = getApproxDivFlux
    
    
    nList = [100, 200, 400, 800, 1600]
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
        operators = yaml.load(open("operators.yaml"),Loader=yaml.SafeLoader)
        operatorsCoeffs = []

        if schemeChoice == 3:
            operatorsCoeffs = spatial_operators.extract_operators(operators)

        elif schemeChoice == 4:
            operatorsCoeffs = spatial_operators.extract_operators_improved(dx, dy)

        hList[k] = dx


        xCoord = np.linspace(xL, xR, n+1, endpoint=True)
        yCoord = np.linspace(yL, yR, n+1, endpoint=True)
        leftGhostIndices = np.arange(-nGhost, 0)
        rightGhostIndices = np.arange(1, nGhost)

        xCoord = np.concatenate([xL + leftGhostIndices * dx, xCoord, xR + rightGhostIndices * dx])
        yCoord = np.concatenate([yL + leftGhostIndices * dy, yCoord, yR + rightGhostIndices * dy])

        X, Y = np.meshgrid(xCoord, yCoord, indexing="ij")


        f, f_x, f_y, f_xx, f_yy, f_xy = test4(X, Y)
        # print("test periodicité donnée : ", gridOp.isPeriodic(f))
        # print("écarts bords verticaux : ", np.max(abs(f[iMin, jMin:jMax+1] - f[iMax+1, jMin:jMax+1])))
        # print("écarts bords verticaux : ", np.max(abs(f[iMin:iMax+1, jMin] - f[iMin:iMax+1, jMax+1])), "\n")

        if (schemeChoice == 1) :
            f_xy[:, :] = 0.

        q1, q2, q3 = np.zeros((3, n + 2*nGhost, n + 2 * nGhost, 3))
        q1[:, :, 0] = f
        q2[:, :, 1] = f
        q3[:, :, 2] = f

        # FIRST COMPONENT OPERATIONS :
        Error_q1_1[k], Error_q1_2[k], Error_q1_3[k] = \
            first_component_operations_errors(q1, iMin, iMax, jMin, jMax, dx, dy, divFlux_method, f_x, f_xx, f_xy, \
                                              gridOp, schemeChoice, operatorsCoeffs)
        Error_q2_1[k], Error_q2_2[k], Error_q2_3[k] = \
            second_component_operations_errors(q2, iMin, iMax, jMin, jMax, dx, dy, divFlux_method, f_y, f_yy, f_xy, \
                                               gridOp, schemeChoice, operatorsCoeffs)
        Error_q3_1[k], Error_q3_2[k], Error_q3_3[k] = \
            third_component_operations_errors(q3, iMin, iMax, jMin, jMax, dx, dy, divFlux_method, f_x, f_y, f_xx, f_yy, \
                                              gridOp, schemeChoice, operatorsCoeffs)

        print("")
        print("nombre de points par côté : ", n)
        print("Erreurs opérations sur 1e compsante : ", Error_q1_1[k], Error_q1_2[k], Error_q1_3[k])
        print("Erreurs opérations sur 2nd compsante : ", Error_q2_1[k], Error_q2_2[k], Error_q2_3[k])
        print("Erreurs opérations sur 3e compsante : ", Error_q3_1[k], Error_q3_2[k], Error_q3_3[k])
        print("")

        k += 1



    cross_term_u = "$\\partial_y \\partial_x u$"
    cross_term_v = "$\\partial_x \\partial_y v$"
    if (schemeChoice == -1):
        schemeName = "TEST SUPG"
        schemeShort = "test_SUPG"
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
    elif (schemeChoice == 4):
        schemeName = "optim modified SUPG"
        schemeShort = "optModSUPG"
    


    
    with PdfPages(f"Plots/ordre 1/test_operators_{schemeShort}") as pdf:
        # Résultats des opérations sur la première composante
        plt.figure()
        plt.title("Erreurs en norme $L^{{\\infty}}$ sur les différentes composantes de $\\widetilde{{\\nabla}} f$ \n pour le schéma "+schemeName+" avec v = p = 0")
        plt.plot(hList, Error_q1_1, label="$\\partial^2_x u$", marker="o")
        plt.plot(hList, Error_q1_2, label=cross_term_u, marker="^")
        plt.plot(hList, Error_q1_3, label="$\\partial_x u$", marker="s")
        plt.plot(hList, ( max(Error_q1_1[0], Error_q1_2[0], Error_q1_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
        plt.plot(hList, ( max(Error_q1_1[0], Error_q1_2[0], Error_q1_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()
        plt.close()

        # Résultats des opérations sur la seonde composante
        plt.figure()
        plt.title("Erreurs en norme $L^{{\\infty}}$ sur les différentes composantes de $\\widetilde{{\\nabla}} f$ \n pour le schéma "+schemeName+" avec u = p = 0")
        plt.plot(hList, Error_q2_1, label=cross_term_v, marker="o")
        plt.plot(hList, Error_q2_2, label="$\\partial^2_y v$", marker="^")
        plt.plot(hList, Error_q2_3, label="$\\partial_y v$", marker="s")
        plt.plot(hList, ( max(Error_q2_1[0], Error_q2_2[0], Error_q2_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
        plt.plot(hList, ( max(Error_q2_1[0], Error_q2_2[0], Error_q2_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()
        plt.close()

        # Résultats des opérations sur la troisième composante
        plt.figure()
        plt.title("Erreurs en norme $L^{{\\infty}}$ sur les différentes composantes de $\\widetilde{{\\nabla}} f$ \n pour le schéma "+schemeName+" avec u = v = 0")
        plt.plot(hList, Error_q3_1, label="$\\partial_x p$", marker="o")
        plt.plot(hList, Error_q3_2, label="$\\partial_y p$", marker="^")
        plt.plot(hList, Error_q3_3, label="$\\Delta p$", marker="s")
        plt.plot(hList, ( max(Error_q3_1[0], Error_q3_2[0], Error_q3_3[0]) * (hList / hList[0])) / 2., label="pente 1", linestyle="--", alpha=0.5)
        plt.plot(hList, ( max(Error_q3_1[0], Error_q3_2[0], Error_q3_3[0]) * (hList / hList[0])**2) / 2., label="pente 2", linestyle="--", alpha=0.5)
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        pdf.savefig()
        plt.close()
