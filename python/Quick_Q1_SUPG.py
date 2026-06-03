# Modules généraux
import numpy as np
import matplotlib.pyplot as plt
import yaml
import math
from collections import defaultdict
import time

# Modules perso
import config
from config import XVEL, YVEL, PRES
import grid_mod
import schemes
from plots import makeSolutionsPlots, openGifWriters, closeGifWriters, openPdfWriters, closePdfWriters
import solutions


####################################################################################################################

#                                         ROUTINE D'EVOLUTION EN TEMPS                                             #

####################################################################################################################



def getOneApproximateSolution(params):
    # DATA extraction from yaml file
    operators = yaml.load(open("operators.yaml"),Loader=yaml.SafeLoader)

    ### grid parameters
    grid_params = params["grid_parameters"]
    Nx, Ny, nGhost = grid_params["mesh_parameters"].values()

    ### time parameters
    Tf, CFL = params["time_parameters"].values()

    ### plot parameters
    nPlots = params["plot_parameters"]["sol_plots"]["nb_plots"]
    do_pdf_plot = (params["plot_parameters"]["sol_plots"]["do_pdf_plot"] == "y")
    do_gif_plot = (params["plot_parameters"]["sol_plots"]["do_gif_plot"] == "y")

    # grid creation 
    gridOp = grid_mod.gridOperator(grid_params)
    iMin, iMax, jMin, jMax = gridOp.valid_grid
    dx, dy = gridOp.steps


    # INITIALISATION
    ### initialise time parameters
    i = 0
    currentTime = 0.0
    dt_apriori = min(dx, dy) * CFL

    ### Initialise data
    q = np.zeros((Nx + 2*nGhost, Ny + 2*nGhost, 3))
    q = solutions.getSolution(currentTime, gridOp.xGrid, gridOp.yGrid, params["solution_parameters"])        
    print("test periodicité donnée initiale : ", gridOp.isPeriodic(q))
    
    ### initial observables
    i_obs = params["plot_parameters"]["observables"]
    if (i_obs == 1 or i_obs == 2):
        if (nPlots > 1) :
            pdf_writers = openPdfWriters(params)
            gif_writers = openGifWriters(params)
            q_valid = q[iMin:iMax+1, jMin:jMax+1] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)
            makeSolutionsPlots(q_valid, currentTime, params, gridOp, pdf_writers, gif_writers) # plots : pdf et/ou gif
        else :
            do_gif_plot = False
    elif (i_obs == 3):
        nb_iter = 2 * int(Tf / dt_apriori)
        norm_sup_q = (-1.) * np.ones((nb_iter, 3))

    timeBetweenPlots = Tf / (1. * (nPlots - 1))
    nPlotsDone= 1.
    doPlot = False

    #############################################################################################################



    # TIME LOOP
    # starting time of the loop
    startTimeLoop = time.time()
    while (abs(currentTime - Tf) > 1.e-10) :
        i += 1

        # currentTime += dt     # : accumule les erreurs d'arrondis !!!!
        # currentTime = i * dt  # : le meilleur choix quand le pas est constant 
        # historique_dt.append(dt)
        # currentTime = math.fsum(historique_dt) # recalcule la somme à chaque fois mais plus précis
        dt = dt_apriori
        if (currentTime + dt_apriori > Tf) :
            dt = currentTime - Tf
            currentTime = Tf
            doPlot = False # The last plot is kept outside of the loop, switch to true if past inside
        elif (currentTime + dt_apriori > nPlotsDone * timeBetweenPlots) :
            dt = nPlotsDone * timeBetweenPlots - currentTime
            currentTime = nPlotsDone * timeBetweenPlots

            doPlot = True
        else :
            currentTime += dt_apriori


        # Div . Fluxes computations
        divF = schemes.getApproxDivFlux(q, gridOp, params["scheme_choice"], operators)


        # Update
        q[iMin:iMax+1, jMin:jMax+1] += - dt * divF[iMin:iMax+1, jMin:jMax+1]


        # Conditions périodiques :
        gridOp.periodize(q)


        # Plots éventuels
        if (doPlot and (do_pdf_plot or do_gif_plot)) :
            q_valid = q[iMin:iMax+1, jMin:jMax+1] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)
            makeSolutionsPlots(q_valid, currentTime, params, gridOp, pdf_writers, gif_writers) # plots : pdf et/ou gif
            nPlotsDone += 1
            doPlot = False
            print("Plot ! temps de simulation :"+str(currentTime))

        #    # Periodicity check :
        #    print("Periodicity test : ", gridOp.isPeriodic(q))
    
    
    # end time of loop
    endTimeLoop = time.time()
    print("Temps de calcul boucle : ", endTimeLoop - startTimeLoop)


    #############################################################################################################


    # FINAL TIME REACHED
    if (nPlots >= 1 and (do_pdf_plot or do_gif_plot)) :
        q_valid = q[iMin:iMax+1, jMin:jMax+1] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)
        makeSolutionsPlots(q_valid, currentTime, params, gridOp, pdf_writers, gif_writers) # plots : pdf et/ou gif
        closePdfWriters(pdf_writers)
        closeGifWriters(gif_writers)




####################################################################################################################

#                                                 AUTRES ROUTINES                                                  #

####################################################################################################################


# Spatial operators :

# xL, xR = 0., 1.
# yL, yR = 0., 1.
# Nx, Ny = 20, 20
# nGhost = 1 operators 
def mass_x(u, dx, iMin, iMax, jMin, jMax) :
    v = np.zeros(np.shape(u))
    v[iMin:iMax+1, jMin:jMax+1] = \
        dx / 6. * (u[iMin+1:iMax+1+1, jMin:jMax+1] \
                + 4.*u[iMin:iMax+1, jMin:jMax+1] \
                + u[iMin-1:iMax, jMin:jMax+1])
    return v

def centered_x_derivative(u, iMin, iMax, jMin, jMax) :
    v = np.zeros(np.shape(u))
    v[iMin:iMax+1, jMin:jMax+1] = \
        (u[iMin+1:iMax+1+1, jMin:jMax+1] \
        - u[iMin-1:iMax, jMin:jMax+1])
    return v

def second_x_derivative(u, dx, iMin, iMax, jMin, jMax) :
    v = np.zeros(np.shape(u))
    v[iMin:iMax+1, jMin:jMax+1] = \
        1./dx * (- u[iMin+1:iMax+1+1, jMin:jMax+1] \
                 + 2.*u[iMin:iMax+1, jMin:jMax+1] \
                 - u[iMin-1:iMax, jMin:jMax+1])
    return v


# y operators
def mass_y(u, dy, iMin, iMax, jMin, jMax) :
    v = np.zeros(np.shape(u))
    v[iMin:iMax+1, jMin:jMax+1] = \
        dy / 6. * (u[iMin:iMax+1, jMin+1:jMax+1+1] \
                + 4.*u[iMin:iMax+1, jMin:jMax+1] \
                + u[iMin:iMax+1, jMin-1:jMax])
    return v

def centered_y_derivative(u, iMin, iMax, jMin, jMax) :
    v = np.zeros(np.shape(u))
    v[iMin:iMax+1, jMin:jMax+1] = \
        (u[iMin:iMax+1, jMin+1:jMax+1+1] \
        - u[iMin:iMax+1, jMin-1:jMax])
    return v

def second_y_derivative(u, dy, iMin, iMax, jMin, jMax) :
    v = np.zeros(np.shape(u))
    v[iMin:iMax+1, jMin:jMax+1] = \
        1./dy * (- u[iMin:iMax+1, jMin+1:jMax+1+1] \
                 + 2.*u[iMin:iMax+1, jMin:jMax+1] \
                 - u[iMin:iMax+1, jMin-1:jMax])
    return v