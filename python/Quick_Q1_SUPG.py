# Modules généraux
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import yaml
import math
from collections import defaultdict
import time
import tracemalloc


# Modules perso
import config
from config import XVEL, YVEL, PRES
import grid_mod
import schemes
import plots
from plots import makeSolutionsPlots, openGifWriters, closeGifWriters, openPdfWriters, closePdfWriters, makeNormPlots, makeConvTestPlots
from plots import getSchemeTitles
import solutions


####################################################################################################################

#                                         ROUTINE D'EVOLUTION EN TEMPS                                             #

####################################################################################################################



def getOneApproximateSolution(params):
    # DATA extraction from yaml file
    operators = yaml.load(open("operators.yaml"),Loader=yaml.SafeLoader)
    operatorsCoeff = schemes.extract_operators(operators)

    ### grid parameters
    grid_params = params["grid_parameters"]
    Nx, Ny, nGhost = grid_params["mesh_parameters"].values()
    simulationChoice = params["solution_parameters"]["simulation_choice"]
    if simulationChoice == 7 :
        params["plot_parameters"]["nb_plots"] = 1

    ### time parameters
    Tf, CFL = params["time_parameters"].values()

    ### plot parameters
    nPlots = params["plot_parameters"]["nb_plots"]
    do_pdf_plot = (params["plot_parameters"]["do_pdf_plot"] == "y")
    do_gif_plot = (params["plot_parameters"]["do_gif_plot"] == "y")

    # scheme choice :
    schemeChoice = params["scheme_choice"]

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
    q[iMin:iMax+1, jMin:jMax+1] = solutions.getSolution(currentTime, gridOp, params["solution_parameters"])
    
    # Boundary conditions 
    if (simulationChoice == 6 or simulationChoice == 7):        # vortex simulation -> Dirichlet BC
        gridOp.dirichlet(q)
    else :                                                      # else (default) -> periodic BC 
        gridOp.periodize(q) 
        print("test periodicité donnée initiale : ", gridOp.isPeriodic(q))

    
    ### initial observables
    nPlotsDone = 1.
    doPlot = False

    nb_iter = 2 * int(Tf / dt_apriori) # taking a margin for the total number of iterations
    divFluxTimeTable = - np.ones(nb_iter)
    updateTimeTable = - np.ones(nb_iter)

    i_obs = params["plot_parameters"]["observables"]
    if (i_obs == 1 or i_obs == 2):
        pdf_writers = openPdfWriters(params)
        gif_writers = openGifWriters(params)
        if (nPlots > 1) :
            q_valid = q[iMin:iMax+1, jMin:jMax+1] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)
            makeSolutionsPlots(q_valid, currentTime, params, gridOp, pdf_writers, gif_writers) # plots : pdf et/ou gif
        else :
            do_gif_plot = False

    elif (i_obs == 3): # Sup norm of the solution (stability purposes)
        sup_norms = (-1.) * np.ones((nb_iter, 3))
        sup_norms[0] = np.max(abs(q[iMin:iMax+1, jMin:jMax+1]), axis=(0, 1))
    
    elif (i_obs == 4): # Sup errors (consistency / convergence purposes) 
        sup_errors = (-1.) * np.ones((nb_iter, 3))
        sup_errors[0] = 0.

    elif (i_obs == 6): # divFlux differences
        schemeChoiceComparison = params["plot_parameters"]["comparison_scheme"]
        divFlux_diff = (-1.) * np.ones(nb_iter)


    if nPlots > 1 :
        timeBetweenPlots = Tf / (1. * (nPlots - 1))
    else : # Only eventual plot is at the end : "currentTime + dt_apriori > nPlotsDone * timeBetweenPlots" never true
        timeBetweenPlots = 2 * Tf



    #############################################################################################################



    # TIME LOOP : AVOID CLASS / DICTIONNARY LOOKUPS !
    # starting time of the loop
    startTimeLoop = time.time()
    while (abs(currentTime - Tf) > 1.e-10) :
        # currentTime += dt     # : accumule les erreurs d'arrondis !!!!
        # currentTime = i * dt  # : le meilleur choix quand le pas est constant 
        # historique_dt.append(dt)
        # currentTime = math.fsum(historique_dt) # recalcule la somme à chaque fois mais plus précis
        dt = dt_apriori
        if (i_obs == 1 or i_obs == 2):
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
        else :
            if (currentTime + dt_apriori > Tf) :
                dt = currentTime - Tf
                currentTime = Tf
            else :
                currentTime += dt_apriori

        # Div . Fluxes computations
        t0 = time.time()
        divF = schemes.getApproxDivFlux(q, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, operators, operatorsCoeff)
        t1 = time.time()

        divFluxTimeTable[i] = t1 - t0


        # Update
        t2 = time.time()
        q[iMin:iMax+1, jMin:jMax+1] += - dt * divF[iMin:iMax+1, jMin:jMax+1]
        t3 = time.time()

        updateTimeTable[i] = t3 - t2


        # Boundary conditions 
        if (simulationChoice == 6 or simulationChoice == 7):     # vortex simulation : Dirichlet BC
            gridOp.dirichlet(q)
        else :                                                   # else (default) : periodic BC 
            gridOp.periodize(q) 


        # observables intermédiaires éventuels
        if (i_obs == 1 or i_obs == 2):
            if (doPlot and (do_pdf_plot or do_gif_plot)) :
                q_valid = q[iMin:iMax+1, jMin:jMax+1] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)
                makeSolutionsPlots(q_valid, currentTime, params, gridOp, pdf_writers, gif_writers) # plots : pdf et/ou gif
                nPlotsDone += 1
                doPlot = False
                print("Plot ! temps de simulation :"+str(currentTime))

        elif (i_obs == 3): # Sup norm of the solution (stability purposes)
            sup_norms[i+1] = np.max(abs(q[iMin:iMax+1, jMin:jMax+1]), axis=(0, 1))
        
        elif (i_obs == 4): # Sup errors (consistency / convergence purposes) 
            q_exact = solutions.getSolution(currentTime, gridOp, params["solution_parameters"])
            sup_errors[i+1] = np.max(abs(q[iMin:iMax+1, jMin:jMax+1] - q_exact), axis=(0, 1))

        elif (i_obs == 6): # Div Fluxes comparisons
            divF_comparaison = schemes.getApproxDivFlux(q, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoiceComparison, operators, operatorsCoeff)
            divFlux_diff[i] = np.max(abs(divF - divF_comparaison))

        i += 1
    

    
    # end time of loop
    endTimeLoop = time.time()
    print("Temps de calcul boucle : ", endTimeLoop - startTimeLoop)
    print("type de la sortie q : ", q.dtype)
    print("moyenne pour le calcul de divFLux  : ", np.sum(divFluxTimeTable[divFluxTimeTable > 0]) / i )
    print("moyenne pour la mise à jour de q   : ", np.sum(updateTimeTable[updateTimeTable > 0]) / i )
    if (i_obs == 6):
        print("ecart flux                         : ", np.max(divFlux_diff))
    print(" ")


    #############################################################################################################


    # FINAL TIME REACHED : last observation
    if (i_obs == 1 or i_obs == 2):
        if (nPlots >= 1 and (do_pdf_plot or do_gif_plot)) :
            q_valid = q[iMin:iMax+1, jMin:jMax+1] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)
            makeSolutionsPlots(q_valid, currentTime, params, gridOp, pdf_writers, gif_writers) # plots : pdf et/ou gif
            closePdfWriters(pdf_writers)
            closeGifWriters(gif_writers)

    elif (i_obs == 3): # Sup norm of the solution (stability purposes)
        makeNormPlots(sup_norms, i, params, gridOp)
    
    elif (i_obs == 4): # Sup errors (consistency / convergence purposes) 
        makeNormPlots(sup_errors, i, params, gridOp)
    

    return q[iMin:iMax+1, jMin:jMax+1]



####################################################################################################################

#                                                 AUTRES ROUTINES                                                  #

####################################################################################################################



def getConvergenceTest(nList, params):
    
    newParams = params.copy()
    finalTime = params["time_parameters"]["end_time"]
    supErrorsList = (-1.) * np.ones((len(nList), 3))
    L2ErrorsList = (-1.) * np.ones((len(nList), 3))
    plotLoc = newParams["plot_parameters"]["plot_loc"]
    xL, xR, yL, yR = params["grid_parameters"]["domain_parameters"].values()

    hList = max(xR - xL, yR - yL) / (np.array(nList) - 1)

    i = 0
    for n in nList :
        newParams["grid_parameters"]["mesh_parameters"]["Nx"] = n
        newParams["grid_parameters"]["mesh_parameters"]["Ny"] = n

        print("calcul de solution pour h = "+str(hList[i])+" || n = "+str(n))
        q = getOneApproximateSolution(newParams)

        grid_params = newParams["grid_parameters"]
        gridOp = grid_mod.gridOperator(grid_params)
        q_exact = solutions.getSolution(finalTime, gridOp, params["solution_parameters"])
        
        supErrorsList[i] = np.max(abs(q - q_exact))
        dx, dy = gridOp.steps
        L2ErrorsList[i] = dx * dy * np.sum( (q - q_exact)**2 )


        # Plotting successive figures
        # if (True) :
        if (False) :
            plots = [
                (XVEL, "U", "Vitesse U", "RdBu_r"),
                (YVEL, "V", "Vitesse V", "RdBu_r"),
                (PRES, "P", "Pression",  "viridis"),
            ]

            numLevels = params["plot_parameters"]["levels"]
            X = gridOp.xValidGrid
            Y = gridOp.yValidGrid

            schemeShort, schemeName = getSchemeTitles(params)

            for var, filename, title, cmap in plots:
                fig, ax = plt.subplots()
                cf = ax.contourf(X, Y, q[:, :, var], levels=numLevels, cmap=cmap)
                ax.contour(X, Y, q[:, :, var], levels=numLevels, colors="k", linewidths=0.3)
                # plt.colorbar(cf, ax=ax, label=title, format="%.2f")
                plt.colorbar(cf, ax=ax, label=title)
                ax.set_title(f"{title} avec le schéma "+schemeName+f" à t={round(finalTime, 1)} \n avec "+str(n)+" points")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_aspect("equal")
                plt.tight_layout()
                plt.savefig(plotLoc+filename+"_"+schemeShort+"_"+str(n))
                
                plt.close()

        i += 1

    # Writing content to a file
    with open('results.txt', 'w') as f:
        for k in range(len(nList)):
            f.write(str(hList[k])+"     "+str(supErrorsList[k, 2])+"     "+str(L2ErrorsList[k, 2])+"\n")


    makeConvTestPlots(hList, supErrorsList, L2ErrorsList, params)




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