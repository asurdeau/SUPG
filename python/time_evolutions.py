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
import spatial_operators
from spatial_operators import get_evaluation_of_arbitrar_order_SUPG_physical_terms
import time_operators
import plots
from plots import makeSolutionsPlots, openGifWriters, closeGifWriters, openPdfWriters, closePdfWriters, makeObservableTablePlots, makeConvTestPlots
from plots import getSchemeKeys
import solutions


####################################################################################################################

#                                         ROUTINE D'EVOLUTION EN TEMPS                                             #

####################################################################################################################



def getOneApproximateSolution(params, q_prep=[]):
    # PARAMETERS EXTRACTION
    ### grid parameters
    Nx, Ny, nGhost = params["grid_parameters"]["mesh_parameters"].values()

    ### time parameters
    finalTime, CFL = params["time_parameters"].values()

    ### plot parameters
    nbPlots = params["plot_parameters"]["nb_plots"]
    do_pdf_plot = (params["plot_parameters"]["do_pdf_plot"] == "y")
    do_gif_plot = (params["plot_parameters"]["do_gif_plot"] == "y")

    ### simulation choice
    simulationChoice = params["simulation_choice"]

    ### scheme choice
    schemeChoice = params["scheme_choice"]

    ### observables choice 
    i_obs_list = params["observables_choices"]
    do_obs_table = plots.getDoObservablesTable(i_obs_list)

    # INITIALISATION
    ### grid creation 
    gridOp = grid_mod.gridOperator(params)
    iMin, iMax, jMin, jMax = gridOp.valid_grid
    dx, dy = gridOp.steps

    ### initialise time parameters
    currentIteration = 0
    lastCompletedIteration = 0
    currentTime = 0.0
    taux_progression = 0.1
    dt_apriori = min(dx, dy) * CFL

    divFluxTimeTable, updateTimeTable = np.zeros((2, int(2. * finalTime / dt_apriori)))


    ### Initialise coefficients
    operators = yaml.load(open("operators.yaml"),Loader=yaml.SafeLoader)
    operatorsCoeff = spatial_operators.extract_operators(operators)
    operatorsCoeffTest = spatial_operators.extract_operators_improved(dx, dy)

    ### Simulation order
    order = params["order"]
    evolCoeffs = spatial_operators.get_arbitrar_order_SUPG__weighted_evolution_operator_coeffs(order, dx, dy)
    massCoeffs = spatial_operators.get_arbitrar_order_SUPG_weighted_mass_operator_coeffs(order, dx, dy)

    ### Initialise data
    # q = np.zeros((Nx + 2*nGhost, Ny + 2*nGhost, 3), order="F")
    q = np.zeros((Nx + 2*nGhost, Ny + 2*nGhost, 3))
    if simulationChoice == 4 :
        q_prep = solutions.getSolution(currentTime, gridOp, simulationChoice, params["solution_parameters"])
        q[iMin:iMax+1, jMin:jMax+1] = q_prep
        X = gridOp.xValidGrid
        Y = gridOp.yValidGrid
        q[iMin:iMax+1, jMin:jMax+1] = solutions.addPerturbation(q[iMin:iMax+1, jMin:jMax+1], X, Y, params["solution_parameters"])
    if (simulationChoice == 5):
        q[iMin:iMax+1, jMin:jMax+1] = q_prep
        X = gridOp.xValidGrid
        Y = gridOp.yValidGrid
        q[iMin:iMax+1, jMin:jMax+1] = solutions.addPerturbation(q[iMin:iMax+1, jMin:jMax+1], X, Y, params["solution_parameters"])
    else : 
        q[iMin:iMax+1, jMin:jMax+1] = solutions.getSolution(currentTime, gridOp, simulationChoice, params["solution_parameters"])

    if (any(np.array(i_obs_list) == 7)):
        physicalTerms = np.zeros_like(q)
    else :
        physicalTerms = []
    
    # Boundary conditions 
    if (simulationChoice == 6 or simulationChoice == 7):        # vortex simulation -> Dirichlet BC
        gridOp.dirichlet(q)
    else :                                                      # else (default) -> periodic BC 
        gridOp.periodize(q) 
        print("test periodicité donnée initiale : ", gridOp.isPeriodic(q))

    
    ### initial observables
    pdf_writers, gif_writers, do_gif_plot, doPlot, nbPlotsDone, timeBetweenPlots, normsTable, errorsTable, massesTable, physicalTermsNormsTable = \
        getInitialObservables(q, q_prep, finalTime, dt_apriori, gridOp, params, do_obs_table, simulationChoice, nbPlots)


    #############################################################################################################



    # TIME LOOP : AVOID CLASS / DICTIONNARY LOOKUPS !
    # starting time of the loop
    startTimeLoop = time.time()
    divFluxTimeTable2, updateTimeTable2 = np.zeros((2, int(2. * finalTime / dt_apriori)))
    schemesCompTable = - np.ones(int(2. * finalTime / dt_apriori))
    q1 = q.copy()
    q2 = q.copy()
    if (any(np.array(i_obs_list) == 7)): # WE HAVE TO COMPUTE THE PHYSICAL TERMS
        while (abs(currentTime - finalTime) > 1.e-10) :
        
            # AJUSTEMENT EVENTUEL DU PAS DE TEMPS ET MAJ DU TEMPS ACTUEL
            dt, currentTime, doPlot, taux_progression = getAdaptedTimeUpdate(dt_apriori, currentTime, finalTime, do_obs_table, nbPlotsDone, timeBetweenPlots, doPlot, taux_progression)

            # MISE A JOUR
            currentIteration += 1
            q = time_operators.getForwardEulerUpdate(q, dt, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, simulationChoice, operators, operatorsCoeff, \
                                                    divFluxTimeTable, updateTimeTable, currentIteration)

            # q = time_operators.getDecUpdate(q, dt, order, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs)

            lastCompletedIteration += 1

            # OBSERVABLES INTERMEDIAIRES EVENTUELLES
            doPlot, nbPlotsDone, normsTable, errorsTable, massesTable, physFactorsTable \
                = getIntermediateObservables(q, q_prep, physicalTerms, currentTime, lastCompletedIteration, gridOp, dx, dy, iMin, iMax, jMin, jMax, \
                                            normsTable, errorsTable, massesTable, physicalTermsNormsTable, params, do_obs_table, \
                                            pdf_writers, gif_writers, doPlot, nbPlotsDone, do_pdf_plot, do_gif_plot)
        
    else : 
        while (abs(currentTime - finalTime) > 1.e-10) :
            
            # AJUSTEMENT EVENTUEL DU PAS DE TEMPS ET MAJ DU TEMPS ACTUEL
            dt, currentTime, doPlot, taux_progression = getAdaptedTimeUpdate(dt_apriori, currentTime, finalTime, do_obs_table, nbPlotsDone, timeBetweenPlots, doPlot, taux_progression)
    
    
            # MISE A JOUR
            currentIteration += 1
            q = time_operators.getForwardEulerUpdate(q, dt, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, simulationChoice, operators, operatorsCoeff, \
                                                     divFluxTimeTable, updateTimeTable, currentIteration)
    
            # q = time_operators.getDecUpdate(q, dt, order, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs)

            # schemeChoice = 3
            # q1 = time_operators.getForwardEulerUpdate(q1, dt, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, simulationChoice, operators, operatorsCoeff, \
            #                                                     divFluxTimeTable, updateTimeTable, currentIteration)

            # schemeChoice = 4
            # q2 = time_operators.getForwardEulerUpdate(q2, dt, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, simulationChoice, operators, operatorsCoeffTest, \
            #                                                     divFluxTimeTable2, updateTimeTable2, currentIteration)
            # schemesCompTable[lastCompletedIteration] = np.max(abs(q1 - q2))
    
            lastCompletedIteration += 1
    
            # OBSERVABLES INTERMEDIAIRES EVENTUELLES
            doPlot, nbPlotsDone, normsTable, errorsTable, massesTable, physFactorsTable \
                = getIntermediateObservables(q, q_prep, physicalTerms, currentTime, lastCompletedIteration, gridOp, dx, dy, iMin, iMax, jMin, jMax, \
                                             normsTable, errorsTable, massesTable, physicalTermsNormsTable, params, do_obs_table, \
                                             pdf_writers, gif_writers, doPlot, nbPlotsDone, do_pdf_plot, do_gif_plot)




    #############################################################################################################



    # FIN DE LA BOUCLE EN TEMPS 
    ### Quelques prints
    endTimeLoop = time.time()
    print("Temps de calcul boucle : ", endTimeLoop - startTimeLoop)
    print("type de la sortie q : ", q.dtype)
    print("moyenne pour le calcul de divFLux  : ", np.sum(divFluxTimeTable[divFluxTimeTable > 0]) / lastCompletedIteration )
    print("moyenne pour la mise à jour de q   : ", np.sum(updateTimeTable[updateTimeTable > 0]) / lastCompletedIteration )

    # print("2e schéma : ")
    # print("moyenne pour le calcul de divFLux  : ", np.sum(divFluxTimeTable2[divFluxTimeTable2 > 0]) / lastCompletedIteration )
    # print("moyenne pour la mise à jour de q   : ", np.sum(updateTimeTable2[updateTimeTable2 > 0]) / lastCompletedIteration )
    # print("max de l'écart entre les deux schémas : ", np.max(schemesCompTable))

    # plt.figure()
    # plt.title("comparaisons entre modSUPG et optModSUPG sur données aléatoires \n de valeurs entre 1e-3 et 1.e3")
    # iterationsList = np.arange(1, lastCompletedIteration)
    # plt.plot(iterationsList, schemesCompTable[1:lastCompletedIteration])
    # plt.xlabel("iterations")
    # plt.ylabel("$||q_1 - q_2||$")
    # plt.savefig("Plots/Tests/Modified_SUPG/Comparaison_opt")
    # plt.show()
    # plt.close()

    ### Dernière observable éventuelle 
    getLastObservables(q, q_prep, currentTime, lastCompletedIteration, gridOp, normsTable, errorsTable, massesTable, physicalTermsNormsTable, \
                       params, order, evolCoeffs, do_obs_table, pdf_writers, gif_writers, nbPlots, do_pdf_plot, do_gif_plot)
    
    # FIN DE LA ROUTINE : ON RENVOIE LA VALEUR FINALE
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
    simulationChoice = params["simulation_choice"]

    hList = max(xR - xL, yR - yL) / (np.array(nList) - 1)

    i = 0
    for n in nList :
        newParams["grid_parameters"]["mesh_parameters"]["Nx"] = n
        newParams["grid_parameters"]["mesh_parameters"]["Ny"] = n

        print("calcul de solution pour h = "+str(hList[i])+" || n = "+str(n))
        q = getOneApproximateSolution(newParams)

        gridOp = grid_mod.gridOperator(newParams)
        q_exact = solutions.getSolution(finalTime, gridOp, simulationChoice, params["solution_parameters"])
        
        supErrorsList[i] = np.max(abs(q - q_exact), axis=(0, 1))
        dx, dy = gridOp.steps
        L2ErrorsList[i] = dx * dy * np.sqrt(np.sum( (q - q_exact)**2, axis=(0, 1)))


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
    outputs = [(XVEL, "U"), (YVEL, "V"), (PRES, "P")]
    for var, name in outputs :
        with open("results_"+name+".txt", 'w') as f:
            for k in range(len(nList)):
                f.write(str(hList[k])+"     "+str(supErrorsList[k, var])+"     "+str(L2ErrorsList[k, var])+"\n")


    makeConvTestPlots(hList, supErrorsList, L2ErrorsList, params)




####################################################################################################################

#                                                 AUTRES ROUTINES                                                  #

####################################################################################################################


def getAdaptedTimeUpdate(dt_apriori, currentTime, finalTime, do_obs_table, nbPlotsDone, timeBetweenbPlots, doPlot, taux_progression):
    if (do_obs_table[1] or do_obs_table[2] or do_obs_table[3]):
        if (currentTime + dt_apriori > finalTime) :
            dt = finalTime - currentTime
            currentTime = finalTime
            doPlot = False # The last plot is kept outside of the loop, switch to true if past inside
        elif (currentTime + dt_apriori > nbPlotsDone * timeBetweenbPlots) :
            dt = nbPlotsDone * timeBetweenbPlots - currentTime
            currentTime = nbPlotsDone * timeBetweenbPlots

            doPlot = True
        else :
            dt = dt_apriori
            currentTime += dt_apriori
    else :
        if (currentTime + dt_apriori > finalTime) :
            dt = finalTime - currentTime
            currentTime = finalTime
        else :
            dt = dt_apriori
            currentTime += dt_apriori

        if ( abs(currentTime - finalTime * taux_progression) < 0.5 * dt_apriori ):
            print("progression du calcul : " + str( 100 * round(taux_progression, 3) )+"%")
            taux_progression += 0.1
            

    return dt, currentTime, doPlot, taux_progression






def getInitialObservables(q, q_prep, finalTime, dt_apriori, grid, params, do_obs_table, simulationChoice, nbPlots):
    
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx, dy = grid.steps

    # Liste des sorties:
    pdf_writers = []
    gif_writers = []
    doPlot = False
    do_gif_plot = params["plot_parameters"]["do_gif_plot"]
    nbPlotsDone = 1
    normsTable = []
    errorsTable = []
    massesTable = []
    physicalTermsNormsTable = []
    timeBetweenPlots = 2 * finalTime # Default : Only eventual plot is at the end : "currentTime + dt_apriori > nbPlotsDone * timeBetweenPlots" never true

    currentTime = 0.
    a_priori_nb_iter = 2 * int(finalTime / dt_apriori)     # a priori number of iterations
    if (do_obs_table[1] or do_obs_table[2] or do_obs_table[3]):
        pdf_writers = openPdfWriters(do_obs_table, params)
        gif_writers = openGifWriters(do_obs_table, params)

        if (nbPlots > 1) :
            q_valid = q[iMin:iMax+1, jMin:jMax+1] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)
            if do_obs_table[1] :
                i_obs = 1
                makeSolutionsPlots(q_valid, [], currentTime, i_obs, params, grid, pdf_writers, gif_writers) # plots : pdf et/ou gif
            if do_obs_table[2] :
                i_obs = 2
                makeSolutionsPlots(q_valid, q_prep, currentTime, i_obs, params, grid, pdf_writers, gif_writers) # plots : pdf et/ou gif
            if do_obs_table[3] :
                i_obs = 3
                makeSolutionsPlots(q_valid, [], currentTime, i_obs, params, grid, pdf_writers, gif_writers) # plots : pdf et/ou gif
            nbPlotsDone = 1
        else :
            do_gif_plot = False

        
        if simulationChoice == 3 : # If we do vortex : only one plot for exact solution !
            params["plot_parameters"]["do_exact_pdf_plot"] = " "
            params["plot_parameters"]["do_exact_gif_plot"] = " "

        
        ### Plot intervalls
        if nbPlots > 1 :
            timeBetweenPlots = finalTime / (1. * (nbPlots - 1))


    if (do_obs_table[4]): # Sup norm of the solution (stability purposes)
        normsTable = (-1.) * np.ones((a_priori_nb_iter, 3))
        normsTable[0] = np.max(abs(q[iMin:iMax+1, jMin:jMax+1]), axis=(0, 1))
    
    if (do_obs_table[5]): # Sup errors (consistency / convergence purposes) 1st ELEMENT ...[0] UNUSED !!!!
        errorsTable = (-1.) * np.ones((a_priori_nb_iter, 3))

    if (do_obs_table[6]): # total masses
        massesTable = (-1.) * np.ones((a_priori_nb_iter, 3))
        massesTable[0] = dx * dy * np.sum(q[iMin:iMax+1, jMin:jMax+1], axis=(0, 1))

    if (do_obs_table[7]): # DivU
        physicalTermsNormsTable = (-1.) * np.ones((a_priori_nb_iter, 2), order="F")

    return pdf_writers, gif_writers, do_gif_plot, doPlot, nbPlotsDone, timeBetweenPlots, normsTable, errorsTable, massesTable, physicalTermsNormsTable





def getIntermediateObservables(q, q_prep, physicalTerms, currentTime, lastCompletedIteration, grid, dx, dy, iMin, iMax, jMin, jMax, \
                               normsTable, errorsTable, massesTable, physicalTermsNormsTable, params, do_obs_table, \
                               pdf_writers, gif_writers, doPlot, nbPlotsDone, do_pdf_plot, do_gif_plot):

    if (do_obs_table[1] or do_obs_table[2] or do_obs_table[3]):
        if (doPlot and (do_pdf_plot or do_gif_plot)) :
            print("Plot ! temps de simulation : "+str(round(currentTime, 5)))
            q_valid = q[iMin:iMax+1, jMin:jMax+1] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)

            if (do_obs_table[1]):
                i_obs = 1
                makeSolutionsPlots(q_valid, [], currentTime, i_obs, params, grid, pdf_writers, gif_writers) # plots : pdf et/ou gif
            if (do_obs_table[2]):
                i_obs = 2
                makeSolutionsPlots(q_valid, q_prep, currentTime, i_obs, params, grid, pdf_writers, gif_writers) # plots : pdf et/ou gif
            if (do_obs_table[3]):
                i_obs = 3
                makeSolutionsPlots(q_valid, [], currentTime, i_obs, params, grid, pdf_writers, gif_writers) # plots : pdf et/ou gif
        
            nbPlotsDone += 1
            doPlot = False

    if (do_obs_table[4]): # Sup norm of the solution (stability purposes)
        normsTable[lastCompletedIteration] = np.max(abs(q[iMin:iMax+1, jMin:jMax+1]), axis=(0, 1))
    
    if (do_obs_table[5]): # Sup errors (consistency / convergence purposes) 
        q_exact = solutions.getSolution(currentTime, grid, params["simulation_choice"], params["solution_parameters"])
        errorsTable[lastCompletedIteration] = np.max(abs(q[iMin:iMax+1, jMin:jMax+1] - q_exact), axis=(0, 1))

    if (do_obs_table[6]): # total masses
        massesTable[lastCompletedIteration] = dx * dy * np.sum(q[iMin:iMax+1, jMin:jMax+1], axis=(0, 1))
    
    if (do_obs_table[7]): # grad P and div U
        physicalTermsNormsTable[lastCompletedIteration-1, 0] = np.max(abs(physicalTerms[:, :, :, :, :-1]))
        physicalTermsNormsTable[lastCompletedIteration-1, 1] = np.max(abs(physicalTerms[:, :, :, :, -1]))
    
    return doPlot, nbPlotsDone, normsTable, errorsTable, massesTable, physicalTermsNormsTable





def getLastObservables(q, q_prep, currentTime, lastCompletedIteration, grid, normsTable, errorsTable, massesTable, physicalTermsNormsTable, \
                       params, order, coeffs, do_obs_table, pdf_writers, gif_writers, nbPlots, do_pdf_plot, do_gif_plot):
    
    iMin, iMax, jMin, jMax = grid.valid_grid
   
    if (do_obs_table[1] or do_obs_table[2] or do_obs_table[3]):
        if (nbPlots >= 1 and (do_pdf_plot or do_gif_plot)) :
            q_valid = q[iMin:iMax+1, jMin:jMax+1] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)

            if do_obs_table[1] :
                i_obs = 1
                makeSolutionsPlots(q_valid, [], currentTime, i_obs, params, grid, pdf_writers, gif_writers) # plots : pdf et/ou gif
            if do_obs_table[2] :
                i_obs = 2
                makeSolutionsPlots(q_valid, q_prep, currentTime, i_obs, params, grid, pdf_writers, gif_writers) # plots : pdf et/ou gif
            if do_obs_table[3] :
                i_obs = 3
                makeSolutionsPlots(q_valid, [], currentTime, i_obs, params, grid, pdf_writers, gif_writers) # plots : pdf et/ou gif

            closePdfWriters(pdf_writers)
            closeGifWriters(gif_writers)

    if (do_obs_table[4]): # Sup norm of the solution (stability purposes)
        i_obs = [4]
        print("Max sur |u| : ", np.max(normsTable[:, XVEL]))
        print("Max sur |v| : ", np.max(normsTable[:, YVEL]))
        print("Max sur |p| : ", np.max(normsTable[:, PRES]))
        print(" ")
        makeObservableTablePlots(i_obs, normsTable, lastCompletedIteration, params, grid)
    
    if (do_obs_table[5]): # Sup errors (consistency / convergence purposes) 
        i_obs = [5]
        print("Max de l'erreur sur u : ", np.max(errorsTable[:, XVEL]))
        print("Max de l'erreur sur v : ", np.max(errorsTable[:, YVEL]))
        print("Max de l'erreur sur p : ", np.max(errorsTable[:, PRES]))
        print(" ")
        makeObservableTablePlots(i_obs, errorsTable, lastCompletedIteration, params, grid)
    
    if (do_obs_table[6]): # Total mass (conservativity purposes) 
        i_obs = [6]
        print("Max de masse totale de u : ", np.max(massesTable[:, XVEL]))
        print("Max de masse totale de v : ", np.max(massesTable[:, YVEL]))
        print("Max de masse totale de p : ", np.max(massesTable[:, PRES]))
        print(" ")
        makeObservableTablePlots(i_obs, massesTable, lastCompletedIteration, params, grid)
    
    if (do_obs_table[7]): # ||grad P|| et ||div U||
        i_obs = [7]

        physicalTerms = get_evaluation_of_arbitrar_order_SUPG_physical_terms(q, iMin, iMax, jMin, jMax, order, coeffs)

        physicalTermsNormsTable[lastCompletedIteration, 0] = np.max(abs(physicalTerms[:, :, :, :, :-1]))
        physicalTermsNormsTable[lastCompletedIteration, 1] = np.max(abs(physicalTerms[:, :, :, :, -1]))

        print("Max de |grad P| : ", np.max(physicalTermsNormsTable[:, 0]))
        print("Max de |div U| : ", np.max(physicalTermsNormsTable[:, 1]))
        print(" ")
        makeObservableTablePlots(i_obs, physicalTermsNormsTable, lastCompletedIteration, params, grid)