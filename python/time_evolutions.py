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
import copy


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
    h = gridOp.characteristic_size
    X = gridOp.xValidGrid
    Y = gridOp.yValidGrid

    ### initialise time parameters
    currentIteration = 0
    lastCompletedIteration = 0
    currentTime = 0.0
    taux_progression = 0.1
    dt_apriori = min(dx, dy) * CFL

    ### time tables
    divFluxTimeTable, updateTimeTable = np.zeros((2, int(2. * finalTime / dt_apriori)))


    ### Initialise coefficients
    operators = yaml.load(open("operators.yaml"),Loader=yaml.SafeLoader)
    if schemeChoice == 3 :
        operatorsCoeff = spatial_operators.extract_operators(operators)
    elif schemeChoice == 4 :
        operatorsCoeff = spatial_operators.extract_operators_improved(dx, dy)

    operatorsCoeffTest = spatial_operators.extract_operators(operators)
    # operatorsCoeffTest = spatial_operators.extract_operators_improved(dx, dy)

    order = params["order"]

    ### Initialise data
    if order == 0 :
        q = np.zeros((Nx + 2*nGhost, Ny + 2*nGhost, 3))
    else : 
        q = np.zeros((Nx + 2*nGhost, Ny + 2*nGhost, order, order, 3))
        evolCoeffs = spatial_operators.get_arbitrar_order_SUPG__weighted_evolution_operator_coeffs(order, dx, dy)
        massCoeffs = spatial_operators.get_arbitrar_order_SUPG_weighted_mass_operator_coeffs(order, dx, dy)

    if simulationChoice == 4 :
        q_prep = solutions.getSolution(currentTime, X, Y, simulationChoice, params["solution_parameters"])
        q[iMin:iMax+2, jMin:jMax+2] = q_prep
        q[iMin:iMax+2, jMin:jMax+2] = solutions.addPerturbation(q[iMin:iMax+1, jMin:jMax+1], X, Y, params["solution_parameters"])

    if (simulationChoice == 5):
        q[iMin:iMax+2, jMin:jMax+2] = q_prep
        q[iMin:iMax+2, jMin:jMax+2] = solutions.addPerturbation(q[iMin:iMax+1, jMin:jMax+1], X, Y, params["solution_parameters"])

    else : 
        q[iMin:iMax+2, jMin:jMax+2] = solutions.getSolution(currentTime, X, Y, simulationChoice, params["solution_parameters"])

    ### Boundary conditions 
    if (simulationChoice == 3 or simulationChoice == 5):        # vortex simulation -> Dirichlet BC
        gridOp.dirichlet(q)
    else :                                                      # else (default) -> periodic BC 
        gridOp.periodize(q) 
        print("test periodicité donnée initiale : ", gridOp.isPeriodic(q))



    if (any(np.array(i_obs_list) == 7)):
        physicalTerms = np.zeros_like(q)
    else :
        physicalTerms = []

    
    ### initial observables
    pdf_writers, gif_writers, doPlot, do_gif_plot, nbPlotsDone, timeBetweenPlots, Xvis, Yvis, \
        obs_tables_step, iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable = \
        getInitialObservables(q, q_prep, finalTime, dt_apriori, gridOp, params, simulationChoice, do_obs_table, nbPlots)

    #############################################################################################################



    # TIME LOOP : AVOID CLASS / DICTIONNARY LOOKUPS !
    # starting time of the loop
    startTimeLoop = time.time()
    divFluxTimeTable2, updateTimeTable2 = np.zeros((2, int(2. * finalTime / dt_apriori)))
    schemesCompTable = - np.ones(int(2. * finalTime / dt_apriori))
    q1 = q.copy()
    q2 = q.copy()

    if order == 0 :   # FIRST ORDER SIMULATION
        if (any(np.array(i_obs_list) == 7)): # WE HAVE TO COMPUTE THE PHYSICAL TERMS
            while (abs(currentTime - finalTime) > 1.e-10) :
            
                # AJUSTEMENT EVENTUEL DU PAS DE TEMPS ET MAJ DU TEMPS ACTUEL
                dt, currentTime, doPlot, taux_progression = getAdaptedTimeUpdate(dt_apriori, currentTime, finalTime, do_obs_table, nbPlotsDone, timeBetweenPlots, doPlot, taux_progression)

                # MISE A JOUR
                currentIteration += 1
                q, physicalTerms = time_operators.getForwardEulerUpdateAndPhysicalTerms(q, dt, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, simulationChoice, operatorsCoeff, \
                                                        divFluxTimeTable, updateTimeTable, currentIteration)

                lastCompletedIteration += 1

                # OBSERVABLES INTERMEDIAIRES EVENTUELLES
                doPlot, nbPlotsDone, iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable \
                    = getIntermediateObservables(q, q_prep, physicalTerms, currentTime, finalTime, lastCompletedIteration, X, Y, h, dx, dy, iMin, iMax, jMin, jMax, \
                                                params, do_obs_table, Xvis, Yvis, pdf_writers, gif_writers, doPlot, nbPlotsDone, do_pdf_plot, do_gif_plot, \
                                                obs_tables_step, iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable)
                
        else : 
            while (abs(currentTime - finalTime) > 1.e-10) :
                
                # AJUSTEMENT EVENTUEL DU PAS DE TEMPS ET MAJ DU TEMPS ACTUEL
                dt, currentTime, doPlot, taux_progression = getAdaptedTimeUpdate(dt_apriori, currentTime, finalTime, do_obs_table, nbPlotsDone, timeBetweenPlots, doPlot, taux_progression)
        
        
                # MISE A JOUR
                currentIteration += 1
                q = time_operators.getForwardEulerUpdate(q, dt, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, simulationChoice, operatorsCoeff, \
                                                        divFluxTimeTable, updateTimeTable, currentIteration)

                # q1 = time_operators.getForwardEulerUpdate(q1, dt, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, simulationChoice, operatorsCoeff, \
                #                                                     divFluxTimeTable, updateTimeTable, currentIteration)
    
                # q2 = time_operators.getForwardEulerUpdate(q2, dt, gridOp, iMin, iMax, jMin, jMax, dx, dy, -1, simulationChoice, operatorsCoeffTest, \
                #                                                     divFluxTimeTable2, updateTimeTable2, currentIteration)
                # schemesCompTable[lastCompletedIteration] = np.max(abs(q1 - q2))
        
                lastCompletedIteration += 1
        
                # OBSERVABLES INTERMEDIAIRES EVENTUELLES
                doPlot, nbPlotsDone, iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable \
                    = getIntermediateObservables(q, q_prep, physicalTerms, currentTime, finalTime, lastCompletedIteration, X, Y, h, dx, dy, iMin, iMax, jMin, jMax, \
                                                params, do_obs_table, Xvis, Yvis, pdf_writers, gif_writers, doPlot, nbPlotsDone, do_pdf_plot, do_gif_plot, \
                                                obs_tables_step, iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable)


    else :          # ARBITRAR ORDER SIMULATION


        if (any(np.array(i_obs_list) == 7) and False): # WE HAVE TO COMPUTE THE PHYSICAL TERMS
            while (abs(currentTime - finalTime) > 1.e-10) :
            
                # AJUSTEMENT EVENTUEL DU PAS DE TEMPS ET MAJ DU TEMPS ACTUEL
                dt, currentTime, doPlot, taux_progression = getAdaptedTimeUpdate(dt_apriori, currentTime, finalTime, do_obs_table, nbPlotsDone, timeBetweenPlots, doPlot, taux_progression)

                # MISE A JOUR
                currentIteration += 1
                q, physicalTerms = time_operators.getForwardEulerUpdateAndPhysicalTerms(q, dt, gridOp, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, simulationChoice, operatorsCoeff, \
                                                        divFluxTimeTable, updateTimeTable, currentIteration)

                q = time_operators.getDecUpdate(q, dt, order, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs)

                lastCompletedIteration += 1

                # OBSERVABLES INTERMEDIAIRES EVENTUELLES
                doPlot, nbPlotsDone, iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable \
                    = getIntermediateObservables(q, q_prep, physicalTerms, currentTime, finalTime, lastCompletedIteration, X, Y, h, dx, dy, iMin, iMax, jMin, jMax, \
                                                params, do_obs_table, Xvis, Yvis, pdf_writers, gif_writers, doPlot, nbPlotsDone, do_pdf_plot, do_gif_plot, \
                                                obs_tables_step, iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable)
            
        else : 
            while (abs(currentTime - finalTime) > 1.e-10) :
                
                # AJUSTEMENT EVENTUEL DU PAS DE TEMPS ET MAJ DU TEMPS ACTUEL
                dt, currentTime, doPlot, taux_progression = getAdaptedTimeUpdate(dt_apriori, currentTime, finalTime, do_obs_table, nbPlotsDone, timeBetweenPlots, doPlot, taux_progression)
        
                # MISE A JOUR
                currentIteration += 1
                q = time_operators.getDecUpdate(q, dt, order, iMin, iMax, jMin, jMax, nGhost, massCoeffs, evolCoeffs)
        
                lastCompletedIteration += 1
        
                # OBSERVABLES INTERMEDIAIRES EVENTUELLES
                doPlot, nbPlotsDone, iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable \
                    = getIntermediateObservables(q, q_prep, physicalTerms, currentTime, finalTime, lastCompletedIteration, X, Y, h, dx, dy, iMin, iMax, jMin, jMax, \
                                                params, do_obs_table, Xvis, Yvis, pdf_writers, gif_writers, doPlot, nbPlotsDone, do_pdf_plot, do_gif_plot, \
                                                obs_tables_step, iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable)
                               
    #############################################################################################################



    # FIN DE LA BOUCLE EN TEMPS 
    ### Quelques prints
    endTimeLoop = time.time()
    print("Temps de calcul boucle : ", endTimeLoop - startTimeLoop)
    print("type de la sortie q : ", q.dtype)
    print("moyenne pour le calcul de divFLux  : ", np.sum(divFluxTimeTable[divFluxTimeTable > 0]) / lastCompletedIteration )
    print("moyenne pour la mise à jour de q   : ", np.sum(updateTimeTable[updateTimeTable > 0]) / lastCompletedIteration )

    print("2e schéma : ")
    print("moyenne pour le calcul de divFLux  : ", np.sum(divFluxTimeTable2[divFluxTimeTable2 > 0]) / lastCompletedIteration )
    print("moyenne pour la mise à jour de q   : ", np.sum(updateTimeTable2[updateTimeTable2 > 0]) / lastCompletedIteration )
    print("max de l'écart entre les deux schémas : ", np.max(schemesCompTable))

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
    getLastObservables(q, q_prep, currentTime, gridOp, params, order, operatorsCoeff, do_obs_table, \
                       pdf_writers, gif_writers, nbPlots, do_pdf_plot, do_gif_plot, \
                       iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable)

    
    # FIN DE LA ROUTINE : ON RENVOIE LA VALEUR FINALE
    return q[iMin:iMax+1, jMin:jMax+1]





def makePerturbedPreparedSimulation(params):

    preparationParams = copy.deepcopy(params)
    simulationChoice = params["simulation_choice"]
    if simulationChoice == 5 : # For vortex + pert : long time simu + add of perturbation
        
        # Modification of params in order to make the preparation
        preparationParams["simulation_choice"] = 3
        preparationParams["time_parameters"]["end_time"] = 50.
        preparationParams["observables_choices"] = []


        print("Préparation des données en cours...")
        q_prep = getOneApproximateSolution(preparationParams)

        print("\n" + "Début de la vraie simulation")
        getOneApproximateSolution(params, q_prep)


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
    if any(do_obs_table[1:4]) :
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






def getInitialObservables(q, q_prep, finalTime, dt_apriori, grid, params, simulationChoice, do_obs_table, nbPlots):
    
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx, dy = grid.steps
    h = grid.characteristic_size

    # Liste des sorties:
    ### Liées à des plots de solution
    pdf_writers = []
    gif_writers = []
    doPlot = False
    do_gif_plot = params["plot_parameters"]["do_gif_plot"]
    nbPlotsDone = 1
    timeBetweenPlots = 2 * finalTime # Default : Only eventual plot is at the end : "currentTime + dt_apriori > nbPlotsDone * timeBetweenPlots" never true
    Xvis, Yvis = [], []

    ### Liées à des observables autres sur les solutions
    obs_tables_step = 0
    iterations_of_obs_table = []
    normsTable = []
    errorsTable = []
    massesTable = []
    physicalTermsNormsTable = []

    currentTime = 0.
    # a_priori_nb_iter_margin = 2 * int(finalTime / dt_apriori)     # a priori number of iterations

    if any(do_obs_table[4:7+1]):
        a_priori_nb_iter = int(finalTime / dt_apriori)     # a priori number of iterations
        if a_priori_nb_iter < 400 :
            obs_tables_step = 1
        else :
            obs_tables_step = int(a_priori_nb_iter / 400)

        
        obs_tables_step = 1
        print("obs_tables_step : \n ", obs_tables_step)

        iterations_of_obs_table = [0]
    
    if any(do_obs_table[1:4]) :
        Xvis, Yvis = plots.getVisualisationGrid(params)

        pdf_writers = openPdfWriters(do_obs_table, params)
        gif_writers = openGifWriters(do_obs_table, params)

        if (nbPlots > 1) :
            q_valid = q[iMin:iMax+2, jMin:jMax+2] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)
            if do_obs_table[1] :
                i_obs = 1
                makeSolutionsPlots(Xvis, Yvis, q_valid, [], currentTime, i_obs, params, h, pdf_writers, gif_writers) # plots : pdf et/ou gif
            if do_obs_table[2] :
                i_obs = 2
                makeSolutionsPlots(Xvis, Yvis, q_valid, q_prep, currentTime, i_obs, params, h, pdf_writers, gif_writers) # plots : pdf et/ou gif
            if do_obs_table[3] :
                i_obs = 3
                makeSolutionsPlots(Xvis, Yvis, q_valid, [], currentTime, i_obs, params, h, pdf_writers, gif_writers) # plots : pdf et/ou gif
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
        # normsTable = (-1.) * np.ones((a_priori_nb_iter_margin, 3))
        # normsTable[0] = np.max(abs(q[iMin:iMax+2, jMin:jMax+2]), axis=(0, 1))
        normsTable.append(np.max(abs(q[iMin:iMax+2, jMin:jMax+2]), axis=(0, 1)))
    
    if (do_obs_table[5]): # Sup errors (consistency / convergence purposes) 1st ELEMENT ...[0] UNUSED !!!!
        # errorsTable = (-1.) * np.ones((a_priori_nb_iter_margin, 3)) 
        errorsTable.append(np.array([np.nan, np.nan, np.nan]))

    if (do_obs_table[6]): # total masses
        # massesTable = (-1.) * np.ones((a_priori_nb_iter_margin, 3))
        # massesTable[0] = dx * dy * np.sum(q[iMin:iMax+1, jMin:jMax+1], axis=(0, 1))
        massesTable.append(dx * dy * np.sum(q[iMin:iMax+1, jMin:jMax+1], axis=(0, 1)))

    # if (do_obs_table[7]): # DivU
    #     physicalTermsNormsTable = (-1.) * np.ones((a_priori_nb_iter_margin, 2), order="F")

    return pdf_writers, gif_writers, doPlot, do_gif_plot, nbPlotsDone, timeBetweenPlots, Xvis, Yvis, \
        obs_tables_step, iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable





def getIntermediateObservables(q, q_prep, physicalTerms, currentTime, finalTime, lastCompletedIteration, X, Y, h, dx, dy, iMin, iMax, jMin, jMax, \
                               params, do_obs_table, Xvis, Yvis, pdf_writers, gif_writers, doPlot, nbPlotsDone, do_pdf_plot, do_gif_plot, \
                               obs_tables_step, iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable):

    if any(do_obs_table[1:4]) :
        if (doPlot and (do_pdf_plot or do_gif_plot)) :
            print("Plot ! temps de simulation : "+str(round(currentTime, 5)))
            q_valid = q[iMin:iMax+2, jMin:jMax+2] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)

            if (do_obs_table[1]):
                i_obs = 1
                makeSolutionsPlots(Xvis, q_valid, [], currentTime, i_obs, params, h, pdf_writers, gif_writers) # plots : pdf et/ou gif
            if (do_obs_table[2]):
                i_obs = 2
                makeSolutionsPlots(Xvis, q_valid, q_prep, currentTime, i_obs, params, h, pdf_writers, gif_writers) # plots : pdf et/ou gif
            if (do_obs_table[3]):
                i_obs = 3
                makeSolutionsPlots(Xvis, q_valid, [], currentTime, i_obs, params, h, pdf_writers, gif_writers) # plots : pdf et/ou gif
        
            nbPlotsDone += 1
            doPlot = False

    # addObservation = True
    # addPhysicalTermsObservation = True
    if any(do_obs_table[4:7+1]):
        addObservation = lastCompletedIteration % obs_tables_step == 0 or abs(currentTime-finalTime) < 1.e-15
        addPhysicalTermsObservation = (lastCompletedIteration-1) % obs_tables_step == 0

        if addObservation :
            iterations_of_obs_table.append(lastCompletedIteration)

    if (do_obs_table[4] and addObservation): # Sup norm of the solution (stability purposes)
        # normsTable[lastCompletedIteration] = np.max(abs(q[iMin:iMax+2, jMin:jMax+2]), axis=(0, 1))
        normsTable.append(np.max(abs(q[iMin:iMax+2, jMin:jMax+2]), axis=(0, 1)))
    
    if (do_obs_table[5] and addObservation): # Sup errors (consistency / convergence purposes) 
        q_exact = solutions.getSolution(currentTime, X, Y, params["simulation_choice"], params["solution_parameters"])
        # errorsTable[lastCompletedIteration] = np.max(abs(q[iMin:iMax+2, jMin:jMax+2] - q_exact), axis=(0, 1))
        errorsTable.append(np.max(abs(q[iMin:iMax+2, jMin:jMax+2] - q_exact), axis=(0, 1)))

    if (do_obs_table[6] and addObservation): # total masses
        # massesTable[lastCompletedIteration] = dx * dy * np.sum(q[iMin:iMax+1, jMin:jMax+1], axis=(0, 1))
        massesTable.append(dx * dy * np.sum(q[iMin:iMax+1, jMin:jMax+1], axis=(0, 1)))
    
    if (do_obs_table[7] and addPhysicalTermsObservation): # grad P and div U
        # physicalTermsNormsTable[lastCompletedIteration-1, 0] = np.max(abs(physicalTerms[:, :, :-1]))
        # physicalTermsNormsTable[lastCompletedIteration-1, 1] = np.max(abs(physicalTerms[:, :, -1]))
        physicalTermsNormsTable.append(np.array([np.max(abs(physicalTerms[:, :, :-1])), np.max(abs(physicalTerms[:, :, -1]))]))
    
    return doPlot, nbPlotsDone, iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable





def getLastObservables(q, q_prep, currentTime, grid, params, order, operatorsCoeff, do_obs_table, \
                       pdf_writers, gif_writers, nbPlots, do_pdf_plot, do_gif_plot, \
                       iterations_of_obs_table, normsTable, errorsTable, massesTable, physicalTermsNormsTable):
    
    iMin, iMax, jMin, jMax = grid.valid_grid
    h = grid.characteristic_size
   
    if (do_obs_table[1] or do_obs_table[2] or do_obs_table[3]):
        if (nbPlots >= 1 and (do_pdf_plot or do_gif_plot)) :
            q_valid = q[iMin:iMax+2, jMin:jMax+2] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)

            if do_obs_table[1] :
                i_obs = 1
                makeSolutionsPlots(q_valid, [], currentTime, i_obs, params, h, pdf_writers, gif_writers) # plots : pdf et/ou gif
            if do_obs_table[2] :
                i_obs = 2
                makeSolutionsPlots(q_valid, q_prep, currentTime, i_obs, params, h, pdf_writers, gif_writers) # plots : pdf et/ou gif
            if do_obs_table[3] :
                i_obs = 3
                makeSolutionsPlots(q_valid, [], currentTime, i_obs, params, h, pdf_writers, gif_writers) # plots : pdf et/ou gif

            closePdfWriters(pdf_writers)
            closeGifWriters(gif_writers)

    if (any(do_obs_table[4:7+1])):
        iterations_of_obs_table = np.array(iterations_of_obs_table)
        # print("iterations_of_obs_table : \n", iterations_of_obs_table)

    if (do_obs_table[4]): # Sup norm of the solution (stability purposes)
        i_obs = [4]
        normsTable = np.array(normsTable)
        # print("normsTable : \n", normsTable)
        print("Max sur |u| : ", np.max(normsTable[:, XVEL]))
        print("Max sur |v| : ", np.max(normsTable[:, YVEL]))
        print("Max sur |p| : ", np.max(normsTable[:, PRES]))
        print(" ")
        makeObservableTablePlots(i_obs, iterations_of_obs_table, normsTable, params, h)
    
    if (do_obs_table[5]): # Sup errors (consistency / convergence purposes) 
        i_obs = [5]
        errorsTable = np.array(errorsTable)
        print("Max de l'erreur sur u : ", np.max(errorsTable[:, XVEL]))
        print("Max de l'erreur sur v : ", np.max(errorsTable[:, YVEL]))
        print("Max de l'erreur sur p : ", np.max(errorsTable[:, PRES]))
        print(" ")
        makeObservableTablePlots(i_obs, iterations_of_obs_table, errorsTable, params, h)
    
    if (do_obs_table[6]): # Total mass (conservativity purposes) 
        i_obs = [6]
        massesTable = np.array(massesTable)
        print("Max de masse totale de u : ", np.max(massesTable[:, XVEL]))
        print("Max de masse totale de v : ", np.max(massesTable[:, YVEL]))
        print("Max de masse totale de p : ", np.max(massesTable[:, PRES]))
        print(" ")
        makeObservableTablePlots(i_obs, iterations_of_obs_table, massesTable, params, h)
    
    if (do_obs_table[7]): # ||grad P|| et ||div U||
        i_obs = [7]

        schemeChoice = params["scheme_choice"]
        dx, dy = grid.steps

        physicalTerms = spatial_operators.getApproxDivFluxAndPhysicalTerms(q, grid, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, operatorsCoeff)[1]

        # physicalTermsNormsTable[lastCompletedIteration, 0] = np.max(abs(physicalTerms[:, :, :-1]))
        # physicalTermsNormsTable[lastCompletedIteration, 1] = np.max(abs(physicalTerms[:, :, -1]))
        physicalTermsNormsTable.append(np.array([np.max(abs(physicalTerms[:, :, :-1])), np.max(abs(physicalTerms[:, :, -1]))]))
        physicalTermsNormsTable = np.array(physicalTermsNormsTable)

        print("Max de |grad P| : ", np.max(physicalTermsNormsTable[:, 0]))
        print("Max de |div U| : ", np.max(physicalTermsNormsTable[:, 1]))
        print(" ")
        makeObservableTablePlots(i_obs, iterations_of_obs_table, physicalTermsNormsTable, params, h)