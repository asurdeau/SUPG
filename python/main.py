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
import plots
import spatial_operators
import solutions
from time_evolutions import getOneApproximateSolution, getConvergenceTest, makePerturbedPreparedSimulation



# TRAQUER LES TEMPS DE CALCUL (20 lignes les plus coûteuses) : python -m cProfile -s cumtime main.py | head -25


####################################################################################################################

#                                                       MAIN                                                       #

####################################################################################################################

# Code executé quand le fichier est executé comme script et pas quand il est importé comme module


if __name__ == "__main__":
    # DATA extraction from yaml file
    params = yaml.load(open("parameters.yaml"),Loader=yaml.SafeLoader)
    simulationChoice = params["simulation_choice"]

    # Affichage du choix de la simulation :
    plots.choicePrints(params)

    # Starting time of computations
    startTimeWhole = time.time()

    i_obs_list = np.array(params["observables_choices"])
    if (any(i_obs_list == 8)) :
        getConvergenceTest(params)

    elif (any(i_obs_list == 9)):
        print()
        # makeSeriesOfTests(params)
        
    else :
        if simulationChoice == 5 : # For vortex + pert : long time simu + add of perturbation
            print("perturbed simulation")
            makePerturbedPreparedSimulation(params)
        
        else :
            getOneApproximateSolution(params)


    endTimeWhole = time.time()

    print("Temps de calcul total : ", endTimeWhole - startTimeWhole)
