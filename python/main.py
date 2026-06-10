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
import schemes
import solutions
from Quick_Q1_SUPG import getOneApproximateSolution, getConvergenceTest



# TRAQUER LES TEMPS DE CALCUL (20 lignes les plus coûteuses) : python -m cProfile -s cumtime main.py | head -25


####################################################################################################################

#                                                       MAIN                                                       #

####################################################################################################################

# Code executé quand le fichier est executé comme script et pas quand il est importé comme module


if __name__ == "__main__":
    # DATA extraction from yaml file
    params = yaml.load(open("parameters.yaml"),Loader=yaml.SafeLoader)

    # Affichage du choix de la simulation :
    plots.choicePrints(params)

    # Starting time of computations
    startTimeWhole = time.time()


    if (params["plot_parameters"]["observables"] == 5) :
        getConvergenceTest(params["nList"], params)
    else :
        getOneApproximateSolution(params)


    endTimeWhole = time.time()

    print("Temps de calcul total : ", endTimeWhole - startTimeWhole)
