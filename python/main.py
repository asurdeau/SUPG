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
from plots import makePlots, openGifWriters, closeGifWriters
import solutions
from Quick_Q1_SUPG import getOneApproximateSolution


####################################################################################################################

#                                                       MAIN                                                       #

####################################################################################################################

# Code executé quand le fichier est executé comme script et pas quand il est importé comme module


if __name__ == "__main__":
    # DATA extraction from yaml file
    params = yaml.load(open("parameters.yaml"),Loader=yaml.SafeLoader)


    # Starting time of computations
    startTimeWhole = time.time()
    getOneApproximateSolution(params)
    endTimeWhole = time.time()

    print("Temps de calcul total : ", endTimeWhole - startTimeWhole)

