# Modules généraux
import numpy as np
import matplotlib.pyplot as plt
import yaml
import math
from collections import defaultdict

# Modules perso
import config
from config import XVEL, YVEL, PRES
import grid_mod
import schemes
from plots import makePlots, saveGifs


####################################################################################################################

#                                                       MAIN                                                       #

####################################################################################################################

# Code executé quand le fichier est executé comme script et pas quand il est importé comme module


if __name__ == "__main__":
  
  # DATA extraction from yaml file
  params = yaml.load(open("parameters.yaml"),Loader=yaml.SafeLoader)

  ### grid parameters
  grid_params = params["grid_parameters"]
  Nx, Ny, nGhost = grid_params["mesh_parameters"].values()
  xL, xR, yL, yR = grid_params["domain_parameters"].values()

  ### time parameters
  Tf, CFL = params["time_parameters"].values()

  ### plot parameters
  plot_params = params["plot_parameters"]
  nPlots = params["plot_parameters"]["nb_plots"]
  do_pdf_plot = params["plot_parameters"]["do_pdf_plot"]
  do_gif_plot = params["plot_parameters"]["do_gif_plot"]


  # grid creation 
  gridOp = grid_mod.gridOperator(grid_params)
  iMin, iMax, jMin, jMax = gridOp.valid_grid
  dx, dy = gridOp.steps



  # Initialise data
  u0, v0, p0 = 1., 2., 3.
  q = u0 * np.ones((Nx + 2*nGhost, Ny+2*nGhost, 3))
  q[:, :, YVEL] = v0 * np.ones((Nx + 2*nGhost, Ny+2*nGhost))
  q[:, :, PRES] = p0 * np.ones((Nx + 2*nGhost, Ny+2*nGhost))

  ### initialise time parameters
  i = 0
  currentTime = 0.0
  dt_apriori = min(dx, dy) * CFL
  historique_dt = []

  ### initial plot
  if (nPlots > 1) :
    gif_frames = defaultdict(list)
    q_valid = q[iMin:iMax+2, jMin:jMax+2] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)
    makePlots(q_valid, currentTime, params, gridOp, gif_frames) # plots : pdf et/ou gif
  else :
     do_gif_plot = False

  timeBetweenPlots = Tf / (1. * (nPlots - 1))
  nPlotsDone= 1.
  doPlot = False

  #############################################################################################################



  # TIME LOOP
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
    divF = schemes.SUPG_developped_divFlux(q, gridOp)


    # Update
    q[iMin:iMax, jMin:jMax] += - dt * divF[iMin:iMax, jMin:jMax]


    # Conditions périodiques :
    gridOp.periodize(q)


    # Plots éventuels
    if (doPlot and (do_pdf_plot or do_gif_plot)) :
       q_valid = q[iMin:iMax+2, jMin:jMax+2] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)
       makePlots(q_valid, currentTime, params, gridOp, gif_frames) # plots : pdf et/ou gif
       nPlotsDone += 1
       doPlot = False



  #############################################################################################################



  # FINAL TIME REACHED
  if (nPlots >= 1 and (do_pdf_plot or do_gif_plot)) :
    q_valid = q[iMin:iMax+2, jMin:jMax+2] # WE ONLY EXTRACT q OVER THE VALID MESH (+ WE ADD THE BORDERS)
    makePlots(q_valid, currentTime, params, gridOp, gif_frames) # plots : pdf et/ou gif
  if (do_gif_plot) :
    saveGifs(gif_frames, plot_params)  # duration = secondes par frame



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