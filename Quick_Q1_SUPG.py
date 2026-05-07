import numpy as np
import matplotlib.pyplot as plt
import sys
import yaml
import math

# Modules perso
import grid_mod
import schemes
from config import XVEL, YVEL, PRES


####################################################################################################################

#                                                       MAIN                                                       #

####################################################################################################################

# Code executé quand le fichier est executé comme script et pas quand il est importé comme module


X = np.linspace(0, 1, 100+1, endpoint=True)
Y = X**2

plt.figure()
plt.plot(X, Y)
plt.show()


if __name__ == "__main__":
  params = yaml.load(open("parameters.yaml"),Loader=yaml.SafeLoader)
  grid_params = params["grid_parameters"]


  Nx, Ny, nGhost = grid_params["mesh_parameters"].values()
  xL, xR, yL, yR = grid_params["domain_parameters"].values()
  Tf, CFL = params["time_parameters"].values()

  # grid creation 
  gridOp = grid_mod.gridOperator(grid_params)
  iMin, iMax, jMin, jMax = gridOp.valid_grid
  dx, dy = gridOp.steps

  # Initial data
  a = 2.
  q = a * np.ones((Nx + 2*nGhost, Ny+2*nGhost, 3))
  X = np.linspace(xL, xR, Nx+1, endpoint=True)
  Y = X**2

  plt.figure()
  plt.plot(X, Y)
#   plt.imshow(q[iMin:iMax+2, jMin:jMax+2, XVEL])
  plt.show()

  dt = min(dx, dy) * CFL
  print(dt)
  time = 0.0
  i = 0
  historique_dt = []
  while (abs(time - Tf) > 1.e-10) :
    i += 1
    time = i * dt  # : accumule les erreurs d'arrondis !!!!
    # historique_dt.append(dt)
    # time = math.fsum(historique_dt) # recalcule la somme à chaque fois

    if(time > Tf) :
        dt = time - Tf
        t = Tf
        
    print(time)

    # Div . Fluxes computations

    divF = schemes.SUPG_developped_divFlux(q, gridOp)

    # Update
    q[iMin:iMax, jMin:jMax] += - dt * divF[iMin:iMax, jMin:jMax]

    # Conditions périodiques :
    gridOp.periodize(q)






####################################################################################################################

#                                                 AUTRES ROUTINES                                                  #

####################################################################################################################


def makePlots(q, time, params, grid):    
    Nx, Ny, nGhost = params["mesh_parameters"].values()
    xL, xR, yL, yR = params["domain_parameters"].values()

    iMin, iMax, jMin, jMax = grid.valid_grid


    imXVEL = imshow(q[iMin:iMax+2, jMin:jMax+2, XVEL],cmap=cm.RdBu) # drawing the function
    cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
    clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
    colorbar(imXVEL) # adding the colobar on the right
    # latex fashion title
    title('time : '+str(time))
    show()


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