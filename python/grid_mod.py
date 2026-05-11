# Modules généraux
import numpy as np

# Modules perso
import config
from config import XVEL, YVEL, PRES 

class gridOperator:

    # Do not place any variable definition here or it will be common for all instances !!
    # Meaning the same AND modified at the same time for each one !

    def __init__(self, params):
        Nx, Ny, nGhost = params["mesh_parameters"].values()
        xL, xR, yL, yR = params["domain_parameters"].values()

        self.valid_grid = nGhost, Nx + nGhost - 1, nGhost, Ny + nGhost - 1

        self.steps = (xR - xL) / (1. * Nx), (yR - yL) / (1. * Ny)

        self.xCoord = np.linspace(xL, xR, Nx+1)
        self.yCoord = np.linspace(yL, yR, Ny+1)
        self.xGrid, self.yGrid = np.meshgrid(self.xCoord, self.yCoord)


    def periodize(self, q) :
        iMin, iMax, jMin, jMax = self.valid_grid
        nGhost = iMin
        
        # BANDES
        # bande verticale à gauche 
        q[:iMin, jMin:jMax+1] = q[iMax-nGhost+1:iMax+1, jMin:jMax+1]

        # bande verticale à droite
        q[iMax+1:, jMin:jMax+1] = q[iMin:iMin+nGhost, jMin:jMax+1]

        # bande horizontale en bas
        q[iMin:iMax+1, :jMin] = q[iMin:iMax+1, jMax-nGhost+1:jMax+1]

        # bande horizontale en haut
        q[iMin:iMax+1, jMax+1:] = q[iMin:iMax+1, jMin:jMin+nGhost]

        # COINS
        # bas gauche 
        q[:iMin, :jMin] = q[iMax-nGhost+1:iMax+1, jMax-nGhost+1:jMax+1]

        # bas droit 
        q[iMax+1:, :jMin] = q[iMin:iMin+nGhost, jMax-nGhost+1:jMax+1]

        # haut gauche 
        q[:iMin, jMax+1:] = q[iMax-nGhost+1:iMax+1, jMin:jMin+nGhost]

        # bas droit 
        q[iMax+1:, jMax+1:] = q[iMin:iMin+nGhost, jMin:jMin+nGhost]

        return q
    


    def isPeriodic(self, q) :
        iMin, iMax, jMin, jMax = self.valid_grid
        nGhost = iMin
        
        # BANDES
        # bande verticale à gauche 
        test = (abs(q[:iMin] - q[iMax-nGhost+1:iMax+1]) < 1.e-16).all()

        # bande verticale à droite
        test = test and (abs(q[iMax+1:] - q[iMin:iMin+nGhost]) < 1.e-16).all()

        # bande horizontale en bas
        test = test and (abs(q[: jMin] - q[:jMax-nGhost+1:jMax+1]) < 1.e-16).all()

        # bande horizontale en haut
        test = test and (abs(q[jMax+1:] - q[jMin:jMin+nGhost]) < 1.e-16).all()

        return test