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

        self.steps = (xR - xL) / (1. * (Nx-1)), (yR - yL) / (1. * (Ny-1))

        # IMPORTANT HERE : THE GRIDS ARE BUILT SO THAT WE HAVE THE WHOLE DOMAIN 
        # EVEN FOR PERIODIC BC !!
        self.xCoord = np.linspace(xL, xR, Nx)
        self.yCoord = np.linspace(yL, yR, Ny)

        # Ajout des mailles fantômes (maillage uniforme uniquement)
        ghost_x = np.arange(1, nGhost + 1)
        self.xCoord = np.concatenate([xL - ghost_x[::-1] * self.steps[0], self.xCoord, xR + ghost_x * self.steps[0]])

        ghost_y = np.arange(1, nGhost + 1)
        self.yCoord = np.concatenate([yL - ghost_y[::-1] * self.steps[1], self.yCoord, yR + ghost_y * self.steps[1]])

        xx, yy = np.meshgrid(self.xCoord, self.yCoord)
        self.xGrid = np.transpose(xx)
        self.yGrid = np.transpose(yy)



    # def periodize(self, q) :
    #     iMin, iMax, jMin, jMax = self.valid_grid
    #     nGhost = iMin
        
    #     # BANDES
    #     # bande verticale à gauche 
    #     q[:iMin, jMin:jMax+1] = q[iMax-nGhost+1:iMax+1, jMin:jMax+1]

    #     # bande verticale à droite
    #     q[iMax+1:, jMin:jMax+1] = q[iMin:iMin+nGhost, jMin:jMax+1]

    #     # bande horizontale en bas
    #     q[iMin:iMax+1, :jMin] = q[iMin:iMax+1, jMax-nGhost+1:jMax+1]

    #     # bande horizontale en haut
    #     q[iMin:iMax+1, jMax+1:] = q[iMin:iMax+1, jMin:jMin+nGhost]

    #     # COINS
    #     # bas gauche 
    #     q[:iMin, :jMin] = q[iMax-nGhost+1:iMax+1, jMax-nGhost+1:jMax+1]

    #     # bas droit 
    #     q[iMax+1:, :jMin] = q[iMin:iMin+nGhost, jMax-nGhost+1:jMax+1]

    #     # haut gauche 
    #     q[:iMin, jMax+1:] = q[iMax-nGhost+1:iMax+1, jMin:jMin+nGhost]

    #     # bas droit 
    #     q[iMax+1:, jMax+1:] = q[iMin:iMin+nGhost, jMin:jMin+nGhost]

    #     return q
    
    # REMEMBER : iMax's value is the same as iMin one's 
    def periodize(self, q) :
        iMin, iMax, jMin, jMax = self.valid_grid
        nGhost = iMin

        # BANDES
        # bande verticale à gauche 
        q[:iMin, jMin:jMax+1] = q[iMax-nGhost:iMax, jMin:jMax+1]

        # bande verticale à droite
        q[iMax:, jMin:jMax+1] = q[iMin:iMin+nGhost+1, jMin:jMax+1]

        # bande horizontale en bas
        q[iMin:iMax+1, :jMin] = q[iMin:iMax+1, jMax-nGhost:jMax]

        # bande horizontale en haut
        q[iMin:iMax+1, jMax:] = q[iMin:iMax+1, jMin:jMin+nGhost+1]

        # COINS
        # bas gauche 
        q[:iMin, :jMin] = q[iMax-nGhost:iMax, jMax-nGhost:jMax]

        # bas droit 
        q[iMax:, :jMin] = q[iMin:iMin+nGhost+1, jMax-nGhost:jMax]

        # haut gauche 
        q[:iMin, jMax:] = q[iMax-nGhost:iMax, jMin:jMin+nGhost+1]

        # bas droit 
        q[iMax:, jMax:] = q[iMin:iMin+nGhost+1, jMin:jMin+nGhost+1]

        return q
    


    # def isPeriodic(self, q) :
    #     iMin, iMax, jMin, jMax = self.valid_grid
    #     nGhost = iMin
        
    #     # BANDES
    #     # bande verticale à gauche 
    #     test = (abs(q[:iMin] - q[iMax-nGhost+1:iMax+1]) < 1.e-16).all()

    #     # bande verticale à droite
    #     test = test and (abs(q[iMax+1:] - q[iMin:iMin+nGhost]) < 1.e-16).all()

    #     # bande horizontale en bas
    #     test = test and (abs(q[: jMin] - q[:jMax-nGhost+1:jMax+1]) < 1.e-16).all()

    #     # bande horizontale en haut
    #     test = test and (abs(q[jMax+1:] - q[jMin:jMin+nGhost]) < 1.e-16).all()

    #     return test
    
    def isPeriodic(self, q) :
        iMin, iMax, jMin, jMax = self.valid_grid
        nGhost = iMin

        # BANDES
        # bande verticale à gauche 
        test = (abs(q[:iMin+1] - q[iMax-nGhost:iMax+1]) < 1.e-16).all()

        # bande verticale à droite
        test = test and (abs(q[iMax:] - q[iMin:iMin+nGhost+1]) < 1.e-16).all()

        # bande horizontale en bas
        test = test and (abs(q[:, :jMin+1] - q[:, jMax-nGhost:jMax+1]) < 1.e-16).all()

        # bande horizontale en haut
        test = test and (abs(q[:, jMax:] - q[:, jMin:jMin+nGhost+1]) < 1.e-16).all()

        return test