# Modules généraux
import numpy as np

# Modules perso
import config
from config import XVEL, YVEL, PRES 

class gridOperator:

    # Do not place any variable definition here or it will be common for all instances !!
    # Meaning the same AND modified at the same time for each one !

    def __init__(self, params):
        Nx, Ny, nGhost = params["grid_parameters"]["mesh_parameters"].values()
        xL, xR, yL, yR = params["grid_parameters"]["domain_parameters"].values()
        order = params["order"]

        self.valid_grid = nGhost, Nx + nGhost - 1, nGhost, Ny + nGhost - 1

        dx, dy = (xR - xL) / (1. * Nx), (yR - yL) / (1. * Ny)
        self.steps = dx, dy
        self.characteristic_size = max(dx, dy)

        # IMPORTANT HERE : THE GRIDS ARE BUILT SO THAT WE HAVE THE WHOLE DOMAIN 
        # EVEN FOR PERIODIC BC !!
        xCoord = np.linspace(xL, xR, Nx+1, endpoint=True)
        yCoord = np.linspace(yL, yR, Ny+1, endpoint=True)

        X, Y = np.meshgrid(xCoord, yCoord, indexing="ij")

        if order == 0 :
            self.xValidGrid = X
            self.yValidGrid = Y

        else :
            X_HO, Y_HO = np.zeros( (np.shape(X) + (order, order,)) )

            for k in range(order):
                for l in range(order):
                    X_HO[:, :, k, l] = X + (1. * k) / order * dx
                    Y_HO[:, :, k, l] = Y + (1. * l) / order * dy

            self.xValidGrid = X_HO
            self.yValidGrid = Y_HO


    
    # REMEMBER : iMax's value is the same as iMin one's 
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
    


    # REMEMBER : iMax's value is the same as iMin one's 
    def dirichlet(self, q) :
        iMin, iMax, jMin, jMax = self.valid_grid


        # Dirichlet homogène 0 pour les vitesses
        # bande verticale à gauche 
        q[:iMin+1, :, :] = 0.

        # bande verticale à droite
        q[iMax+1:, :, :] = 0.

        # bande horizontale en bas
        q[:, :jMin+1, :] = 0.

        # bande horizontale en haut
        q[:, jMax+1:, :] = 0.


        # Dirichlet homogène 1 pour la pression
        # bande verticale à gauche 
        q[:iMin+1, :, PRES] = 1.

        # bande verticale à droite
        q[iMax+1:, :, PRES] = 1.

        # bande horizontale en bas
        q[:, :jMin+1, PRES] = 1.

        # bande horizontale en haut
        q[:, jMax+1:, PRES] = 1.

        return q
    

    
    def isPeriodic(self, q) :
        iMin, iMax, jMin, jMax = self.valid_grid
        nGhost = iMin

        # BANDES
        # bande verticale à gauche 
        test = (abs(q[:iMin+1] - q[iMax-nGhost+1:iMax+2]) < 1.e-14).all()

        # bande verticale à droite
        test = test and (abs(q[iMax+1:] - q[iMin:iMin+nGhost]) < 1.e-14).all()

        # bande horizontale en bas
        test = test and (abs(q[:, :jMin+1] - q[:, jMax-nGhost+1:jMax+2]) < 1.e-14).all()

        # bande horizontale en haut
        test = test and (abs(q[:, jMax+1:] - q[:, jMin:jMin+nGhost]) < 1.e-14).all()

        return test