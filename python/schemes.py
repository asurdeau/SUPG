# Modules généraux
import numpy as np

# Modules perso
import config
from config import XVEL, YVEL, PRES
import grid_mod




####################################################################################################################

#                                                 CLASSICAL UPWIND                                                 #

####################################################################################################################


def UPWIND_divFlux(q, grid):
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx, dy = grid.steps 
    
    divF = np.zeros((np.shape(q)))

    ### u component
    divF[iMin:iMax+1, jMin:jMax+1, XVEL] =  \
        0.5 / dx * \
        (1. * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
        - 1. * q[iMin-1:iMax, jMin:jMax+1, PRES]) \
        - 0.5 / dx * \
        (1. * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
        - 2. * q[iMin:iMax+1, jMin:jMax+1, XVEL]
        + 1. * q[iMin-1:iMax, jMin:jMax+1, XVEL]) \

    ### v component 
    divF[iMin:iMax+1, jMin:jMax+1, YVEL] =  \
        0.5 / dy * \
        (1. * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
        - 1. * q[iMin:iMax+1, jMin-1:jMax, PRES]) \
        - 0.5 / dy * \
        (1. * q[iMin:iMax+1, jMin+1:jMax+2, XVEL] \
        - 2. * q[iMin:iMax+1, jMin:jMax+1, XVEL]
        + 1. * q[iMin:iMax+1, jMin-1:jMax, XVEL]) \


    ### p component 
    divF[iMin:iMax+1, jMin:jMax+1, PRES] =  \
        0.5 / dx * \
        (1. * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
        - 1. * q[iMin-1:iMax, jMin:jMax+1, XVEL]) \
        + 0.5 / dy * \
        (1. * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
        - 1. * q[iMin:iMax+1, jMin-1:jMax, YVEL]) \
        - 0.5 / dx * \
        (1. * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
        - 2. * q[iMin:iMax+1, jMin:jMax+1, PRES]
        + 1. * q[iMin-1:iMax, jMin:jMax+1, PRES]) \
        - 0.5 / dy * \
        (1. * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
        - 2. * q[iMin:iMax+1, jMin:jMax+1, PRES]
        + 1. * q[iMin:iMax+1, jMin-1:jMax, PRES]) \

    return divF


####################################################################################################################

#                                                   SUPG STANDARD                                                  #

####################################################################################################################

def SUPG_developped_divFlux(q, grid):
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx, dy = grid.steps 
    
    divF = np.zeros((np.shape(q)))

    ### u component

    divF[iMin:iMax+1, jMin:jMax+1, XVEL] =  \
    1. / (dx * dy) * \
    ( \
    dx/2. * \
      ( - 1. * ( 1. * q[iMin+1:iMax+2, jMin+1:jMax+2, XVEL] \
                + 4. * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
                + 1. * q[iMin+1:iMax+2, jMin-1:jMax, XVEL]) \
        + 2. * ( 1. * q[iMin:iMax+1, jMin+1:jMax+2, XVEL] \
                + 4. * q[iMin:iMax+1, jMin:jMax+1, XVEL] \
                + 1. * q[iMin:iMax+1, jMin-1:jMax, XVEL]) \
        - 1. * ( 1. * q[iMin-1:iMax, jMin+1:jMax+2, XVEL] \
                + 4. * q[iMin-1:iMax, jMin:jMax+1, XVEL] \
                + 1. * q[iMin-1:iMax, jMin-1:jMax, XVEL]) \
      ) * dx * (dy / 6.) \
    \
    + dx/2. * \
      ( - 1. * ( 1. * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                - 1. * q[iMin+1:iMax+2, jMin-1:jMax, YVEL]) \
        + 1. * ( 1. * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
                - 1. * q[iMin-1:iMax, jMin-1:jMax, YVEL]) \
      ) * (1. / 2.) * (1. / 2.) \
    \
    + 1. * \
      ( + 1. * ( 1. * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                + 4. * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                + 1. * q[iMin+1:iMax+2, jMin-1:jMax, PRES]) \
        - 1. * ( 1. * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                + 4. * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                + 1. * q[iMin-1:iMax, jMin-1:jMax, PRES]) \
      ) * (1. / 2.) * (dy / 6. )  \
    ) 


    ### v component 

    divF[iMin:iMax+1, jMin:jMax+1, YVEL] =  \
    1. / (dx * dy) * \
    ( \
    dy/2. * \
      ( + 1. * (- 1. * q[iMin+1:iMax+2, jMin+1:jMax+2, XVEL] \
                + 1. * q[iMin+1:iMax+2, jMin-1:jMax, XVEL]) \
        - 1. * (- 1. * q[iMin-1:iMax, jMin+1:jMax+2, XVEL] \
                + 1. * q[iMin-1:iMax, jMin-1:jMax, XVEL]) \
      ) * (1. / 2.) * (1. / 2.) \
    \
    + dy/2. * \
      ( + 1. * (- 1. * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                + 2. * q[iMin+1:iMax+2, jMin:jMax+1, YVEL] \
                - 1. * q[iMin+1:iMax+2, jMin-1:jMax, YVEL]) \
        + 4. * (- 1. * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
                + 2. * q[iMin:iMax+1, jMin:jMax+1, YVEL] \
                - 1. * q[iMin:iMax+1, jMin-1:jMax, YVEL]) \
        + 1. * (- 1. * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
                + 2. * q[iMin-1:iMax, jMin:jMax+1, YVEL] \
                - 1. * q[iMin-1:iMax, jMin-1:jMax, YVEL]) \
      ) * (dx / 6.) * dy \
    \
    + 1. *  \
      ( + 1. * ( 1. * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                - 1. * q[iMin+1:iMax+2, jMin-1:jMax, PRES]) \
        + 4. * ( 1. * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                - 1. * q[iMin:iMax+1, jMin-1:jMax, PRES]) \
        + 1. * ( 1. * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                - 1. * q[iMin-1:iMax, jMin-1:jMax, PRES]) \
      ) * (dx / 6. ) * (1. / 2.) \
    )


    ### p component 

    divF[iMin:iMax+1, jMin:jMax+1, PRES] =  \
    1. / (dx * dy) * \
    ( \
    + 1. * \
      ( + 1. * ( 1. * q[iMin+1:iMax+2, jMin+1:jMax+2, XVEL] \
                + 4. * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
                + 1. * q[iMin+1:iMax+2, jMin-1:jMax, XVEL]) \
        - 1. * ( 1. * q[iMin-1:iMax, jMin+1:jMax+2, XVEL] \
                + 4. * q[iMin-1:iMax, jMin:jMax+1, XVEL] \
                + 1. * q[iMin-1:iMax, jMin-1:jMax, XVEL]) \
      ) * (1. / 2.) * (dy / 6. )  \
    \
    + 1. * \
      ( + 1. * ( 1. * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                - 1. * q[iMin+1:iMax+2, jMin-1:jMax, YVEL]) \
        + 4. * ( 1. * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
                - 1. * q[iMin:iMax+1, jMin-1:jMax, YVEL]) \
        + 1. * ( 1. * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
                - 1. * q[iMin-1:iMax, jMin-1:jMax, YVEL]) \
      ) * (dx / 6. ) * (1. / 2.) \
    \
    + dx/2. * \
      ( - 1. * ( 1. * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                + 4. * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                + 1. * q[iMin+1:iMax+2, jMin-1:jMax, PRES]) \
        + 2. * ( 1. * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                + 4. * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                + 1. * q[iMin:iMax+1, jMin-1:jMax, PRES]) \
        - 1. * ( 1. * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                + 4. * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                + 1. * q[iMin-1:iMax, jMin-1:jMax, PRES]) \
      ) * dx * (dy / 6.) \
    + dy/2. * \
      ( + 1. * (- 1. * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                + 2. * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                - 1. * q[iMin+1:iMax+2, jMin-1:jMax, PRES]) \
        + 4. * (- 1. * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                + 2. * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                - 1. * q[iMin:iMax+1, jMin-1:jMax, PRES]) \
        + 1. * (- 1. * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                + 2. * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                - 1. * q[iMin-1:iMax, jMin-1:jMax, PRES]) \
      ) * (dx / 6.) * dy \
    )

    return divF






####################################################################################################################

#                                                 CLASSICAL UPWIND                                                 #

####################################################################################################################


def getApproxDivFlux(q, grid, schemeChoice):
    if schemeChoice == 1 :
        return UPWIND_divFlux(q, grid)
    if schemeChoice == 2 :
        return SUPG_developped_divFlux(q, grid)