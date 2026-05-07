# Modules généraux
import numpy as np

# Modules perso
import config
from config import XVEL, YVEL, PRES
import grid_mod


####################################################################################################################

#                                                   SUPG STANDARD                                                  #

####################################################################################################################

def SUPG_developped_divFlux(q, grid):
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx, dy = grid.steps 
    
    divF = np.zeros((np.shape(q)))

    ### u component

    divF[iMin:iMax, jMin:jMax, XVEL] =  \
    1. / (dx * dy) * \
    ( \
    dx/2. * \
      ( - 1. * ( 1. * q[iMin+1:iMax+1, jMin+1:jMax+1, XVEL] \
                + 4. * q[iMin+1:iMax+1, jMin:jMax, XVEL] \
                + 1. * q[iMin+1:iMax+1, jMin-1:jMax-1, XVEL]) \
        + 2. * ( 1. * q[iMin:iMax, jMin+1:jMax+1, XVEL] \
                + 4. * q[iMin:iMax, jMin:jMax, XVEL] \
                + 1. * q[iMin:iMax, jMin-1:jMax-1, XVEL]) \
        - 1. * ( 1. * q[iMin-1:iMax-1, jMin+1:jMax+1, XVEL] \
                + 4. * q[iMin-1:iMax-1, jMin:jMax, XVEL] \
                + 1. * q[iMin-1:iMax-1, jMin-1:jMax-1, XVEL]) \
      ) * dx * (dy / 6.) \
    \
    + dx/2. * \
      ( - 1. * ( 1. * q[iMin+1:iMax+1, jMin+1:jMax+1, YVEL] \
                - 1. * q[iMin+1:iMax+1, jMin-1:jMax-1, YVEL]) \
        + 1. * ( 1. * q[iMin-1:iMax-1, jMin+1:jMax+1, YVEL] \
                - 1. * q[iMin-1:iMax-1, jMin-1:jMax-1, YVEL]) \
      ) * (1. / 2.) * (1. / 2.) \
    \
    + 1. * \
      ( + 1. * ( 1. * q[iMin+1:iMax+1, jMin+1:jMax+1, PRES] \
                + 4. * q[iMin+1:iMax+1, jMin:jMax, PRES] \
                + 1. * q[iMin+1:iMax+1, jMin-1:jMax-1, PRES]) \
        - 1. * ( 1. * q[iMin-1:iMax-1, jMin+1:jMax+1, PRES] \
                + 4. * q[iMin-1:iMax-1, jMin:jMax, PRES] \
                + 1. * q[iMin-1:iMax-1, jMin-1:jMax-1, PRES]) \
      ) * (1. / 2.) * (dy / 6. )  \
    ) 


    ### v component 

    divF[iMin:iMax, jMin:jMax, YVEL] =  \
    1. / (dx * dy) * \
    ( \
    dy/2. * \
      ( + 1. * (- 1. * q[iMin+1:iMax+1, jMin+1:jMax+1, XVEL] \
                + 1. * q[iMin+1:iMax+1, jMin-1:jMax-1, XVEL]) \
        - 1. * (- 1. * q[iMin-1:iMax-1, jMin+1:jMax+1, XVEL] \
                + 1. * q[iMin-1:iMax-1, jMin-1:jMax-1, XVEL]) \
      ) * (1. / 2.) * (1. / 2.) \
    \
    + dy/2. * \
      ( + 1. * (- 1. * q[iMin+1:iMax+1, jMin+1:jMax+1, YVEL] \
                + 2. * q[iMin+1:iMax+1, jMin:jMax, YVEL] \
                - 1. * q[iMin+1:iMax+1, jMin-1:jMax-1, YVEL]) \
        + 4. * (- 1. * q[iMin:iMax, jMin+1:jMax+1, YVEL] \
                + 2. * q[iMin:iMax, jMin:jMax, YVEL] \
                - 1. * q[iMin:iMax, jMin-1:jMax-1, YVEL]) \
        + 1. * (- 1. * q[iMin-1:iMax-1, jMin+1:jMax+1, YVEL] \
                + 2. * q[iMin-1:iMax-1, jMin:jMax, YVEL] \
                - 1. * q[iMin-1:iMax-1, jMin-1:jMax-1, YVEL]) \
      ) * (dx / 6.) * dy \
    \
    + 1. *  \
      ( + 1. * ( 1. * q[iMin+1:iMax+1, jMin+1:jMax+1, PRES] \
                - 1. * q[iMin+1:iMax+1, jMin-1:jMax-1, PRES]) \
        + 4. * ( 1. * q[iMin:iMax, jMin+1:jMax+1, PRES] \
                - 1. * q[iMin:iMax, jMin-1:jMax-1, PRES]) \
        + 1. * ( 1. * q[iMin-1:iMax-1, jMin+1:jMax+1, PRES] \
                - 1. * q[iMin-1:iMax-1, jMin-1:jMax-1, PRES]) \
      ) * (dx / 6. ) * (1. / 2.) \
    )


    ### p component 

    divF[iMin:iMax, jMin:jMax, PRES] =  \
    1. / (dx * dy) * \
    ( \
    + 1. * \
      ( + 1. * ( 1. * q[iMin+1:iMax+1, jMin+1:jMax+1, XVEL] \
                + 4. * q[iMin+1:iMax+1, jMin:jMax, XVEL] \
                + 1. * q[iMin+1:iMax+1, jMin-1:jMax-1, XVEL]) \
        - 1. * ( 1. * q[iMin-1:iMax-1, jMin+1:jMax+1, XVEL] \
                + 4. * q[iMin-1:iMax-1, jMin:jMax, XVEL] \
                + 1. * q[iMin-1:iMax-1, jMin-1:jMax-1, XVEL]) \
      ) * (1. / 2.) * (dy / 6. )  \
    \
    + 1. * \
      ( + 1. * ( 1. * q[iMin+1:iMax+1, jMin+1:jMax+1, YVEL] \
                - 1. * q[iMin+1:iMax+1, jMin-1:jMax-1, YVEL]) \
        + 4. * ( 1. * q[iMin:iMax, jMin+1:jMax+1, YVEL] \
                - 1. * q[iMin:iMax, jMin-1:jMax-1, YVEL]) \
        + 1. * ( 1. * q[iMin-1:iMax-1, jMin+1:jMax+1, YVEL] \
                - 1. * q[iMin-1:iMax-1, jMin-1:jMax-1, YVEL]) \
      ) * (dx / 6. ) * (1. / 2.) \
    \
    + dx/2. * \
      ( - 1. * ( 1. * q[iMin+1:iMax+1, jMin+1:jMax+1, PRES] \
                + 4. * q[iMin+1:iMax+1, jMin:jMax, PRES] \
                + 1. * q[iMin+1:iMax+1, jMin-1:jMax-1, PRES]) \
        + 2. * ( 1. * q[iMin:iMax, jMin+1:jMax+1, PRES] \
                + 4. * q[iMin:iMax, jMin:jMax, PRES] \
                + 1. * q[iMin:iMax, jMin-1:jMax-1, PRES]) \
        - 1. * ( 1. * q[iMin-1:iMax-1, jMin+1:jMax+1, PRES] \
                + 4. * q[iMin-1:iMax-1, jMin:jMax, PRES] \
                + 1. * q[iMin-1:iMax-1, jMin-1:jMax-1, PRES]) \
      ) * dx * (dy / 6.) \
    + dy/2. * \
      ( + 1. * (- 1. * q[iMin+1:iMax+1, jMin+1:jMax+1, PRES] \
                + 2. * q[iMin+1:iMax+1, jMin:jMax, PRES] \
                - 1. * q[iMin+1:iMax+1, jMin-1:jMax-1, PRES]) \
        + 4. * (- 1. * q[iMin:iMax, jMin+1:jMax+1, PRES] \
                + 2. * q[iMin:iMax, jMin:jMax, PRES] \
                - 1. * q[iMin:iMax, jMin-1:jMax-1, PRES]) \
        + 1. * (- 1. * q[iMin-1:iMax-1, jMin+1:jMax+1, PRES] \
                + 2. * q[iMin-1:iMax-1, jMin:jMax, PRES] \
                - 1. * q[iMin-1:iMax-1, jMin-1:jMax-1, PRES]) \
      ) * (dx / 6.) * dy \
    )

    return divF