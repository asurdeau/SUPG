# Modules généraux
import numpy as np
from numba import njit

# Modules perso
import config
from config import XVEL, YVEL, PRES
import grid_mod




####################################################################################################################

#                                                 CLASSICAL UPWIND                                                 #

####################################################################################################################



def get_UPWIND_divFlux(q, grid):
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
        - 2. * q[iMin:iMax+1, jMin:jMax+1, XVEL] \
        + 1. * q[iMin-1:iMax, jMin:jMax+1, XVEL])

    ### v component 
    divF[iMin:iMax+1, jMin:jMax+1, YVEL] =  \
        0.5 / dy * \
        (1. * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
        - 1. * q[iMin:iMax+1, jMin-1:jMax, PRES]) \
        - 0.5 / dy * \
        (1. * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
        - 2. * q[iMin:iMax+1, jMin:jMax+1, YVEL] \
        + 1. * q[iMin:iMax+1, jMin-1:jMax, YVEL])


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
        - 2. * q[iMin:iMax+1, jMin:jMax+1, PRES] \
        + 1. * q[iMin-1:iMax, jMin:jMax+1, PRES]) \
        - 0.5 / dy * \
        (1. * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
        - 2. * q[iMin:iMax+1, jMin:jMax+1, PRES] \
        + 1. * q[iMin:iMax+1, jMin-1:jMax, PRES])

    return divF



####################################################################################################################

#                                                   SUPG STANDARD                                                  #

####################################################################################################################



def get_SUPG_developped_divFlux(q, grid):
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
      ) * (1. / dx) * (dy / 6.) \
    \
    + dx/2. * \
      ( - 1. * (  1. * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                - 1. * q[iMin+1:iMax+2, jMin-1:jMax, YVEL]) \
        + 1. * (  1. * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
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
      ) * (dx / 6.) * (1. / dy) \
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
      ) * (1. / 2.) * (dy / 6.)  \
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
      ) * (1. / dx) * (dy / 6.) \
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
      ) * (dx / 6.) * (1. / dy) \
    )

    return divF



####################################################################################################################

#                                            MODIFS SUPG "A LA CARTE"                                              #

####################################################################################################################



def get_modif_SUPG_developped_divFlux(q, grid, operators):
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx, dy = grid.steps 

    uEq_uTerm = operators["u_equation"]["u_term"]
    uEq_vTerm = operators["u_equation"]["v_term"]
    uEq_pTerm = operators["u_equation"]["p_term"]

    vEq_uTerm = operators["v_equation"]["u_term"]
    vEq_vTerm = operators["v_equation"]["v_term"]
    vEq_pTerm = operators["v_equation"]["p_term"]

    pEq_uTerm = operators["p_equation"]["u_term"]
    pEq_vTerm = operators["p_equation"]["v_term"]
    pEq_pTerm = operators["p_equation"]["p_term"]

    # print("max(q)", np.max(abs(q[iMin:iMax+1,jMin:jMax+1,:])))


    
    divF = np.zeros((np.shape(q)))
    # print(uEq_uTerm["y"][0], uEq_uTerm["y"][1], uEq_uTerm["y"][2])
    # print(uEq_uTerm["x"][0], uEq_uTerm["x"][1], uEq_uTerm["x"][2])

    ### u equation

    divF[iMin:iMax+1, jMin:jMax+1, XVEL] =  \
    1. / (dx * dy) * \
    ( \
    dx/2. * \
      ( \
           uEq_uTerm["x"][0] * (  \
                                  uEq_uTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, XVEL] \
                                + uEq_uTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
                                + uEq_uTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, XVEL] \
                                ) \
        +  uEq_uTerm["x"][1] * (  \
                                  uEq_uTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, XVEL] \
                                + uEq_uTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, XVEL] \
                                + uEq_uTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, XVEL] \
                                ) \
        +  uEq_uTerm["x"][2] * (  \
                                  uEq_uTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, XVEL] \
                                + uEq_uTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, XVEL] \
                                + uEq_uTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, XVEL] \
                                ) \
      ) * (1. / dx) * (dy / sum(uEq_uTerm["y"])) \
    \
    + dx/2. * \
      (    uEq_vTerm["x"][0] * (  \
                                  uEq_vTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                                # + uEq_vTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, YVEL] \
                                + uEq_vTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, YVEL] \
                                ) \
        # +  uEq_vTerm["x"][1] * (  \
        #                           uEq_vTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
        #                         + uEq_vTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, YVEL] \
        #                         + uEq_vTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, YVEL] \
        #                         ) \
        +  uEq_vTerm["x"][2] * (  \
                                  uEq_vTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
                                # + uEq_vTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, YVEL] \
                                + uEq_vTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, YVEL] \
                                ) \
      ) \
    \
    + 1. * \
      (    uEq_pTerm["x"][0] * (  \
                                  uEq_pTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                + uEq_pTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                + uEq_pTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, PRES] \
                                ) \
        # +  uEq_pTerm["x"][1] * (  \
        #                           uEq_pTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
        #                         + uEq_pTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, PRES] \
        #                         + uEq_pTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, PRES] \
        #                         ) \
        +  uEq_pTerm["x"][2] * (  \
                                  uEq_pTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                + uEq_pTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                + uEq_pTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, PRES] \
                                ) \
      ) * (dy / sum(uEq_pTerm["y"]) )  \
    ) 


    ### v equation 

    divF[iMin:iMax+1, jMin:jMax+1, YVEL] =  \
    1. / (dx * dy) * \
    ( \
    dy/2. * \
      (    vEq_uTerm["x"][0] * (  \
                                  vEq_uTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, XVEL] \
                                # + vEq_uTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
                                + vEq_uTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, XVEL] \
                                ) \
        # +  vEq_uTerm["x"][1] * (  \
        #                           vEq_uTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, XVEL] \
        #                         + vEq_uTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, XVEL] \
        #                         + vEq_uTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, XVEL] \
                                # ) \
        +  vEq_uTerm["x"][2] * (  \
                                  vEq_uTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, XVEL] \
                                # + vEq_uTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, XVEL] \
                                + vEq_uTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, XVEL] \
                                ) \
      ) \
    \
    + dy/2. * \
      (    vEq_vTerm["x"][0] * (  \
                                  vEq_vTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                                + vEq_vTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, YVEL] \
                                + vEq_vTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, YVEL] \
                                ) \
        +  vEq_vTerm["x"][1] * (  \
                                  vEq_vTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
                                + vEq_vTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, YVEL] \
                                + vEq_vTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, YVEL] \
                                ) \
        +  vEq_vTerm["x"][2] * (  \
                                  vEq_vTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
                                + vEq_vTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, YVEL] \
                                + vEq_vTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, YVEL] \
                                ) \
      ) * (dx / sum(vEq_vTerm["x"])) * (1. / dy) \
    \
    + 1. * \
      (    vEq_pTerm["x"][0] * (  \
                                  vEq_pTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                # + vEq_pTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                + vEq_pTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, PRES] \
                                ) \
        +  vEq_pTerm["x"][1] * (  \
                                  vEq_pTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                                # + vEq_pTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                                + vEq_pTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, PRES] \
                                ) \
        +  vEq_pTerm["x"][2] * (  \
                                  vEq_pTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                # + vEq_pTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                + vEq_pTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, PRES] \
                                ) \
      ) * (dx / sum(vEq_pTerm["x"]) )  \
    ) 


    ### p equation 

    divF[iMin:iMax+1, jMin:jMax+1, PRES] =  \
    1. / (dx * dy) * \
    ( \
    1. * \
      (    pEq_uTerm["x"][0] * (  \
                                  pEq_uTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, XVEL] \
                                + pEq_uTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
                                + pEq_uTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, XVEL] \
                                ) \
        # +  pEq_uTerm["x"][1] * (  \
        #                           pEq_uTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, XVEL] \
        #                         + pEq_uTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, XVEL] \
        #                         + pEq_uTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, XVEL] \
        #                         ) \
        +  pEq_uTerm["x"][2] * (  \
                                  pEq_uTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, XVEL] \
                                + pEq_uTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, XVEL] \
                                + pEq_uTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, XVEL] \
                                ) \
      ) * (dy / sum(pEq_uTerm["y"]) )  \
    \
    + 1. * \
      (    pEq_vTerm["x"][0] * (  \
                                  pEq_vTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                                # + pEq_vTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, YVEL] \
                                + pEq_vTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, YVEL] \
                                ) \
        +  pEq_vTerm["x"][1] * (  \
                                  pEq_vTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
                                # + pEq_vTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, YVEL] \
                                + pEq_vTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, YVEL] \
                                ) \
        +  pEq_vTerm["x"][2] * (  \
                                  pEq_vTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
                                # + pEq_vTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, YVEL] \
                                + pEq_vTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, YVEL] \
                                ) \
      ) * (dx / sum(pEq_vTerm["x"]) )  \
    \
    + dx/2. * \
      (    pEq_pTerm["second_der"][0] * (  \
                                           pEq_pTerm["mass"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                         + pEq_pTerm["mass"][1] * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                         + pEq_pTerm["mass"][2] * q[iMin+1:iMax+2, jMin-1:jMax, PRES] \
                                         ) \
        +  pEq_pTerm["second_der"][1] * (  \
                                           pEq_pTerm["mass"][0] * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                                         + pEq_pTerm["mass"][1] * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                                         + pEq_pTerm["mass"][2] * q[iMin:iMax+1, jMin-1:jMax, PRES] \
                                         ) \
        +  pEq_pTerm["second_der"][2] * (  \
                                           pEq_pTerm["mass"][0] * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                         + pEq_pTerm["mass"][1] * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                         + pEq_pTerm["mass"][2] * q[iMin-1:iMax, jMin-1:jMax, PRES] \
                                         ) \
      ) * (1. / dx) * (dy / sum(pEq_pTerm["mass"])) \
    + dy/2. * \
      (    pEq_pTerm["mass"][0] * (  \
                                     pEq_pTerm["second_der"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                   + pEq_pTerm["second_der"][1] * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                   + pEq_pTerm["second_der"][2] * q[iMin+1:iMax+2, jMin-1:jMax, PRES] \
                                    ) \
        +  pEq_pTerm["mass"][1] * (  \
                                     pEq_pTerm["second_der"][0] * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                                   + pEq_pTerm["second_der"][1] * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                                   + pEq_pTerm["second_der"][2] * q[iMin:iMax+1, jMin-1:jMax, PRES] \
                                   ) \
        +  pEq_pTerm["mass"][2] * (  \
                                     pEq_pTerm["second_der"][0] * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                   + pEq_pTerm["second_der"][1] * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                   + pEq_pTerm["second_der"][2] * q[iMin-1:iMax, jMin-1:jMax, PRES] \
                                   ) \
      ) * (dx / sum(pEq_pTerm["mass"])) * (1. / dy) \
    )


    return divF



def get_optim_modif_SUPG_developped_divFlux(q, grid, operators):
    iMin, iMax, jMin, jMax = grid.valid_grid
    dx, dy = grid.steps

    # allocation de variables pour chaque coeff
    # u equation
    uEq_uTerm_xp1, uEq_uTerm_x, uEq_uTerm_xm1 = operators["u_equation"]["u_term"]["x"]
    uEq_uTerm_yp1, uEq_uTerm_y, uEq_uTerm_ym1 = operators["u_equation"]["u_term"]["y"]

    uEq_vTerm_xp1, uEq_vTerm_x, uEq_vTerm_xm1 = operators["u_equation"]["v_term"]["x"]
    uEq_vTerm_yp1, uEq_vTerm_y, uEq_vTerm_ym1 = operators["u_equation"]["v_term"]["y"]

    uEq_pTerm_xp1, uEq_pTerm_x, uEq_pTerm_xm1 = operators["u_equation"]["p_term"]["x"]
    uEq_pTerm_yp1, uEq_pTerm_y, uEq_pTerm_ym1 = operators["u_equation"]["p_term"]["y"]

    # v equation
    vEq_uTerm_xp1, vEq_uTerm_x, vEq_uTerm_xm1 = operators["v_equation"]["u_term"]["x"]
    vEq_uTerm_yp1, vEq_uTerm_y, vEq_uTerm_ym1 = operators["v_equation"]["u_term"]["y"]

    vEq_vTerm_xp1, vEq_vTerm_x, vEq_vTerm_xm1 = operators["v_equation"]["v_term"]["x"]
    vEq_vTerm_yp1, vEq_vTerm_y, vEq_vTerm_ym1 = operators["v_equation"]["v_term"]["y"]

    vEq_pTerm_xp1, vEq_pTerm_x, vEq_pTerm_xm1 = operators["v_equation"]["p_term"]["x"]
    vEq_pTerm_yp1, vEq_pTerm_y, vEq_pTerm_ym1 = operators["v_equation"]["p_term"]["y"]

    # p equation
    pEq_uTerm_xp1, pEq_uTerm_x, pEq_uTerm_xm1 = operators["p_equation"]["u_term"]["x"]
    pEq_uTerm_yp1, pEq_uTerm_y, pEq_uTerm_ym1 = operators["p_equation"]["u_term"]["y"]

    pEq_vTerm_xp1, pEq_vTerm_x, pEq_vTerm_xm1 = operators["p_equation"]["v_term"]["x"]
    pEq_vTerm_yp1, pEq_vTerm_y, pEq_vTerm_ym1 = operators["p_equation"]["v_term"]["y"]

    pEq_pTerm_sdp1, pEq_pTerm_sd, pEq_pTerm_sdm1 = operators["p_equation"]["p_term"]["second_der"]
    pEq_pTerm_mp1, pEq_pTerm_m, pEq_pTerm_mm1 = operators["p_equation"]["p_term"]["mass"]

    
    
    
    # # allocations de slices au départ :
    # q_ip1_jp1 = q[iMin+1:iMax+2, jMin+1:jMax+2, :]
    # q_ip1_j = q[iMin+1:iMax+2, jMin:jMax+1, :]
    # q_ip1_jm1 = q[iMin+1:iMax+2, jMin-1:jMax, :]

    # q_i_jp1 = q[iMin:iMax+1, jMin+1:jMax+2, :]
    # q_i_j = q[iMin:iMax+1, jMin:jMax+1, :]
    # q_i_jm1 = q[iMin:iMax+1, jMin-1:jMax, :]

    # q_im1_jp1 = q[iMin-1:iMax, jMin+1:jMax+2, :]
    # q_im1_j = q[iMin-1:iMax, jMin:jMax+1, :]
    # q_im1_jm1 = q[iMin-1:iMax, jMin-1:jMax, :]


    
    divF = np.zeros((np.shape(q)))
    # print(uEq_uTerm["y"][0], uEq_uTerm["y"][1], uEq_uTerm["y"][2])
    # print(uEq_uTerm["x"][0], uEq_uTerm["x"][1], uEq_uTerm["x"][2])

    ### u equation

    divF[iMin:iMax+1, jMin:jMax+1, XVEL] =  \
    1. / (dx * dy) * \
    ( \
    dx/2. * \
      ( \
           uEq_uTerm_xp1 * (  \
                                  uEq_uTerm_yp1 * q[iMin+1:iMax+2, jMin+1:jMax+2, XVEL] \
                                + uEq_uTerm_y * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
                                + uEq_uTerm_ym1 * q[iMin+1:iMax+2, jMin-1:jMax, XVEL] \
                                ) \
        +  uEq_uTerm_x * (  \
                                  uEq_uTerm_yp1 * q[iMin:iMax+1, jMin+1:jMax+2, XVEL] \
                                + uEq_uTerm_y * q[iMin:iMax+1, jMin:jMax+1, XVEL] \
                                + uEq_uTerm_ym1 * q[iMin:iMax+1, jMin-1:jMax, XVEL] \
                                ) \
        +  uEq_uTerm_xm1 * (  \
                                  uEq_uTerm_yp1 * q[iMin-1:iMax, jMin+1:jMax+2, XVEL] \
                                + uEq_uTerm_y * q[iMin-1:iMax, jMin:jMax+1, XVEL] \
                                + uEq_uTerm_ym1 * q[iMin-1:iMax, jMin-1:jMax, XVEL] \
                                ) \
      ) * (1. / dx) * (dy / (uEq_uTerm_yp1 + uEq_uTerm_y + uEq_uTerm_ym1)) \
    \
    + dx/2. * \
      (    uEq_vTerm_xp1 * (  \
                                  uEq_vTerm_yp1 * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                                # + uEq_vTerm_y * q[iMin+1:iMax+2, jMin:jMax+1, YVEL] \
                                + uEq_vTerm_ym1 * q[iMin+1:iMax+2, jMin-1:jMax, YVEL] \
                                ) \
        # +  uEq_vTerm_x * (  \
        #                           uEq_vTerm_yp1 * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
        #                         + uEq_vTerm_y * q[iMin:iMax+1, jMin:jMax+1, YVEL] \
        #                         + uEq_vTerm_ym1 * q[iMin:iMax+1, jMin-1:jMax, YVEL] \
        #                         ) \
        +  uEq_vTerm_xm1 * (  \
                                  uEq_vTerm_yp1 * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
                                # + uEq_vTerm_y * q[iMin-1:iMax, jMin:jMax+1, YVEL] \
                                + uEq_vTerm_ym1 * q[iMin-1:iMax, jMin-1:jMax, YVEL] \
                                ) \
      ) \
    \
    + 1. * \
      (    uEq_pTerm_xp1 * (  \
                                  uEq_pTerm_yp1 * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                + uEq_pTerm_y * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                + uEq_pTerm_ym1 * q[iMin+1:iMax+2, jMin-1:jMax, PRES] \
                                ) \
        # +  uEq_pTerm_x * (  \
        #                           uEq_pTerm_yp1 * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
        #                         + uEq_pTerm_y * q[iMin:iMax+1, jMin:jMax+1, PRES] \
        #                         + uEq_pTerm_ym1 * q[iMin:iMax+1, jMin-1:jMax, PRES] \
        #                         ) \
        +  uEq_pTerm_xm1 * (  \
                                  uEq_pTerm_yp1 * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                + uEq_pTerm_y * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                + uEq_pTerm_ym1 * q[iMin-1:iMax, jMin-1:jMax, PRES] \
                                ) \
      ) * (dy / (uEq_pTerm_yp1 + uEq_pTerm_y + uEq_pTerm_ym1) )  \
    ) 


    ### v equation 

    divF[iMin:iMax+1, jMin:jMax+1, YVEL] =  \
    1. / (dx * dy) * \
    ( \
    dy/2. * \
      (    vEq_uTerm_xp1 * (  \
                                  vEq_uTerm_yp1 * q[iMin+1:iMax+2, jMin+1:jMax+2, XVEL] \
                                # + vEq_uTerm_y * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
                                + vEq_uTerm_ym1 * q[iMin+1:iMax+2, jMin-1:jMax, XVEL] \
                                ) \
        # +  vEq_uTerm_x * (  \
        #                           vEq_uTerm_yp1 * q[iMin:iMax+1, jMin+1:jMax+2, XVEL] \
        #                         + vEq_uTerm_y * q[iMin:iMax+1, jMin:jMax+1, XVEL] \
        #                         + vEq_uTerm_ym1 * q[iMin:iMax+1, jMin-1:jMax, XVEL] \
                                # ) \
        +  vEq_uTerm_xm1 * (  \
                                  vEq_uTerm_yp1 * q[iMin-1:iMax, jMin+1:jMax+2, XVEL] \
                                # + vEq_uTerm_y * q[iMin-1:iMax, jMin:jMax+1, XVEL] \
                                + vEq_uTerm_ym1 * q[iMin-1:iMax, jMin-1:jMax, XVEL] \
                                ) \
      ) \
    \
    + dy/2. * \
      (    vEq_vTerm_xp1 * (  \
                                  vEq_vTerm_yp1 * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                                + vEq_vTerm_y * q[iMin+1:iMax+2, jMin:jMax+1, YVEL] \
                                + vEq_vTerm_ym1 * q[iMin+1:iMax+2, jMin-1:jMax, YVEL] \
                                ) \
        +  vEq_vTerm_x * (  \
                                  vEq_vTerm_yp1 * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
                                + vEq_vTerm_y * q[iMin:iMax+1, jMin:jMax+1, YVEL] \
                                + vEq_vTerm_ym1 * q[iMin:iMax+1, jMin-1:jMax, YVEL] \
                                ) \
        +  vEq_vTerm_xm1 * (  \
                                  vEq_vTerm_yp1 * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
                                + vEq_vTerm_y * q[iMin-1:iMax, jMin:jMax+1, YVEL] \
                                + vEq_vTerm_ym1 * q[iMin-1:iMax, jMin-1:jMax, YVEL] \
                                ) \
      ) * (dx / (vEq_vTerm_xp1 + vEq_vTerm_x + vEq_vTerm_xm1)) * (1. / dy) \
    \
    + 1. * \
      (    vEq_pTerm_xp1 * (  \
                                  vEq_pTerm_yp1 * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                # + vEq_pTerm_y * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                + vEq_pTerm_ym1 * q[iMin+1:iMax+2, jMin-1:jMax, PRES] \
                                ) \
        +  vEq_pTerm_x * (  \
                                  vEq_pTerm_yp1 * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                                # + vEq_pTerm_y * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                                + vEq_pTerm_ym1 * q[iMin:iMax+1, jMin-1:jMax, PRES] \
                                ) \
        +  vEq_pTerm_xm1 * (  \
                                  vEq_pTerm_yp1 * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                # + vEq_pTerm_y * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                + vEq_pTerm_ym1 * q[iMin-1:iMax, jMin-1:jMax, PRES] \
                                ) \
      ) * (dx / (vEq_pTerm_xp1 + vEq_pTerm_x + vEq_pTerm_xm1) )  \
    ) 


    ### p equation 

    divF[iMin:iMax+1, jMin:jMax+1, PRES] =  \
    1. / (dx * dy) * \
    ( \
    1. * \
      (    pEq_uTerm_xp1 * (  \
                                  pEq_uTerm_yp1 * q[iMin+1:iMax+2, jMin+1:jMax+2, XVEL] \
                                + pEq_uTerm_y * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
                                + pEq_uTerm_ym1 * q[iMin+1:iMax+2, jMin-1:jMax, XVEL] \
                                ) \
        # +  pEq_uTerm_x * (  \
        #                           pEq_uTerm_yp1 * q[iMin:iMax+1, jMin+1:jMax+2, XVEL] \
        #                         + pEq_uTerm_y * q[iMin:iMax+1, jMin:jMax+1, XVEL] \
        #                         + pEq_uTerm_ym1 * q[iMin:iMax+1, jMin-1:jMax, XVEL] \
        #                         ) \
        +  pEq_uTerm_xm1 * (  \
                                  pEq_uTerm_yp1 * q[iMin-1:iMax, jMin+1:jMax+2, XVEL] \
                                + pEq_uTerm_y * q[iMin-1:iMax, jMin:jMax+1, XVEL] \
                                + pEq_uTerm_ym1 * q[iMin-1:iMax, jMin-1:jMax, XVEL] \
                                ) \
      ) * (dy / (pEq_uTerm_yp1 + pEq_uTerm_y + pEq_uTerm_ym1) )  \
    \
    + 1. * \
      (    pEq_vTerm_xp1 * (  \
                                  pEq_vTerm_yp1 * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                                # + pEq_vTerm_y * q[iMin+1:iMax+2, jMin:jMax+1, YVEL] \
                                + pEq_vTerm_ym1 * q[iMin+1:iMax+2, jMin-1:jMax, YVEL] \
                                ) \
        +  pEq_vTerm_x * (  \
                                  pEq_vTerm_yp1 * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
                                # + pEq_vTerm_y * q[iMin:iMax+1, jMin:jMax+1, YVEL] \
                                + pEq_vTerm_ym1 * q[iMin:iMax+1, jMin-1:jMax, YVEL] \
                                ) \
        +  pEq_vTerm_xm1 * (  \
                                  pEq_vTerm_yp1 * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
                                # + pEq_vTerm_y * q[iMin-1:iMax, jMin:jMax+1, YVEL] \
                                + pEq_vTerm_ym1 * q[iMin-1:iMax, jMin-1:jMax, YVEL] \
                                ) \
      ) * (dx / (pEq_vTerm_xp1 + pEq_vTerm_x + pEq_vTerm_xm1) )  \
    \
    + dx/2. * \
      (    pEq_pTerm_sdp1 * (  \
                                           pEq_pTerm_mp1 * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                         + pEq_pTerm_m * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                         + pEq_pTerm_mm1 * q[iMin+1:iMax+2, jMin-1:jMax, PRES] \
                                         ) \
        +  pEq_pTerm_sd * (  \
                                           pEq_pTerm_mp1 * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                                         + pEq_pTerm_m * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                                         + pEq_pTerm_mm1 * q[iMin:iMax+1, jMin-1:jMax, PRES] \
                                         ) \
        +  pEq_pTerm_sdm1 * (  \
                                           pEq_pTerm_mp1 * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                         + pEq_pTerm_m * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                         + pEq_pTerm_mm1 * q[iMin-1:iMax, jMin-1:jMax, PRES] \
                                         ) \
      ) * (1. / dx) * (dy / (pEq_pTerm_mp1 + pEq_pTerm_m + pEq_pTerm_mm1)) \
    + dy/2. * \
      (    pEq_pTerm_mp1 * (  \
                                     pEq_pTerm_sdp1 * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                   + pEq_pTerm_sd * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                   + pEq_pTerm_sdm1 * q[iMin+1:iMax+2, jMin-1:jMax, PRES] \
                                    ) \
        +  pEq_pTerm_m * (  \
                                     pEq_pTerm_sdp1 * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                                   + pEq_pTerm_sd * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                                   + pEq_pTerm_sdm1 * q[iMin:iMax+1, jMin-1:jMax, PRES] \
                                   ) \
        +  pEq_pTerm_mm1 * (  \
                                     pEq_pTerm_sdp1 * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                   + pEq_pTerm_sd * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                   + pEq_pTerm_sdm1 * q[iMin-1:iMax, jMin-1:jMax, PRES] \
                                   ) \
      ) * (dx / (pEq_pTerm_mp1 + pEq_pTerm_m + pEq_pTerm_mm1)) * (1. / dy) \
    )


    return divF





@njit
def get_numba_modif_SUPG_developped_divFlux(q, iMin, iMax, jMin, jMax, dx, dy, coeffs):
    # allocation de variables pour chaque coeff
    # u equation - u term
    uEq_uTerm_xp1, uEq_uTerm_x, uEq_uTerm_xm1 = coeffs[0, 0, 0, :]
    uEq_uTerm_yp1, uEq_uTerm_y, uEq_uTerm_ym1 = coeffs[0, 0, 1, :]

    # u equation - v term
    uEq_vTerm_xp1, uEq_vTerm_x, uEq_vTerm_xm1 = coeffs[0, 1, 0, :]
    uEq_vTerm_yp1, uEq_vTerm_y, uEq_vTerm_ym1 = coeffs[0, 1, 1, :]

    # u equation - p term
    uEq_pTerm_xp1, uEq_pTerm_x, uEq_pTerm_xm1 = coeffs[0, 2, 0, :]
    uEq_pTerm_yp1, uEq_pTerm_y, uEq_pTerm_ym1 = coeffs[0, 2, 1, :]

    # v equation - u term
    vEq_uTerm_xp1, vEq_uTerm_x, vEq_uTerm_xm1 = coeffs[1, 0, 0, :]
    vEq_uTerm_yp1, vEq_uTerm_y, vEq_uTerm_ym1 = coeffs[1, 0, 1, :]

    # v equation - v term
    vEq_vTerm_xp1, vEq_vTerm_x, vEq_vTerm_xm1 = coeffs[1, 1, 0, :]
    vEq_vTerm_yp1, vEq_vTerm_y, vEq_vTerm_ym1 = coeffs[1, 1, 1, :]

    # v equation - p term
    vEq_pTerm_xp1, vEq_pTerm_x, vEq_pTerm_xm1 = coeffs[1, 2, 0, :]
    vEq_pTerm_yp1, vEq_pTerm_y, vEq_pTerm_ym1 = coeffs[1, 2, 1, :]

    # p equation - u term
    pEq_uTerm_xp1, pEq_uTerm_x, pEq_uTerm_xm1 = coeffs[2, 0, 0, :]
    pEq_uTerm_yp1, pEq_uTerm_y, pEq_uTerm_ym1 = coeffs[2, 0, 1, :]

    # p equation - v term
    pEq_vTerm_xp1, pEq_vTerm_x, pEq_vTerm_xm1 = coeffs[2, 1, 0, :]
    pEq_vTerm_yp1, pEq_vTerm_y, pEq_vTerm_ym1 = coeffs[2, 1, 1, :]

    # p equation - p term (cas spécial avec mass et second_der)
    pEq_pTerm_mp1, pEq_pTerm_m, pEq_pTerm_mm1 = coeffs[2, 2, 2, :]
    pEq_pTerm_sdp1, pEq_pTerm_sd, pEq_pTerm_sdm1 = coeffs[2, 2, 3, :]


    
    divF = np.zeros((np.shape(q)))


    # u equation
    divF[iMin : iMax+1, jMin : jMax+1, XVEL] =  \
      1. / (dx * dy) * \
      ( \
      dx/2. * \
        ( uEq_uTerm_xp1 * (uEq_uTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, XVEL] + uEq_uTerm_y * q[iMin+1 : iMax+2, jMin : jMax+1, XVEL] + uEq_uTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, XVEL]) \
        + uEq_uTerm_x   * (uEq_uTerm_yp1 * q[iMin   : iMax+1, jMin+1 : jMax+2, XVEL] + uEq_uTerm_y * q[iMin   : iMax+1, jMin : jMax+1, XVEL] + uEq_uTerm_ym1 * q[iMin   : iMax+1, jMin-1 : jMax, XVEL]) \
        + uEq_uTerm_xm1 * (uEq_uTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, XVEL] + uEq_uTerm_y * q[iMin-1 : iMax  , jMin : jMax+1, XVEL] + uEq_uTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, XVEL]) \
        ) * (1. / dx) * (dy / (uEq_uTerm_yp1 + uEq_uTerm_y + uEq_uTerm_ym1)) \
      \
      + dx/2. * \
        ( uEq_vTerm_xp1 * (uEq_vTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, YVEL] + uEq_vTerm_y * q[iMin+1 : iMax+2, jMin : jMax+1, YVEL] + uEq_vTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, YVEL]) \
        + uEq_vTerm_x   * (uEq_vTerm_yp1 * q[iMin   : iMax+1, jMin+1 : jMax+2, YVEL] + uEq_vTerm_y * q[iMin   : iMax+1, jMin : jMax+1, YVEL] + uEq_vTerm_ym1 * q[iMin   : iMax+1, jMin-1 : jMax, YVEL]) \
        + uEq_vTerm_xm1 * (uEq_vTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, YVEL] + uEq_vTerm_y * q[iMin-1 : iMax  , jMin : jMax+1, YVEL] + uEq_vTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, YVEL]) \
        ) \
      # + dx/2. * \
      #   ( uEq_vTerm_xp1 * (uEq_vTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, YVEL] + uEq_vTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, YVEL]) \
      #   + uEq_vTerm_xm1 * (uEq_vTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, YVEL] + uEq_vTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, YVEL]) \
      #   ) \
      + 1.* \
        ( uEq_pTerm_xp1 * (uEq_pTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, PRES] + uEq_pTerm_y * q[iMin+1 : iMax+2, jMin : jMax+1, PRES] + uEq_pTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, PRES]) \
        + uEq_pTerm_x   * (uEq_pTerm_yp1 * q[iMin   : iMax+1, jMin+1 : jMax+2, PRES] + uEq_pTerm_y * q[iMin   : iMax+1, jMin : jMax+1, PRES] + uEq_pTerm_ym1 * q[iMin   : iMax+1, jMin-1 : jMax, PRES]) \
        + uEq_pTerm_xm1 * (uEq_pTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, PRES] + uEq_pTerm_y * q[iMin-1 : iMax  , jMin : jMax+1, PRES] + uEq_pTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, PRES]) \
        ) * (dy / (uEq_pTerm_yp1 + uEq_pTerm_y + uEq_pTerm_ym1)) \
      # + 1.* \
      #   ( uEq_pTerm_xp1 * (uEq_pTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, PRES] + uEq_pTerm_y * q[iMin+1 : iMax+2, jMin : jMax+1, PRES] + uEq_pTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, PRES]) \
      #   + uEq_pTerm_xm1 * (uEq_pTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, PRES] + uEq_pTerm_y * q[iMin-1 : iMax  , jMin : jMax+1, PRES] + uEq_pTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, PRES]) \
      #   ) * (1. / dx) * (dy / (uEq_pTerm_yp1 + uEq_pTerm_y + uEq_pTerm_ym1)) \
      )
      


    ### v equation 

    divF[iMin : iMax+1, jMin : jMax+1, YVEL] =  \
      1. / (dx * dy) * \
      ( \
      dy/2. * \
        ( vEq_uTerm_xp1 * (vEq_uTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, XVEL] + vEq_uTerm_y * q[iMin+1 : iMax+2, jMin : jMax+1, XVEL] + vEq_uTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, XVEL]) \
        + vEq_uTerm_x   * (vEq_uTerm_yp1 * q[iMin   : iMax+1, jMin+1 : jMax+2, XVEL] + vEq_uTerm_y * q[iMin   : iMax+1, jMin : jMax+1, XVEL] + vEq_uTerm_ym1 * q[iMin   : iMax+1, jMin-1 : jMax, XVEL]) \
        + vEq_uTerm_xm1 * (vEq_uTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, XVEL] + vEq_uTerm_y * q[iMin-1 : iMax  , jMin : jMax+1, XVEL] + vEq_uTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, XVEL]) \
        )
      # + dy/2. * \
      #   ( vEq_uTerm_xp1 * (vEq_uTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, XVEL] + vEq_uTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, XVEL]) \
      #   + vEq_uTerm_xm1 * (vEq_uTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, XVEL] + vEq_uTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, XVEL]) \
      #   ) \
      \
      + dy/2. * \
        ( vEq_vTerm_xp1 * (vEq_vTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, YVEL] + vEq_vTerm_y * q[iMin+1 : iMax+2, jMin : jMax+1, YVEL] + vEq_vTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, YVEL]) \
        + vEq_vTerm_x   * (vEq_vTerm_yp1 * q[iMin   : iMax+1, jMin+1 : jMax+2, YVEL] + vEq_vTerm_y * q[iMin   : iMax+1, jMin : jMax+1, YVEL] + vEq_vTerm_ym1 * q[iMin   : iMax+1, jMin-1 : jMax, YVEL]) \
        + vEq_vTerm_xm1 * (vEq_vTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, YVEL] + vEq_vTerm_y * q[iMin-1 : iMax  , jMin : jMax+1, YVEL] + vEq_vTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, YVEL]) \
        ) * (1. / dy) * (dx / (vEq_vTerm_xp1 + vEq_vTerm_x + vEq_vTerm_xm1)) \
      + 1.* \
        ( vEq_pTerm_xp1 * (vEq_pTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, PRES] + vEq_pTerm_y * q[iMin+1 : iMax+2, jMin : jMax+1, PRES] + vEq_pTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, PRES]) \
        + vEq_pTerm_x   * (vEq_pTerm_yp1 * q[iMin   : iMax+1, jMin+1 : jMax+2, PRES] + vEq_pTerm_y * q[iMin   : iMax+1, jMin : jMax+1, PRES] + vEq_pTerm_ym1 * q[iMin   : iMax+1, jMin-1 : jMax, PRES]) \
        + vEq_pTerm_xm1 * (vEq_pTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, PRES] + vEq_pTerm_y * q[iMin-1 : iMax  , jMin : jMax+1, PRES] + vEq_pTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, PRES]) \
        ) * (dx / (vEq_pTerm_xp1 + vEq_pTerm_x + vEq_pTerm_xm1)) \
      # + 1.* \
      #   ( vEq_pTerm_xp1 * (vEq_pTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, PRES] + vEq_pTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, PRES]) \
      #   + vEq_pTerm_x   * (vEq_pTerm_yp1 * q[iMin   : iMax+1, jMin+1 : jMax+2, PRES] + vEq_pTerm_ym1 * q[iMin   : iMax+1, jMin-1 : jMax, PRES]) \
      #   + vEq_pTerm_xm1 * (vEq_pTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, PRES] + vEq_pTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, PRES]) \
      #   ) * (1. / dx) * (dy / (vEq_pTerm_yp1 + vEq_pTerm_y + vEq_pTerm_ym1)) \
      ) 


    ### p equation 

    divF[iMin : iMax+1, jMin : jMax+1, PRES] =  \
      1. / (dx * dy) * \
      ( \
      1.* \
        ( pEq_uTerm_xp1 * (pEq_uTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, XVEL] + pEq_uTerm_y * q[iMin+1 : iMax+2, jMin : jMax+1, XVEL] + pEq_uTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, XVEL]) \
        + pEq_uTerm_x   * (pEq_uTerm_yp1 * q[iMin   : iMax+1, jMin+1 : jMax+2, XVEL] + pEq_uTerm_y * q[iMin   : iMax+1, jMin : jMax+1, XVEL] + pEq_uTerm_ym1 * q[iMin   : iMax+1, jMin-1 : jMax, XVEL]) \
        + pEq_uTerm_xm1 * (pEq_uTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, XVEL] + pEq_uTerm_y * q[iMin-1 : iMax  , jMin : jMax+1, XVEL] + pEq_uTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, XVEL]) \
        ) * (dy / (pEq_uTerm_yp1 + pEq_uTerm_y + pEq_uTerm_ym1)) \
      # + 1.* \
      #   ( pEq_uTerm_xp1 * (pEq_uTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, XVEL] + pEq_uTerm_y * q[iMin+1 : iMax+2, jMin : jMax+1, XVEL] + pEq_uTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, XVEL]) \
      #   + pEq_uTerm_xm1 * (pEq_uTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, XVEL] + pEq_uTerm_y * q[iMin-1 : iMax  , jMin : jMax+1, XVEL] + pEq_uTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, XVEL]) \
      #   ) * (1. / dx) * (dy / (pEq_uTerm_yp1 + pEq_uTerm_y + pEq_uTerm_ym1)) \
      + 1.* \
        ( pEq_vTerm_xp1 * (pEq_vTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, YVEL] + pEq_vTerm_y * q[iMin+1 : iMax+2, jMin : jMax+1, YVEL] + pEq_vTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, YVEL]) \
        + pEq_vTerm_x   * (pEq_vTerm_yp1 * q[iMin   : iMax+1, jMin+1 : jMax+2, YVEL] + pEq_vTerm_y * q[iMin   : iMax+1, jMin : jMax+1, YVEL] + pEq_vTerm_ym1 * q[iMin   : iMax+1, jMin-1 : jMax, YVEL]) \
        + pEq_vTerm_xm1 * (pEq_vTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, YVEL] + pEq_vTerm_y * q[iMin-1 : iMax  , jMin : jMax+1, YVEL] + pEq_vTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, YVEL]) \
        ) * (dx / (pEq_vTerm_xp1 + pEq_vTerm_x + pEq_vTerm_xm1)) \
      # + 1.* \
      #   ( pEq_vTerm_xp1 * (pEq_vTerm_yp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, YVEL] + pEq_vTerm_ym1 * q[iMin+1 : iMax+2, jMin-1 : jMax, YVEL]) \
      #   + pEq_vTerm_x   * (pEq_vTerm_yp1 * q[iMin   : iMax+1, jMin+1 : jMax+2, YVEL] + pEq_vTerm_ym1 * q[iMin   : iMax+1, jMin-1 : jMax, YVEL]) \
      #   + pEq_vTerm_xm1 * (pEq_vTerm_yp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, YVEL] + pEq_vTerm_ym1 * q[iMin-1 : iMax  , jMin-1 : jMax, YVEL]) \
      #   ) * (1. / dx) * (dy / (pEq_vTerm_yp1 + pEq_vTerm_y + pEq_vTerm_ym1)) \
      \
      + dx/2. * \
        ( pEq_pTerm_sdp1 * (pEq_pTerm_mp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, PRES] + pEq_pTerm_m * q[iMin+1 : iMax+2, jMin : jMax+1, PRES] + pEq_pTerm_mm1 * q[iMin+1 : iMax+2, jMin-1 : jMax, PRES]) \
        + pEq_pTerm_sd   * (pEq_pTerm_mp1 * q[iMin   : iMax+1, jMin+1 : jMax+2, PRES] + pEq_pTerm_m * q[iMin   : iMax+1, jMin : jMax+1, PRES] + pEq_pTerm_mm1 * q[iMin   : iMax+1, jMin-1 : jMax, PRES]) \
        + pEq_pTerm_sdm1 * (pEq_pTerm_mp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, PRES] + pEq_pTerm_m * q[iMin-1 : iMax  , jMin : jMax+1, PRES] + pEq_pTerm_mm1 * q[iMin-1 : iMax  , jMin-1 : jMax, PRES]) \
        ) * (dy / (pEq_pTerm_mp1 + pEq_pTerm_m + pEq_pTerm_mm1)) \
      + dy/2. * \
        ( pEq_pTerm_mp1 * (pEq_pTerm_sdp1 * q[iMin+1 : iMax+2, jMin+1 : jMax+2, PRES] + pEq_pTerm_sd * q[iMin+1 : iMax+2, jMin : jMax+1, PRES] + pEq_pTerm_sdm1 * q[iMin+1 : iMax+2, jMin-1 : jMax, PRES]) \
        + pEq_pTerm_m   * (pEq_pTerm_sdp1 * q[iMin   : iMax+1, jMin+1 : jMax+2, PRES] + pEq_pTerm_sd * q[iMin   : iMax+1, jMin : jMax+1, PRES] + pEq_pTerm_sdm1 * q[iMin   : iMax+1, jMin-1 : jMax, PRES]) \
        + pEq_pTerm_mm1 * (pEq_pTerm_sdp1 * q[iMin-1 : iMax  , jMin+1 : jMax+2, PRES] + pEq_pTerm_sd * q[iMin-1 : iMax  , jMin : jMax+1, PRES] + pEq_pTerm_sdm1 * q[iMin-1 : iMax  , jMin-1 : jMax, PRES]) \
        ) * (dx / (pEq_pTerm_mp1 + pEq_pTerm_m + pEq_pTerm_mm1)) \
      )


    return divF





@njit
def get_looped_numba_modif_SUPG_developped_divFlux(q, iMin, iMax, jMin, jMax, dx, dy, coeffs):
    # allocation de variables pour chaque coeff
    # u equation - u term
    uEq_uTerm_xp1, uEq_uTerm_x, uEq_uTerm_xm1 = coeffs[0, 0, 0, :]
    uEq_uTerm_yp1, uEq_uTerm_y, uEq_uTerm_ym1 = coeffs[0, 0, 1, :]

    # u equation - v term
    uEq_vTerm_xp1, uEq_vTerm_x, uEq_vTerm_xm1 = coeffs[0, 1, 0, :]
    uEq_vTerm_yp1, uEq_vTerm_y, uEq_vTerm_ym1 = coeffs[0, 1, 1, :]

    # u equation - p term
    uEq_pTerm_xp1, uEq_pTerm_x, uEq_pTerm_xm1 = coeffs[0, 2, 0, :]
    uEq_pTerm_yp1, uEq_pTerm_y, uEq_pTerm_ym1 = coeffs[0, 2, 1, :]

    # v equation - u term
    vEq_uTerm_xp1, vEq_uTerm_x, vEq_uTerm_xm1 = coeffs[1, 0, 0, :]
    vEq_uTerm_yp1, vEq_uTerm_y, vEq_uTerm_ym1 = coeffs[1, 0, 1, :]

    # v equation - v term
    vEq_vTerm_xp1, vEq_vTerm_x, vEq_vTerm_xm1 = coeffs[1, 1, 0, :]
    vEq_vTerm_yp1, vEq_vTerm_y, vEq_vTerm_ym1 = coeffs[1, 1, 1, :]

    # v equation - p term
    vEq_pTerm_xp1, vEq_pTerm_x, vEq_pTerm_xm1 = coeffs[1, 2, 0, :]
    vEq_pTerm_yp1, vEq_pTerm_y, vEq_pTerm_ym1 = coeffs[1, 2, 1, :]

    # p equation - u term
    pEq_uTerm_xp1, pEq_uTerm_x, pEq_uTerm_xm1 = coeffs[2, 0, 0, :]
    pEq_uTerm_yp1, pEq_uTerm_y, pEq_uTerm_ym1 = coeffs[2, 0, 1, :]

    # p equation - v term
    pEq_vTerm_xp1, pEq_vTerm_x, pEq_vTerm_xm1 = coeffs[2, 1, 0, :]
    pEq_vTerm_yp1, pEq_vTerm_y, pEq_vTerm_ym1 = coeffs[2, 1, 1, :]

    # p equation - p term (cas spécial avec mass et second_der)
    pEq_pTerm_mp1, pEq_pTerm_m, pEq_pTerm_mm1 = coeffs[2, 2, 2, :]
    pEq_pTerm_sdp1, pEq_pTerm_sd, pEq_pTerm_sdm1 = coeffs[2, 2, 3, :]


    
    divF = np.zeros((np.shape(q)))
    
    for i in range(iMin, iMax+1):
        for j in range(jMin, jMax+1):

            # u equation
            divF[i, j, XVEL] =  \
              1. / (dx * dy) * \
              ( \
              dx/2. * \
                ( uEq_uTerm_xp1 * (uEq_uTerm_yp1 * q[i+1, j+1, XVEL] + uEq_uTerm_y * q[i+1, j, XVEL] + uEq_uTerm_ym1 * q[i+1, j-1, XVEL]) \
                + uEq_uTerm_x   * (uEq_uTerm_yp1 * q[i  , j+1, XVEL] + uEq_uTerm_y * q[i  , j, XVEL] + uEq_uTerm_ym1 * q[i  , j-1, XVEL]) \
                + uEq_uTerm_xm1 * (uEq_uTerm_yp1 * q[i-1, j+1, XVEL] + uEq_uTerm_y * q[i-1, j, XVEL] + uEq_uTerm_ym1 * q[i-1, j-1, XVEL]) \
                ) * (1. / dx) * (dy / (uEq_uTerm_yp1 + uEq_uTerm_y + uEq_uTerm_ym1)) \
              \
              + dx/2. * \
                ( uEq_vTerm_xp1 * (uEq_vTerm_yp1 * q[i+1, j+1, YVEL] + uEq_vTerm_y * q[i+1, j, YVEL] + uEq_vTerm_ym1 * q[i+1, j-1, YVEL]) \
                + uEq_vTerm_x   * (uEq_vTerm_yp1 * q[i  , j+1, YVEL] + uEq_vTerm_y * q[i  , j, YVEL] + uEq_vTerm_ym1 * q[i  , j-1, YVEL]) \
                + uEq_vTerm_xm1 * (uEq_vTerm_yp1 * q[i-1, j+1, YVEL] + uEq_vTerm_y * q[i-1, j, YVEL] + uEq_vTerm_ym1 * q[i-1, j-1, YVEL]) \
                ) \
              # + dx/2. * \
              #   ( uEq_vTerm_xp1 * (uEq_vTerm_yp1 * q[i+1, j+1, YVEL] + uEq_vTerm_ym1 * q[i+1, j-1, YVEL]) \
              #   + uEq_vTerm_xm1 * (uEq_vTerm_yp1 * q[i-1, j+1, YVEL] + uEq_vTerm_ym1 * q[i-1, j-1, YVEL]) \
              #   ) \
              + 1.* \
                ( uEq_pTerm_xp1 * (uEq_pTerm_yp1 * q[i+1, j+1, PRES] + uEq_pTerm_y * q[i+1, j, PRES] + uEq_pTerm_ym1 * q[i+1, j-1, PRES]) \
                + uEq_pTerm_x   * (uEq_pTerm_yp1 * q[i  , j+1, PRES] + uEq_pTerm_y * q[i  , j, PRES] + uEq_pTerm_ym1 * q[i  , j-1, PRES]) \
                + uEq_pTerm_xm1 * (uEq_pTerm_yp1 * q[i-1, j+1, PRES] + uEq_pTerm_y * q[i-1, j, PRES] + uEq_pTerm_ym1 * q[i-1, j-1, PRES]) \
                ) * (dy / (uEq_pTerm_yp1 + uEq_pTerm_y + uEq_pTerm_ym1)) \
              # + 1.* \
              #   ( uEq_pTerm_xp1 * (uEq_pTerm_yp1 * q[i+1, j+1, PRES] + uEq_pTerm_y * q[i+1, j, PRES] + uEq_pTerm_ym1 * q[i+1, j-1, PRES]) \
              #   + uEq_pTerm_xm1 * (uEq_pTerm_yp1 * q[i-1, j+1, PRES] + uEq_pTerm_y * q[i-1, j, PRES] + uEq_pTerm_ym1 * q[i-1, j-1, PRES]) \
              #   ) * (1. / dx) * (dy / (uEq_pTerm_yp1 + uEq_pTerm_y + uEq_pTerm_ym1)) \
              )
              


            ### v equation 

            divF[i, j, YVEL] =  \
              1. / (dx * dy) * \
              ( \
              dy/2. * \
                ( vEq_uTerm_xp1 * (vEq_uTerm_yp1 * q[i+1, j+1, XVEL] + vEq_uTerm_y * q[i+1, j, XVEL] + vEq_uTerm_ym1 * q[i+1, j-1, XVEL]) \
                + vEq_uTerm_x   * (vEq_uTerm_yp1 * q[i  , j+1, XVEL] + vEq_uTerm_y * q[i  , j, XVEL] + vEq_uTerm_ym1 * q[i  , j-1, XVEL]) \
                + vEq_uTerm_xm1 * (vEq_uTerm_yp1 * q[i-1, j+1, XVEL] + vEq_uTerm_y * q[i-1, j, XVEL] + vEq_uTerm_ym1 * q[i-1, j-1, XVEL]) \
                )
              # + dy/2. * \
              #   ( vEq_uTerm_xp1 * (vEq_uTerm_yp1 * q[i+1, j+1, XVEL] + vEq_uTerm_ym1 * q[i+1, j-1, XVEL]) \
              #   + vEq_uTerm_xm1 * (vEq_uTerm_yp1 * q[i-1, j+1, XVEL] + vEq_uTerm_ym1 * q[i-1, j-1, XVEL]) \
              #   ) \
              \
              + dy/2. * \
                ( vEq_vTerm_xp1 * (vEq_vTerm_yp1 * q[i+1, j+1, YVEL] + vEq_vTerm_y * q[i+1, j, YVEL] + vEq_vTerm_ym1 * q[i+1, j-1, YVEL]) \
                + vEq_vTerm_x   * (vEq_vTerm_yp1 * q[i  , j+1, YVEL] + vEq_vTerm_y * q[i  , j, YVEL] + vEq_vTerm_ym1 * q[i  , j-1, YVEL]) \
                + vEq_vTerm_xm1 * (vEq_vTerm_yp1 * q[i-1, j+1, YVEL] + vEq_vTerm_y * q[i-1, j, YVEL] + vEq_vTerm_ym1 * q[i-1, j-1, YVEL]) \
                ) * (1. / dy) * (dx / (vEq_vTerm_xp1 + vEq_vTerm_x + vEq_vTerm_xm1)) \
              + 1.* \
                ( vEq_pTerm_xp1 * (vEq_pTerm_yp1 * q[i+1, j+1, PRES] + vEq_pTerm_y * q[i+1, j, PRES] + vEq_pTerm_ym1 * q[i+1, j-1, PRES]) \
                + vEq_pTerm_x   * (vEq_pTerm_yp1 * q[i  , j+1, PRES] + vEq_pTerm_y * q[i  , j, PRES] + vEq_pTerm_ym1 * q[i  , j-1, PRES]) \
                + vEq_pTerm_xm1 * (vEq_pTerm_yp1 * q[i-1, j+1, PRES] + vEq_pTerm_y * q[i-1, j, PRES] + vEq_pTerm_ym1 * q[i-1, j-1, PRES]) \
                ) * (dx / (vEq_pTerm_xp1 + vEq_pTerm_x + vEq_pTerm_xm1)) \
              # + 1.* \
              #   ( vEq_pTerm_xp1 * (vEq_pTerm_yp1 * q[i+1, j+1, PRES] + vEq_pTerm_ym1 * q[i+1, j-1, PRES]) \
              #   + vEq_pTerm_x   * (vEq_pTerm_yp1 * q[i  , j+1, PRES] + vEq_pTerm_ym1 * q[i  , j-1, PRES]) \
              #   + vEq_pTerm_xm1 * (vEq_pTerm_yp1 * q[i-1, j+1, PRES] + vEq_pTerm_ym1 * q[i-1, j-1, PRES]) \
              #   ) * (1. / dx) * (dy / (vEq_pTerm_yp1 + vEq_pTerm_y + vEq_pTerm_ym1)) \
              ) 


            ### p equation 

            divF[i, j, PRES] =  \
              1. / (dx * dy) * \
              ( \
              1.* \
                ( pEq_uTerm_xp1 * (pEq_uTerm_yp1 * q[i+1, j+1, XVEL] + pEq_uTerm_y * q[i+1, j, XVEL] + pEq_uTerm_ym1 * q[i+1, j-1, XVEL]) \
                + pEq_uTerm_x   * (pEq_uTerm_yp1 * q[i  , j+1, XVEL] + pEq_uTerm_y * q[i  , j, XVEL] + pEq_uTerm_ym1 * q[i  , j-1, XVEL]) \
                + pEq_uTerm_xm1 * (pEq_uTerm_yp1 * q[i-1, j+1, XVEL] + pEq_uTerm_y * q[i-1, j, XVEL] + pEq_uTerm_ym1 * q[i-1, j-1, XVEL]) \
                ) * (dy / (pEq_uTerm_yp1 + pEq_uTerm_y + pEq_uTerm_ym1)) \
              # + 1.* \
              #   ( pEq_uTerm_xp1 * (pEq_uTerm_yp1 * q[i+1, j+1, XVEL] + pEq_uTerm_y * q[i+1, j, XVEL] + pEq_uTerm_ym1 * q[i+1, j-1, XVEL]) \
              #   + pEq_uTerm_xm1 * (pEq_uTerm_yp1 * q[i-1, j+1, XVEL] + pEq_uTerm_y * q[i-1, j, XVEL] + pEq_uTerm_ym1 * q[i-1, j-1, XVEL]) \
              #   ) * (1. / dx) * (dy / (pEq_uTerm_yp1 + pEq_uTerm_y + pEq_uTerm_ym1)) \
              + 1.* \
                ( pEq_vTerm_xp1 * (pEq_vTerm_yp1 * q[i+1, j+1, YVEL] + pEq_vTerm_y * q[i+1, j, YVEL] + pEq_vTerm_ym1 * q[i+1, j-1, YVEL]) \
                + pEq_vTerm_x   * (pEq_vTerm_yp1 * q[i  , j+1, YVEL] + pEq_vTerm_y * q[i  , j, YVEL] + pEq_vTerm_ym1 * q[i  , j-1, YVEL]) \
                + pEq_vTerm_xm1 * (pEq_vTerm_yp1 * q[i-1, j+1, YVEL] + pEq_vTerm_y * q[i-1, j, YVEL] + pEq_vTerm_ym1 * q[i-1, j-1, YVEL]) \
                ) * (dx / (pEq_vTerm_xp1 + pEq_vTerm_x + pEq_vTerm_xm1)) \
              # + 1.* \
              #   ( pEq_vTerm_xp1 * (pEq_vTerm_yp1 * q[i+1, j+1, YVEL] + pEq_vTerm_ym1 * q[i+1, j-1, YVEL]) \
              #   + pEq_vTerm_x   * (pEq_vTerm_yp1 * q[i  , j+1, YVEL] + pEq_vTerm_ym1 * q[i  , j-1, YVEL]) \
              #   + pEq_vTerm_xm1 * (pEq_vTerm_yp1 * q[i-1, j+1, YVEL] + pEq_vTerm_ym1 * q[i-1, j-1, YVEL]) \
              #   ) * (1. / dx) * (dy / (pEq_vTerm_yp1 + pEq_vTerm_y + pEq_vTerm_ym1)) \
              \
              + dx/2. * \
                ( pEq_pTerm_sdp1 * (pEq_pTerm_mp1 * q[i+1, j+1, PRES] + pEq_pTerm_m * q[i+1, j, PRES] + pEq_pTerm_mm1 * q[i+1, j-1, PRES]) \
                + pEq_pTerm_sd   * (pEq_pTerm_mp1 * q[i  , j+1, PRES] + pEq_pTerm_m * q[i  , j, PRES] + pEq_pTerm_mm1 * q[i  , j-1, PRES]) \
                + pEq_pTerm_sdm1 * (pEq_pTerm_mp1 * q[i-1, j+1, PRES] + pEq_pTerm_m * q[i-1, j, PRES] + pEq_pTerm_mm1 * q[i-1, j-1, PRES]) \
                ) * (1. / dx ) * (dy / (pEq_pTerm_mp1 + pEq_pTerm_m + pEq_pTerm_mm1)) \
              + dy/2. * \
                ( pEq_pTerm_mp1 * (pEq_pTerm_sdp1 * q[i+1, j+1, PRES] + pEq_pTerm_sd * q[i+1, j, PRES] + pEq_pTerm_sdm1 * q[i+1, j-1, PRES]) \
                + pEq_pTerm_m   * (pEq_pTerm_sdp1 * q[i  , j+1, PRES] + pEq_pTerm_sd * q[i  , j, PRES] + pEq_pTerm_sdm1 * q[i  , j-1, PRES]) \
                + pEq_pTerm_mm1 * (pEq_pTerm_sdp1 * q[i-1, j+1, PRES] + pEq_pTerm_sd * q[i-1, j, PRES] + pEq_pTerm_sdm1 * q[i-1, j-1, PRES]) \
                ) * (dx / (pEq_pTerm_mp1 + pEq_pTerm_m + pEq_pTerm_mm1)) * (1. / dy ) \
              )


    return divF


####################################################################################################################

#                                                 CLASSICAL UPWIND                                                 #

####################################################################################################################



def getApproxDivFlux(q, grid, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, operators, operatorsCoeff):
    if schemeChoice == 1 :
        return get_UPWIND_divFlux(q, grid)
    if schemeChoice == 2 :
        return get_SUPG_developped_divFlux(q, grid)
    if schemeChoice == 3 :
        return get_modif_SUPG_developped_divFlux(q, grid, operators)
    if schemeChoice == 4 :
        # return get_optim_modif_SUPG_developped_divFlux(q, grid, operators)
        return get_looped_numba_modif_SUPG_developped_divFlux(q, iMin, iMax, jMin, jMax, dx, dy, operatorsCoeff)
    

    


def extract_operators(operators):
    EQUATIONS = ["u_equation", "v_equation", "p_equation"]
    TERMS      = ["u_term", "v_term", "p_term"]
    KEYS       = ["x", "y", "mass", "second_der"]
    
    # shape : (3 equations, 3 terms, 4 keys, 3 valeurs)
    coeffs = np.zeros((3, 3, 4, 3), dtype=np.float64)
    for i, eq in enumerate(EQUATIONS):
        for j, term in enumerate(TERMS):
            for k, key in enumerate(KEYS):
                if key in operators[eq][term]:
                    coeffs[i, j, k, :] = operators[eq][term][key]
    return coeffs