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
      (    pEq_pTerm["x_second_der"]["x"][0] * (  \
                                           pEq_pTerm["x_second_der"]["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                         + pEq_pTerm["x_second_der"]["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                         + pEq_pTerm["x_second_der"]["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, PRES] \
                                         ) \
        +  pEq_pTerm["x_second_der"]["x"][1] * (  \
                                           pEq_pTerm["x_second_der"]["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                                         + pEq_pTerm["x_second_der"]["y"][1] * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                                         + pEq_pTerm["x_second_der"]["y"][2] * q[iMin:iMax+1, jMin-1:jMax, PRES] \
                                         ) \
        +  pEq_pTerm["x_second_der"]["x"][2] * (  \
                                           pEq_pTerm["x_second_der"]["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                         + pEq_pTerm["x_second_der"]["y"][1] * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                         + pEq_pTerm["x_second_der"]["y"][2] * q[iMin-1:iMax, jMin-1:jMax, PRES] \
                                         ) \
      ) * (1. / dx) * (dy / sum(pEq_pTerm["x_second_der"]["y"])) \
    + dy/2. * \
      (    pEq_pTerm["y_second_der"]["x"][0] * (  \
                                     pEq_pTerm["y_second_der"]["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                   + pEq_pTerm["y_second_der"]["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                   + pEq_pTerm["y_second_der"]["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, PRES] \
                                    ) \
        +  pEq_pTerm["y_second_der"]["x"][1] * (  \
                                     pEq_pTerm["y_second_der"]["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                                   + pEq_pTerm["y_second_der"]["y"][1] * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                                   + pEq_pTerm["y_second_der"]["y"][2] * q[iMin:iMax+1, jMin-1:jMax, PRES] \
                                   ) \
        +  pEq_pTerm["y_second_der"]["x"][2] * (  \
                                     pEq_pTerm["y_second_der"]["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                   + pEq_pTerm["y_second_der"]["y"][1] * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                   + pEq_pTerm["y_second_der"]["y"][2] * q[iMin-1:iMax, jMin-1:jMax, PRES] \
                                   ) \
      ) * (dx / sum(pEq_pTerm["y_second_der"]["x"])) * (1. / dy) \
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
    uEq_uTerm_xp1, uEq_uTerm_x, uEq_uTerm_xm1 = coeffs[0, :]
    uEq_uTerm_yp1, uEq_uTerm_y, uEq_uTerm_ym1 = coeffs[1, :]

    # u equation - v term
    uEq_vTerm_xp1, uEq_vTerm_x, uEq_vTerm_xm1 = coeffs[2, :]
    uEq_vTerm_yp1, uEq_vTerm_y, uEq_vTerm_ym1 = coeffs[3, :]

    # u equation - p term
    uEq_pTerm_xp1, uEq_pTerm_x, uEq_pTerm_xm1 = coeffs[4, :]
    uEq_pTerm_yp1, uEq_pTerm_y, uEq_pTerm_ym1 = coeffs[5, :]

    # v equation - u term
    vEq_uTerm_xp1, vEq_uTerm_x, vEq_uTerm_xm1 = coeffs[6, :]
    vEq_uTerm_yp1, vEq_uTerm_y, vEq_uTerm_ym1 = coeffs[7, :]

    # v equation - v term
    vEq_vTerm_xp1, vEq_vTerm_x, vEq_vTerm_xm1 = coeffs[8, :]
    vEq_vTerm_yp1, vEq_vTerm_y, vEq_vTerm_ym1 = coeffs[9, :]

    # v equation - p term
    vEq_pTerm_xp1, vEq_pTerm_x, vEq_pTerm_xm1 = coeffs[10, :]
    vEq_pTerm_yp1, vEq_pTerm_y, vEq_pTerm_ym1 = coeffs[11, :]

    # p equation - u term
    pEq_uTerm_xp1, pEq_uTerm_x, pEq_uTerm_xm1 = coeffs[12, :]
    pEq_uTerm_yp1, pEq_uTerm_y, pEq_uTerm_ym1 = coeffs[13, :]

    # p equation - v term
    pEq_vTerm_xp1, pEq_vTerm_x, pEq_vTerm_xm1 = coeffs[14, :]
    pEq_vTerm_yp1, pEq_vTerm_y, pEq_vTerm_ym1 = coeffs[15, :]

    # p equation - p term (cas spécial avec mass et second_der)
    pEq_pTerm1_xp1, pEq_pTerm1_x, pEq_pTerm1_xm1 = coeffs[16, :]
    pEq_pTerm1_yp1, pEq_pTerm1_y, pEq_pTerm1_ym1 = coeffs[17, :]

    pEq_pTerm2_xp1, pEq_pTerm2_x, pEq_pTerm2_xm1 = coeffs[18, :]
    pEq_pTerm2_yp1, pEq_pTerm2_y, pEq_pTerm2_ym1 = coeffs[19, :]


    
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
                ( pEq_pTerm1_xp1 * (pEq_pTerm1_yp1 * q[i+1, j+1, PRES] + pEq_pTerm1_y * q[i+1, j, PRES] + pEq_pTerm1_ym1 * q[i+1, j-1, PRES]) \
                + pEq_pTerm1_x   * (pEq_pTerm1_yp1 * q[i  , j+1, PRES] + pEq_pTerm1_y * q[i  , j, PRES] + pEq_pTerm1_ym1 * q[i  , j-1, PRES]) \
                + pEq_pTerm1_xm1 * (pEq_pTerm1_yp1 * q[i-1, j+1, PRES] + pEq_pTerm1_y * q[i-1, j, PRES] + pEq_pTerm1_ym1 * q[i-1, j-1, PRES]) \
                ) * (1. / dx ) * (dy / (pEq_pTerm1_yp1 + pEq_pTerm1_y + pEq_pTerm1_ym1)) \
              + dy/2. * \
                ( pEq_pTerm2_xp1 * (pEq_pTerm2_yp1 * q[i+1, j+1, PRES] + pEq_pTerm2_y * q[i+1, j, PRES] + pEq_pTerm2_ym1 * q[i+1, j-1, PRES]) \
                + pEq_pTerm2_x   * (pEq_pTerm2_yp1 * q[i  , j+1, PRES] + pEq_pTerm2_y * q[i  , j, PRES] + pEq_pTerm2_ym1 * q[i  , j-1, PRES]) \
                + pEq_pTerm2_xm1 * (pEq_pTerm2_yp1 * q[i-1, j+1, PRES] + pEq_pTerm2_y * q[i-1, j, PRES] + pEq_pTerm2_ym1 * q[i-1, j-1, PRES]) \
                ) * (dx / (pEq_pTerm2_xp1 + pEq_pTerm2_x + pEq_pTerm2_xm1)) * (1. / dy ) \
              )


    return divF





@njit
def get_optim_numba_modif_SUPG_developped_divFlux(q, iMin, iMax, jMin, jMax, coeffs):
    # allocation de variables pour chaque coeff
    # u equation - u term
    uEq_uTerm_xp1, uEq_uTerm_x, uEq_uTerm_xm1 = coeffs[0]
    uEq_uTerm_yp1, uEq_uTerm_y, uEq_uTerm_ym1 = coeffs[1]

    # u equation - v term
    uEq_vTerm_xp1, uEq_vTerm_x, uEq_vTerm_xm1 = coeffs[2]
    uEq_vTerm_yp1, uEq_vTerm_y, uEq_vTerm_ym1 = coeffs[3]

    # u equation - p term
    uEq_pTerm_xp1, uEq_pTerm_x, uEq_pTerm_xm1 = coeffs[4]
    uEq_pTerm_yp1, uEq_pTerm_y, uEq_pTerm_ym1 = coeffs[5]

    # v equation - u term
    vEq_uTerm_xp1, vEq_uTerm_x, vEq_uTerm_xm1 = coeffs[6]
    vEq_uTerm_yp1, vEq_uTerm_y, vEq_uTerm_ym1 = coeffs[7]

    # v equation - v term
    vEq_vTerm_xp1, vEq_vTerm_x, vEq_vTerm_xm1 = coeffs[8]
    vEq_vTerm_yp1, vEq_vTerm_y, vEq_vTerm_ym1 = coeffs[9]

    # v equation - p term
    vEq_pTerm_xp1, vEq_pTerm_x, vEq_pTerm_xm1 = coeffs[10]
    vEq_pTerm_yp1, vEq_pTerm_y, vEq_pTerm_ym1 = coeffs[11]

    # p equation - u term
    pEq_uTerm_xp1, pEq_uTerm_x, pEq_uTerm_xm1 = coeffs[12]
    pEq_uTerm_yp1, pEq_uTerm_y, pEq_uTerm_ym1 = coeffs[13]

    # p equation - v term
    pEq_vTerm_xp1, pEq_vTerm_x, pEq_vTerm_xm1 = coeffs[14]
    pEq_vTerm_yp1, pEq_vTerm_y, pEq_vTerm_ym1 = coeffs[15]

    # p equation - p term (cas spécial avec mass et second_der)
    pEq_pTerm1_xp1, pEq_pTerm1_x, pEq_pTerm1_xm1 = coeffs[16]
    pEq_pTerm1_yp1, pEq_pTerm1_y, pEq_pTerm1_ym1 = coeffs[17]

    pEq_pTerm2_xp1, pEq_pTerm2_x, pEq_pTerm2_xm1 = coeffs[18]
    pEq_pTerm2_yp1, pEq_pTerm2_y, pEq_pTerm2_ym1 = coeffs[19]


    
    divF = np.zeros((np.shape(q)))

    shape = np.shape(q)[:-1]
    uEq_uTerm_yOp, uEq_vTerm_yOp, uEq_pTerm_yOp, vEq_uTerm_yOp, vEq_vTerm_yOp, vEq_pTerm_yOp, \
      pEq_uTerm_yOp, pEq_vTerm_yOp, pEq_pTerm1_yOp, pEq_pTerm2_yOp = np.zeros(((10, ) + shape))


    # COMPUTATIONS OF THE Y OPERATORS ON THE STENCIL iMin-1:iMax+1 x jMin:jMax
    for i in range(iMin-1, iMax+3):
        for j in range(jMin, jMax+2):
            # U equation
            uEq_uTerm_yOp[i, j] = uEq_uTerm_yp1 * q[i, j+1, XVEL] + uEq_uTerm_y * q[i, j, XVEL] + uEq_uTerm_ym1 * q[i, j-1, XVEL]
            uEq_vTerm_yOp[i, j] = uEq_vTerm_yp1 * q[i, j+1, YVEL] + uEq_vTerm_y * q[i, j, YVEL] + uEq_vTerm_ym1 * q[i, j-1, YVEL]
            uEq_pTerm_yOp[i, j] = uEq_pTerm_yp1 * q[i, j+1, PRES] + uEq_pTerm_y * q[i, j, PRES] + uEq_pTerm_ym1 * q[i, j-1, PRES]


            # V equation
            vEq_uTerm_yOp[i, j] = vEq_uTerm_yp1 * q[i, j+1, XVEL] + vEq_uTerm_y * q[i, j, XVEL] + vEq_uTerm_ym1 * q[i, j-1, XVEL]
            vEq_vTerm_yOp[i, j] = vEq_vTerm_yp1 * q[i, j+1, YVEL] + vEq_vTerm_y * q[i, j, YVEL] + vEq_vTerm_ym1 * q[i, j-1, YVEL]
            vEq_pTerm_yOp[i, j] = vEq_pTerm_yp1 * q[i, j+1, PRES] + vEq_pTerm_y * q[i, j, PRES] + vEq_pTerm_ym1 * q[i, j-1, PRES]


            # P equation
            pEq_uTerm_yOp[i, j] = pEq_uTerm_yp1 * q[i, j+1, XVEL] + pEq_uTerm_y * q[i, j, XVEL] + pEq_uTerm_ym1 * q[i, j-1, XVEL]
            pEq_vTerm_yOp[i, j] = pEq_vTerm_yp1 * q[i, j+1, YVEL] + pEq_vTerm_y * q[i, j, YVEL] + pEq_vTerm_ym1 * q[i, j-1, YVEL]
            pEq_pTerm1_yOp[i, j] = pEq_pTerm1_yp1 * q[i, j+1, PRES] + pEq_pTerm1_y * q[i, j, PRES] + pEq_pTerm1_ym1 * q[i, j-1, PRES]
            pEq_pTerm2_yOp[i, j] = pEq_pTerm2_yp1 * q[i, j+1, PRES] + pEq_pTerm2_y * q[i, j, PRES] + pEq_pTerm2_ym1 * q[i, j-1, PRES]



    # COMPLETING INTO FULL TENSOR OPERATORS + ASSEMBLING
    for i in range(iMin, iMax+2):
        for j in range(jMin, jMax+2):
            # U equation
            uEq_uTerm_ij = uEq_uTerm_xp1 * uEq_uTerm_yOp[i+1, j] + uEq_uTerm_x * uEq_uTerm_yOp[i, j] + uEq_uTerm_xm1 * uEq_uTerm_yOp[i-1, j]
            uEq_vTerm_ij = uEq_vTerm_xp1 * uEq_vTerm_yOp[i+1, j] + uEq_vTerm_x * uEq_vTerm_yOp[i, j] + uEq_vTerm_xm1 * uEq_vTerm_yOp[i-1, j]
            uEq_pTerm_ij = uEq_pTerm_xp1 * uEq_pTerm_yOp[i+1, j] + uEq_pTerm_x * uEq_pTerm_yOp[i, j] + uEq_pTerm_xm1 * uEq_pTerm_yOp[i-1, j]
            divF[i, j, XVEL] = 0.5 * uEq_uTerm_ij + 0.5 * uEq_vTerm_ij + uEq_pTerm_ij


            # V equation
            vEq_uTerm_ij = vEq_uTerm_xp1 * vEq_uTerm_yOp[i+1, j] + vEq_uTerm_x * vEq_uTerm_yOp[i, j] + vEq_uTerm_xm1 * vEq_uTerm_yOp[i-1, j]
            vEq_vTerm_ij = vEq_vTerm_xp1 * vEq_vTerm_yOp[i+1, j] + vEq_vTerm_x * vEq_vTerm_yOp[i, j] + vEq_vTerm_xm1 * vEq_vTerm_yOp[i-1, j]
            vEq_pTerm_ij = vEq_pTerm_xp1 * vEq_pTerm_yOp[i+1, j] + vEq_pTerm_x * vEq_pTerm_yOp[i, j] + vEq_pTerm_xm1 * vEq_pTerm_yOp[i-1, j]
            divF[i, j, YVEL] = 0.5 * vEq_uTerm_ij + 0.5 * vEq_vTerm_ij + vEq_pTerm_ij

            # P equation
            pEq_uTerm_ij = pEq_uTerm_xp1 * pEq_uTerm_yOp[i+1, j] + pEq_uTerm_x * pEq_uTerm_yOp[i, j] + pEq_uTerm_xm1 * pEq_uTerm_yOp[i-1, j]
            pEq_vTerm_ij = pEq_vTerm_xp1 * pEq_vTerm_yOp[i+1, j] + pEq_vTerm_x * pEq_vTerm_yOp[i, j] + pEq_vTerm_xm1 * pEq_vTerm_yOp[i-1, j]
            pEq_pTerm1_ij = pEq_pTerm1_xp1 * pEq_pTerm1_yOp[i+1, j] + pEq_pTerm1_x * pEq_pTerm1_yOp[i, j] + pEq_pTerm1_xm1 * pEq_pTerm1_yOp[i-1, j]
            pEq_pTerm2_ij = pEq_pTerm2_xp1 * pEq_pTerm2_yOp[i+1, j] + pEq_pTerm2_x * pEq_pTerm2_yOp[i, j] + pEq_pTerm2_xm1 * pEq_pTerm2_yOp[i-1, j]
            divF[i, j, PRES] = pEq_uTerm_ij + pEq_vTerm_ij + 0.5 * pEq_pTerm1_ij + 0.5 * pEq_pTerm2_ij

    return divF






def getApproxDivFlux(q, grid, iMin, iMax, jMin, jMax, dx, dy, schemeChoice, operators, operatorsCoeff):
    if schemeChoice == 1 :
        return get_UPWIND_divFlux(q, grid)
    if schemeChoice == 2 :
        return get_SUPG_developped_divFlux(q, grid)
    if schemeChoice == 3 :
        return get_numba_modif_SUPG_developped_divFlux(q, iMin, iMax, jMin, jMax, dx, dy, operatorsCoeff)
    if schemeChoice == 4 :
        return get_optim_numba_modif_SUPG_developped_divFlux(q, iMin, iMax, jMin, jMax, operatorsCoeff)
    if schemeChoice == -1 :
        # return get_SUPG_developped_divFlux(q, grid)
        # return get_modif_SUPG_developped_divFlux(q, grid, operators)
        # return get_optim_modif_SUPG_developped_divFlux(q, grid, operators)
        # return get_numba_modif_SUPG_developped_divFlux(q, iMin, iMax, jMin, jMax, dx, dy, operatorsCoeff)
        return get_optim_numba_modif_SUPG_developped_divFlux(q, iMin, iMax, jMin, jMax, operatorsCoeff)
    

    


# Les coefficients sont structurés de la façon suivante :
# [[uEq_uTerm_x], [uEq_uTerm_y], [uEq_vTerm_x], ...]
def extract_operators(operators):
    all_equations_except_p = ["u_equation", "v_equation"]
    all_terms      = ["u_term", "v_term", "p_term"]
    all_terms_except_p      = ["u_term", "v_term"]

    # shape : 10 opérateurs tensoriels en tout : 3 pour l'eq en u, 3 pour v, et 1 + 1 + 2 pour p (Dxx My et Mx Dyy !!!)
    # Structure des coeffs
    # 3+3, 3+3, 3+3
    # 3+3, 3+3, 3+3
    # 3+3, 3+3, (3+3, 3+3)
    # ce qui fait le double en coefficient : 1 op tensoriel = 1 op en x + 1 op en y
    coeffs = np.zeros((20, 3), dtype=np.float64)

    # tous les termes ont la même structure simple sauf pEq_pTerm
    # on commence par les equations en u et v
    for i, eq in enumerate(all_equations_except_p):
        for j, term in enumerate(all_terms):
            coeffs[6 * i + 2 * j, :] = operators[eq][term]["x"]
            coeffs[6 * i + 2 * j + 1, :] = operators[eq][term]["y"]

    # on regarde ensuite les termes en u et v de l'equation en p
    for j, term in enumerate(all_terms_except_p):
      coeffs[12 + 2 * j, :] = operators["p_equation"][term]["x"]
      coeffs[12 + 2 * j + 1, :] = operators["p_equation"][term]["y"]

    # on termine par les terms en p de l'équation en p :
    coeffs[16, :] = operators["p_equation"]["p_term"]["x_second_der"]["x"]
    coeffs[17, :] = operators["p_equation"]["p_term"]["x_second_der"]["y"]
    coeffs[18, :] = operators["p_equation"]["p_term"]["y_second_der"]["x"]
    coeffs[19, :] = operators["p_equation"]["p_term"]["y_second_der"]["y"]

    return coeffs



# Les coefficients sont structurés de la façon suivante :
# [[uEq_uTerm_x], [uEq_uTerm_y], [uEq_vTerm_x], ...]
def extract_operators_improved(dx, dy):
    coeffs = np.zeros((20, 3), dtype=np.float64)

    # U EQUATION
    ### uEq_uTerm : derivée seconde en x et masse en y
    coeffs[0, :] = np.array([-1., 2., -1.]) / dx
    coeffs[1, :] = np.array([1., 4., 1.]) / 6.

    ### uEq_vTerm : derivées simples mais diffusion par dx
    coeffs[2, :] = np.array([-1., 0, 1.]) / 2.
    coeffs[3, :] = np.array([1., 0, -1.]) / 2. / dy

    ### uEq_pTerm : derivée simple en x et masse en y
    coeffs[4, :] = np.array([1., 0, -1.]) / 2. / dx
    coeffs[5, :] = np.array([1., 4., 1.]) / 6.



    # V EQUATION
    ### vEq_uTerm : derivées simples
    coeffs[6, :] = np.array([1., 0, -1.]) / 2. / dx
    coeffs[7, :] = np.array([-1., 0, 1.]) / 2.

    ### vEq_vTerm : masse en x et derivée seconde en y
    coeffs[8, :] = np.array([1., 4., 1.]) /  6.
    coeffs[9, :] = np.array([-1., 2., -1.]) / dy

    ### vEq_pTerm : masse en x et derivée simple en y
    coeffs[10, :] = np.array([1., 4., 1.]) / 6.
    coeffs[11, :] = np.array([1., 0, -1.]) / 2. / dy



    # P EQUATION
    ### pEq_uTerm : derivée simple en x et masse en y
    coeffs[12, :] = np.array([1., 0, -1.]) / 2. / dx
    coeffs[13, :] = np.array([1., 4., 1.]) / 6.

    ### pEq_vTerm : masse en x et derivée simple en y
    coeffs[14, :] = np.array([1., 4., 1.]) / 6.
    coeffs[15, :] = np.array([1., 0, -1.]) / 2. / dy

    ### pEq_pTerms
    coeffs[16, :] = np.array([-1., 2., -1.]) / dx
    coeffs[17, :] = np.array([1., 4., 1.]) / 6.

    coeffs[18, :] = np.array([1., 4., 1.]) / 6.
    coeffs[19, :] = np.array([-1., 2., -1.]) / dy

    return coeffs



####################################################################################################################

#                                                  HIGHER ORDER                                                    #

####################################################################################################################





@njit
def get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator(q, iMin, iMax, jMin, jMax, order, evolCoeffs):
    # allocation de variables pour chaque coeff
    # u equation - u term
    uEq_uTerm_xp1, uEq_uTerm_x, uEq_uTerm_xm1 = evolCoeffs[0]
    uEq_uTerm_yp1, uEq_uTerm_y, uEq_uTerm_ym1 = evolCoeffs[1]

    # u equation - v term
    uEq_vTerm_xp1, uEq_vTerm_x, uEq_vTerm_xm1 = evolCoeffs[2]
    uEq_vTerm_yp1, uEq_vTerm_y, uEq_vTerm_ym1 = evolCoeffs[3]

    # u equation - p term
    uEq_pTerm_xp1, uEq_pTerm_x, uEq_pTerm_xm1 = evolCoeffs[4]
    uEq_pTerm_yp1, uEq_pTerm_y, uEq_pTerm_ym1 = evolCoeffs[5]

    # v equation - u term
    vEq_uTerm_xp1, vEq_uTerm_x, vEq_uTerm_xm1 = evolCoeffs[6]
    vEq_uTerm_yp1, vEq_uTerm_y, vEq_uTerm_ym1 = evolCoeffs[7]

    # v equation - v term
    vEq_vTerm_xp1, vEq_vTerm_x, vEq_vTerm_xm1 = evolCoeffs[8]
    vEq_vTerm_yp1, vEq_vTerm_y, vEq_vTerm_ym1 = evolCoeffs[9]

    # v equation - p term
    vEq_pTerm_xp1, vEq_pTerm_x, vEq_pTerm_xm1 = evolCoeffs[10]
    vEq_pTerm_yp1, vEq_pTerm_y, vEq_pTerm_ym1 = evolCoeffs[11]

    # p equation - u term
    pEq_uTerm_xp1, pEq_uTerm_x, pEq_uTerm_xm1 = evolCoeffs[12]
    pEq_uTerm_yp1, pEq_uTerm_y, pEq_uTerm_ym1 = evolCoeffs[13]

    # p equation - v term
    pEq_vTerm_xp1, pEq_vTerm_x, pEq_vTerm_xm1 = evolCoeffs[14]
    pEq_vTerm_yp1, pEq_vTerm_y, pEq_vTerm_ym1 = evolCoeffs[15]

    # p equation - p term (cas spécial avec mass et second_der)
    pEq_pTerm1_xp1, pEq_pTerm1_x, pEq_pTerm1_xm1 = evolCoeffs[16]
    pEq_pTerm1_yp1, pEq_pTerm1_y, pEq_pTerm1_ym1 = evolCoeffs[17]

    pEq_pTerm2_xp1, pEq_pTerm2_x, pEq_pTerm2_xm1 = evolCoeffs[18]
    pEq_pTerm2_yp1, pEq_pTerm2_y, pEq_pTerm2_ym1 = evolCoeffs[19]


    # print("="*30, "\n"*2)
    # print(" "*14, "Test", "\n"*2)
    # print("="*30, "\n"*2)


    
    divF = np.zeros((np.shape(q)))
    # INTERMEDIATE QUANTITIES
    shape = np.shape(q)[:-1]
    uEq_uTerm_yOp, uEq_vTerm_yOp, uEq_pTerm_yOp, vEq_uTerm_yOp, vEq_vTerm_yOp, vEq_pTerm_yOp, \
      pEq_uTerm_yOp, pEq_vTerm_yOp, pEq_pTerm1_yOp, pEq_pTerm2_yOp = np.zeros(((10, ) + shape))


    # COMPUTATIONS OF THE Y OPERATORS ON THE STENCIL iMin-1:iMax+1 x jMin:jMax
    for i in range(iMin-1, iMax+2):
      for j in range(jMin, jMax+1):
        for k in range(order): # compute qtties for all dofs
          for l in range(order): # compute qtties for all dofs
            # U equation
            uEq_uTerm_yOp[i, j, k, l] = uEq_uTerm_yp1[l, 0] * q[i, j+1, k, 0, XVEL] + uEq_uTerm_y[l, 0] * q[i, j, k, 0, XVEL] + uEq_uTerm_ym1[l, 0] * q[i, j-1, k, 0, XVEL]
            uEq_vTerm_yOp[i, j, k, l] = uEq_vTerm_yp1[l, 0] * q[i, j+1, k, 0, YVEL] + uEq_vTerm_y[l, 0] * q[i, j, k, 0, YVEL] + uEq_vTerm_ym1[l, 0] * q[i, j-1, k, 0, YVEL]
            uEq_pTerm_yOp[i, j, k, l] = uEq_pTerm_yp1[l, 0] * q[i, j+1, k, 0, PRES] + uEq_pTerm_y[l, 0] * q[i, j, k, 0, PRES] + uEq_pTerm_ym1[l, 0] * q[i, j-1, k, 0, PRES]


            # V equation
            vEq_uTerm_yOp[i, j, k, l] = vEq_uTerm_yp1[l, 0] * q[i, j+1, k, 0, XVEL] + vEq_uTerm_y[l, 0] * q[i, j, k, 0, XVEL] + vEq_uTerm_ym1[l, 0] * q[i, j-1, k, 0, XVEL]
            vEq_vTerm_yOp[i, j, k, l] = vEq_vTerm_yp1[l, 0] * q[i, j+1, k, 0, YVEL] + vEq_vTerm_y[l, 0] * q[i, j, k, 0, YVEL] + vEq_vTerm_ym1[l, 0] * q[i, j-1, k, 0, YVEL]
            vEq_pTerm_yOp[i, j, k, l] = vEq_pTerm_yp1[l, 0] * q[i, j+1, k, 0, PRES] + vEq_pTerm_y[l, 0] * q[i, j, k, 0, PRES] + vEq_pTerm_ym1[l, 0] * q[i, j-1, k, 0, PRES]


            # P equation
            pEq_uTerm_yOp[i, j, k, l] = pEq_uTerm_yp1[l, 0] * q[i, j+1, k, 0, XVEL] + pEq_uTerm_y[l, 0] * q[i, j, k, 0, XVEL] + pEq_uTerm_ym1[l, 0] * q[i, j-1, k, 0, XVEL]
            pEq_vTerm_yOp[i, j, k, l] = pEq_vTerm_yp1[l, 0] * q[i, j+1, k, 0, YVEL] + pEq_vTerm_y[l, 0] * q[i, j, k, 0, YVEL] + pEq_vTerm_ym1[l, 0] * q[i, j-1, k, 0, YVEL]
            pEq_pTerm1_yOp[i, j, k, l] = pEq_pTerm1_yp1[l, 0] * q[i, j+1, k, 0, PRES] + pEq_pTerm1_y[l, 0] * q[i, j, k, 0, PRES] + pEq_pTerm1_ym1[l, 0] * q[i, j-1, k, 0, PRES]
            pEq_pTerm2_yOp[i, j, k, l] = pEq_pTerm2_yp1[l, 0] * q[i, j+1, k, 0, PRES] + pEq_pTerm2_y[l, 0] * q[i, j, k, 0, PRES] + pEq_pTerm2_ym1[l, 0] * q[i, j-1, k, 0, PRES]

            for p in range(1, order): 
              # U equation
              uEq_uTerm_yOp[i, j, k, l] += uEq_uTerm_yp1[l, p] * q[i, j+1, k, p, XVEL] + uEq_uTerm_y[l, p] * q[i, j, k, p, XVEL] + uEq_uTerm_ym1[l, p] * q[i, j-1, k, p, XVEL]
              uEq_vTerm_yOp[i, j, k, l] += uEq_vTerm_yp1[l, p] * q[i, j+1, k, p, YVEL] + uEq_vTerm_y[l, p] * q[i, j, k, p, YVEL] + uEq_vTerm_ym1[l, p] * q[i, j-1, k, p, YVEL]
              uEq_pTerm_yOp[i, j, k, l] += uEq_pTerm_yp1[l, p] * q[i, j+1, k, p, PRES] + uEq_pTerm_y[l, p] * q[i, j, k, p, PRES] + uEq_pTerm_ym1[l, p] * q[i, j-1, k, p, PRES]

              # V equation
              vEq_uTerm_yOp[i, j, k, l] += vEq_uTerm_yp1[l, p] * q[i, j+1, k, p, XVEL] + vEq_uTerm_y[l, p] * q[i, j, k, p, XVEL] + vEq_uTerm_ym1[l, p] * q[i, j-1, k, p, XVEL]
              vEq_vTerm_yOp[i, j, k, l] += vEq_vTerm_yp1[l, p] * q[i, j+1, k, p, YVEL] + vEq_vTerm_y[l, p] * q[i, j, k, p, YVEL] + vEq_vTerm_ym1[l, p] * q[i, j-1, k, p, YVEL]
              vEq_pTerm_yOp[i, j, k, l] += vEq_pTerm_yp1[l, p] * q[i, j+1, k, p, PRES] + vEq_pTerm_y[l, p] * q[i, j, k, p, PRES] + vEq_pTerm_ym1[l, p] * q[i, j-1, k, p, PRES]

              # P equation
              pEq_uTerm_yOp[i, j, k, l] += pEq_uTerm_yp1[l, p] * q[i, j+1, k, p, XVEL] + pEq_uTerm_y[l, p] * q[i, j, k, p, XVEL] + pEq_uTerm_ym1[l, p] * q[i, j-1, k, p, XVEL]
              pEq_vTerm_yOp[i, j, k, l] += pEq_vTerm_yp1[l, p] * q[i, j+1, k, p, YVEL] + pEq_vTerm_y[l, p] * q[i, j, k, p, YVEL] + pEq_vTerm_ym1[l, p] * q[i, j-1, k, p, YVEL]
              pEq_pTerm1_yOp[i, j, k, l] += pEq_pTerm1_yp1[l, p] * q[i, j+1, k, p, PRES] + pEq_pTerm1_y[l, p] * q[i, j, k, p, PRES] + pEq_pTerm1_ym1[l, p] * q[i, j-1, k, p, PRES]
              pEq_pTerm2_yOp[i, j, k, l] += pEq_pTerm2_yp1[l, p] * q[i, j+1, k, p, PRES] + pEq_pTerm2_y[l, p] * q[i, j, k, p, PRES] + pEq_pTerm2_ym1[l, p] * q[i, j-1, k, p, PRES]





    # COMPUTATIONS OF THE Y OPERATORS ON THE STENCIL iMin-1:iMax+1 x jMin:jMax
    for i in range(iMin, iMax+1):
      for j in range(jMin, jMax+1):
        for k in range(order): # compute qtties for all dofs
          for l in range(order): # compute qtties for all dofs
            # # U equation
            # uEq_uTerm_ijkl = np.dot(uEq_uTerm_xp1[k], uEq_uTerm_yOp[i+1, j, :, l]) + np.dot(uEq_uTerm_x[k], uEq_uTerm_yOp[i, j, :, l]) \
            #                  + np.dot(uEq_uTerm_xm1[k], uEq_uTerm_yOp[i-1, j, :, l])
            # uEq_vTerm_ijkl = np.dot(uEq_vTerm_xp1[k], uEq_vTerm_yOp[i+1, j, :, l]) + np.dot(uEq_vTerm_x[k], uEq_vTerm_yOp[i, j, :, l]) \
            #                  + np.dot(uEq_vTerm_xm1[k], uEq_vTerm_yOp[i-1, j, :, l])
            # uEq_pTerm_ijkl = np.dot(uEq_pTerm_xp1[k], uEq_pTerm_yOp[i+1, j, :, l]) + np.dot(uEq_pTerm_x[k], uEq_pTerm_yOp[i, j, :, l]) \
            #                  + np.dot(uEq_pTerm_xm1[k], uEq_pTerm_yOp[i-1, j, :, l])


            # # V equation
            # vEq_uTerm_ijkl = np.dot(vEq_uTerm_xp1[k], vEq_uTerm_yOp[i+1, j, :, l]) + np.dot(vEq_uTerm_x[k], vEq_uTerm_yOp[i, j, :, l]) \
            #                   + np.dot(vEq_uTerm_xm1[k], vEq_uTerm_yOp[i-1, j, :, l])
            # vEq_vTerm_ijkl = np.dot(vEq_vTerm_xp1[k], vEq_vTerm_yOp[i+1, j, :, l]) + np.dot(vEq_vTerm_x[k], vEq_vTerm_yOp[i, j, :, l]) \
            #                   + np.dot(vEq_vTerm_xm1[k], vEq_vTerm_yOp[i-1, j, :, l])
            # vEq_pTerm_ijkl = np.dot(vEq_pTerm_xp1[k], vEq_pTerm_yOp[i+1, j, :, l]) + np.dot(vEq_pTerm_x[k], vEq_pTerm_yOp[i, j, :, l]) \
            #                   + np.dot(vEq_pTerm_xm1[k], vEq_pTerm_yOp[i-1, j, :, l])


            # # P equation
            # pEq_uTerm_ijkl = np.dot(pEq_uTerm_xp1[k], pEq_uTerm_yOp[i+1, j, :, l]) + np.dot(pEq_uTerm_x[k], pEq_uTerm_yOp[i, j, :, l]) \
            #                   + np.dot(pEq_uTerm_xm1[k], pEq_uTerm_yOp[i-1, j, :, l])
            # pEq_vTerm_ijkl = np.dot(pEq_vTerm_xp1[k], pEq_vTerm_yOp[i+1, j, :, l]) + np.dot(pEq_vTerm_x[k], pEq_vTerm_yOp[i, j, :, l]) \
            #                   + np.dot(pEq_vTerm_xm1[k], pEq_vTerm_yOp[i-1, j, :, l])
            # pEq_pTerm1_ijkl = np.dot(pEq_pTerm1_xp1[k], pEq_pTerm1_yOp[i+1, j, :, l]) + np.dot(pEq_pTerm1_x[k], pEq_pTerm1_yOp[i, j, :, l]) \
            #                   + np.dot(pEq_pTerm1_xm1[k], pEq_pTerm1_yOp[i-1, j, :, l])
            # pEq_pTerm2_ijkl = np.dot(pEq_pTerm2_xp1[k], pEq_pTerm2_yOp[i+1, j, :, l]) + np.dot(pEq_pTerm2_x[k], pEq_pTerm2_yOp[i, j, :, l]) \
            #                   + np.dot(pEq_pTerm2_xm1[k], pEq_pTerm2_yOp[i-1, j, :, l])

            # U equation
            uEq_uTerm_ijkl = uEq_uTerm_xp1[k, 0] * uEq_uTerm_yOp[i+1, j, 0, l] + uEq_uTerm_x[k, 0] * uEq_uTerm_yOp[i, j, 0, l] + uEq_uTerm_xm1[k, 0] * uEq_uTerm_yOp[i-1, j, 0, l]
            uEq_vTerm_ijkl = uEq_vTerm_xp1[k, 0] * uEq_vTerm_yOp[i+1, j, 0, l] + uEq_vTerm_x[k, 0] * uEq_vTerm_yOp[i, j, 0, l] + uEq_vTerm_xm1[k, 0] * uEq_vTerm_yOp[i-1, j, 0, l]
            uEq_pTerm_ijkl = uEq_pTerm_xp1[k, 0] * uEq_pTerm_yOp[i+1, j, 0, l] + uEq_pTerm_x[k, 0] * uEq_pTerm_yOp[i, j, 0, l] + uEq_pTerm_xm1[k, 0] * uEq_pTerm_yOp[i-1, j, 0, l]


            # V equation
            vEq_uTerm_ijkl = vEq_uTerm_xp1[k, 0] * vEq_uTerm_yOp[i+1, j, 0, l] + vEq_uTerm_x[k, 0] * vEq_uTerm_yOp[i, j, 0, l] + vEq_uTerm_xm1[k, 0] * vEq_uTerm_yOp[i-1, j, 0, l]
            vEq_vTerm_ijkl = vEq_vTerm_xp1[k, 0] * vEq_vTerm_yOp[i+1, j, 0, l] + vEq_vTerm_x[k, 0] * vEq_vTerm_yOp[i, j, 0, l] + vEq_vTerm_xm1[k, 0] * vEq_vTerm_yOp[i-1, j, 0, l]
            vEq_pTerm_ijkl = vEq_pTerm_xp1[k, 0] * vEq_pTerm_yOp[i+1, j, 0, l] + vEq_pTerm_x[k, 0] * vEq_pTerm_yOp[i, j, 0, l] + vEq_pTerm_xm1[k, 0] * vEq_pTerm_yOp[i-1, j, 0, l]


            # P equation
            pEq_uTerm_ijkl = pEq_uTerm_xp1[k, 0] * pEq_uTerm_yOp[i+1, j, 0, l] + pEq_uTerm_x[k, 0] * pEq_uTerm_yOp[i, j, 0, l] + pEq_uTerm_xm1[k, 0] * pEq_uTerm_yOp[i-1, j, 0, l]
            pEq_vTerm_ijkl = pEq_vTerm_xp1[k, 0] * pEq_vTerm_yOp[i+1, j, 0, l] + pEq_vTerm_x[k, 0] * pEq_vTerm_yOp[i, j, 0, l] + pEq_vTerm_xm1[k, 0] * pEq_vTerm_yOp[i-1, j, 0, l]
            pEq_pTerm1_ijkl = pEq_pTerm1_xp1[k, 0] * pEq_pTerm1_yOp[i+1, j, 0, l] + pEq_pTerm1_x[k, 0] * pEq_pTerm1_yOp[i, j, 0, l] + pEq_pTerm1_xm1[k, 0] * pEq_pTerm1_yOp[i-1, j, 0, l]
            pEq_pTerm2_ijkl = pEq_pTerm2_xp1[k, 0] * pEq_pTerm2_yOp[i+1, j, 0, l] + pEq_pTerm2_x[k, 0] * pEq_pTerm2_yOp[i, j, 0, l] + pEq_pTerm2_xm1[k, 0] * pEq_pTerm2_yOp[i-1, j, 0, l]
  
            for p in range(1, order): 
              # U equation
              uEq_uTerm_ijkl += uEq_uTerm_xp1[k, p] * uEq_uTerm_yOp[i+1, j, p, l] + uEq_uTerm_x[k, p] * uEq_uTerm_yOp[i, j, p, l] + uEq_uTerm_xm1[k, p] * uEq_uTerm_yOp[i-1, j, p, l]
              uEq_vTerm_ijkl += uEq_vTerm_xp1[k, p] * uEq_vTerm_yOp[i+1, j, p, l] + uEq_vTerm_x[k, p] * uEq_vTerm_yOp[i, j, p, l] + uEq_vTerm_xm1[k, p] * uEq_vTerm_yOp[i-1, j, p, l]
              uEq_pTerm_ijkl += uEq_pTerm_xp1[k, p] * uEq_pTerm_yOp[i+1, j, p, l] + uEq_pTerm_x[k, p] * uEq_pTerm_yOp[i, j, p, l] + uEq_pTerm_xm1[k, p] * uEq_pTerm_yOp[i-1, j, p, l]
  
  
              # V equation
              vEq_uTerm_ijkl += vEq_uTerm_xp1[k, p] * vEq_uTerm_yOp[i+1, j, p, l] + vEq_uTerm_x[k, p] * vEq_uTerm_yOp[i, j, p, l] + vEq_uTerm_xm1[k, p] * vEq_uTerm_yOp[i-1, j, p, l]
              vEq_vTerm_ijkl += vEq_vTerm_xp1[k, p] * vEq_vTerm_yOp[i+1, j, p, l] + vEq_vTerm_x[k, p] * vEq_vTerm_yOp[i, j, p, l] + vEq_vTerm_xm1[k, p] * vEq_vTerm_yOp[i-1, j, p, l]
              vEq_pTerm_ijkl += vEq_pTerm_xp1[k, p] * vEq_pTerm_yOp[i+1, j, p, l] + vEq_pTerm_x[k, p] * vEq_pTerm_yOp[i, j, p, l] + vEq_pTerm_xm1[k, p] * vEq_pTerm_yOp[i-1, j, p, l]
  
  
              # P equation
              pEq_uTerm_ijkl += pEq_uTerm_xp1[k, p] * pEq_uTerm_yOp[i+1, j, p, l] + pEq_uTerm_x[k, p] * pEq_uTerm_yOp[i, j, p, l] + pEq_uTerm_xm1[k, p] * pEq_uTerm_yOp[i-1, j, p, l]
              pEq_vTerm_ijkl += pEq_vTerm_xp1[k, p] * pEq_vTerm_yOp[i+1, j, p, l] + pEq_vTerm_x[k, p] * pEq_vTerm_yOp[i, j, p, l] + pEq_vTerm_xm1[k, p] * pEq_vTerm_yOp[i-1, j, p, l]
              pEq_pTerm1_ijkl += pEq_pTerm1_xp1[k, p] * pEq_pTerm1_yOp[i+1, j, p, l] + pEq_pTerm1_x[k, p] * pEq_pTerm1_yOp[i, j, p, l] + pEq_pTerm1_xm1[k, p] * pEq_pTerm1_yOp[i-1, j, p, l]
              pEq_pTerm2_ijkl += pEq_pTerm2_xp1[k, p] * pEq_pTerm2_yOp[i+1, j, p, l] + pEq_pTerm2_x[k, p] * pEq_pTerm2_yOp[i, j, p, l] + pEq_pTerm2_xm1[k, p] * pEq_pTerm2_yOp[i-1, j, p, l]


            ### ASSEMBLING
            divF[i, j, k, l, XVEL] = uEq_uTerm_ijkl + uEq_vTerm_ijkl + uEq_pTerm_ijkl
            divF[i, j, k, l, YVEL] = vEq_uTerm_ijkl + vEq_vTerm_ijkl + vEq_pTerm_ijkl
            divF[i, j, k, l, PRES] = pEq_uTerm_ijkl + pEq_vTerm_ijkl + pEq_pTerm1_ijkl + pEq_pTerm2_ijkl

    return divF





@njit
def get_evaluation_of_arbitrar_order_SUPG_weighted_evolution_operator_AND_physical_terms(q, iMin, iMax, jMin, jMax, order, evolCoeffs):
    # allocation de variables pour chaque coeff
    # u equation - u term
    uEq_uTerm_xp1, uEq_uTerm_x, uEq_uTerm_xm1 = evolCoeffs[0]
    uEq_uTerm_yp1, uEq_uTerm_y, uEq_uTerm_ym1 = evolCoeffs[1]

    # u equation - v term
    uEq_vTerm_xp1, uEq_vTerm_x, uEq_vTerm_xm1 = evolCoeffs[2]
    uEq_vTerm_yp1, uEq_vTerm_y, uEq_vTerm_ym1 = evolCoeffs[3]

    # u equation - p term
    uEq_pTerm_xp1, uEq_pTerm_x, uEq_pTerm_xm1 = evolCoeffs[4]
    uEq_pTerm_yp1, uEq_pTerm_y, uEq_pTerm_ym1 = evolCoeffs[5]

    # v equation - u term
    vEq_uTerm_xp1, vEq_uTerm_x, vEq_uTerm_xm1 = evolCoeffs[6]
    vEq_uTerm_yp1, vEq_uTerm_y, vEq_uTerm_ym1 = evolCoeffs[7]

    # v equation - v term
    vEq_vTerm_xp1, vEq_vTerm_x, vEq_vTerm_xm1 = evolCoeffs[8]
    vEq_vTerm_yp1, vEq_vTerm_y, vEq_vTerm_ym1 = evolCoeffs[9]

    # v equation - p term
    vEq_pTerm_xp1, vEq_pTerm_x, vEq_pTerm_xm1 = evolCoeffs[10]
    vEq_pTerm_yp1, vEq_pTerm_y, vEq_pTerm_ym1 = evolCoeffs[11]

    # p equation - u term
    pEq_uTerm_xp1, pEq_uTerm_x, pEq_uTerm_xm1 = evolCoeffs[12]
    pEq_uTerm_yp1, pEq_uTerm_y, pEq_uTerm_ym1 = evolCoeffs[13]

    # p equation - v term
    pEq_vTerm_xp1, pEq_vTerm_x, pEq_vTerm_xm1 = evolCoeffs[14]
    pEq_vTerm_yp1, pEq_vTerm_y, pEq_vTerm_ym1 = evolCoeffs[15]

    # p equation - p term (cas spécial avec mass et second_der)
    pEq_pTerm1_xp1, pEq_pTerm1_x, pEq_pTerm1_xm1 = evolCoeffs[16]
    pEq_pTerm1_yp1, pEq_pTerm1_y, pEq_pTerm1_ym1 = evolCoeffs[17]

    pEq_pTerm2_xp1, pEq_pTerm2_x, pEq_pTerm2_xm1 = evolCoeffs[18]
    pEq_pTerm2_yp1, pEq_pTerm2_y, pEq_pTerm2_ym1 = evolCoeffs[19]

    divF = np.zeros((np.shape(q)))
    physicalTerms = np.zeros((np.shape(q))) # dx_p, dy_p and div U in this order
    # INTERMEDIATE QUANTITIES
    shape = np.shape(q)[:-1]
    uEq_uTerm_yOp, uEq_vTerm_yOp, uEq_pTerm_yOp, vEq_uTerm_yOp, vEq_vTerm_yOp, vEq_pTerm_yOp, \
      pEq_uTerm_yOp, pEq_vTerm_yOp, pEq_pTerm1_yOp, pEq_pTerm2_yOp = np.zeros(((10, ) + shape))


    # COMPUTATIONS OF THE Y OPERATORS ON THE STENCIL iMin-1:iMax+1 x jMin:jMax
    for i in range(iMin-1, iMax+3):
      for j in range(jMin, jMax+2):
        for k in range(order): # compute qtties for all dofs
          for l in range(order): # compute qtties for all dofs
            # U equation
            uEq_uTerm_yOp[i, j, k, l] = uEq_uTerm_yp1[l, 0] * q[i, j+1, k, 0, XVEL] + uEq_uTerm_y[l, 0] * q[i, j, k, 0, XVEL] + uEq_uTerm_ym1[l, 0] * q[i, j-1, k, 0, XVEL]
            uEq_vTerm_yOp[i, j, k, l] = uEq_vTerm_yp1[l, 0] * q[i, j+1, k, 0, YVEL] + uEq_vTerm_y[l, 0] * q[i, j, k, 0, YVEL] + uEq_vTerm_ym1[l, 0] * q[i, j-1, k, 0, YVEL]
            uEq_pTerm_yOp[i, j, k, l] = uEq_pTerm_yp1[l, 0] * q[i, j+1, k, 0, PRES] + uEq_pTerm_y[l, 0] * q[i, j, k, 0, PRES] + uEq_pTerm_ym1[l, 0] * q[i, j-1, k, 0, PRES]


            # V equation
            vEq_uTerm_yOp[i, j, k, l] = vEq_uTerm_yp1[l, 0] * q[i, j+1, k, 0, XVEL] + vEq_uTerm_y[l, 0] * q[i, j, k, 0, XVEL] + vEq_uTerm_ym1[l, 0] * q[i, j-1, k, 0, XVEL]
            vEq_vTerm_yOp[i, j, k, l] = vEq_vTerm_yp1[l, 0] * q[i, j+1, k, 0, YVEL] + vEq_vTerm_y[l, 0] * q[i, j, k, 0, YVEL] + vEq_vTerm_ym1[l, 0] * q[i, j-1, k, 0, YVEL]
            vEq_pTerm_yOp[i, j, k, l] = vEq_pTerm_yp1[l, 0] * q[i, j+1, k, 0, PRES] + vEq_pTerm_y[l, 0] * q[i, j, k, 0, PRES] + vEq_pTerm_ym1[l, 0] * q[i, j-1, k, 0, PRES]


            # P equation
            pEq_uTerm_yOp[i, j, k, l] = pEq_uTerm_yp1[l, 0] * q[i, j+1, k, 0, XVEL] + pEq_uTerm_y[l, 0] * q[i, j, k, 0, XVEL] + pEq_uTerm_ym1[l, 0] * q[i, j-1, k, 0, XVEL]
            pEq_vTerm_yOp[i, j, k, l] = pEq_vTerm_yp1[l, 0] * q[i, j+1, k, 0, YVEL] + pEq_vTerm_y[l, 0] * q[i, j, k, 0, YVEL] + pEq_vTerm_ym1[l, 0] * q[i, j-1, k, 0, YVEL]
            pEq_pTerm1_yOp[i, j, k, l] = pEq_pTerm1_yp1[l, 0] * q[i, j+1, k, 0, PRES] + pEq_pTerm1_y[l, 0] * q[i, j, k, 0, PRES] + pEq_pTerm1_ym1[l, 0] * q[i, j-1, k, 0, PRES]
            pEq_pTerm2_yOp[i, j, k, l] = pEq_pTerm2_yp1[l, 0] * q[i, j+1, k, 0, PRES] + pEq_pTerm2_y[l, 0] * q[i, j, k, 0, PRES] + pEq_pTerm2_ym1[l, 0] * q[i, j-1, k, 0, PRES]

            for p in range(1, order): 
              # U equation
              uEq_uTerm_yOp[i, j, k, l] += uEq_uTerm_yp1[l, p] * q[i, j+1, k, p, XVEL] + uEq_uTerm_y[l, p] * q[i, j, k, p, XVEL] + uEq_uTerm_ym1[l, p] * q[i, j-1, k, p, XVEL]
              uEq_vTerm_yOp[i, j, k, l] += uEq_vTerm_yp1[l, p] * q[i, j+1, k, p, YVEL] + uEq_vTerm_y[l, p] * q[i, j, k, p, YVEL] + uEq_vTerm_ym1[l, p] * q[i, j-1, k, p, YVEL]
              uEq_pTerm_yOp[i, j, k, l] += uEq_pTerm_yp1[l, p] * q[i, j+1, k, p, PRES] + uEq_pTerm_y[l, p] * q[i, j, k, p, PRES] + uEq_pTerm_ym1[l, p] * q[i, j-1, k, p, PRES]

              # V equation
              vEq_uTerm_yOp[i, j, k, l] += vEq_uTerm_yp1[l, p] * q[i, j+1, k, p, XVEL] + vEq_uTerm_y[l, p] * q[i, j, k, p, XVEL] + vEq_uTerm_ym1[l, p] * q[i, j-1, k, p, XVEL]
              vEq_vTerm_yOp[i, j, k, l] += vEq_vTerm_yp1[l, p] * q[i, j+1, k, p, YVEL] + vEq_vTerm_y[l, p] * q[i, j, k, p, YVEL] + vEq_vTerm_ym1[l, p] * q[i, j-1, k, p, YVEL]
              vEq_pTerm_yOp[i, j, k, l] += vEq_pTerm_yp1[l, p] * q[i, j+1, k, p, PRES] + vEq_pTerm_y[l, p] * q[i, j, k, p, PRES] + vEq_pTerm_ym1[l, p] * q[i, j-1, k, p, PRES]

              # P equation
              pEq_uTerm_yOp[i, j, k, l] += pEq_uTerm_yp1[l, p] * q[i, j+1, k, p, XVEL] + pEq_uTerm_y[l, p] * q[i, j, k, p, XVEL] + pEq_uTerm_ym1[l, p] * q[i, j-1, k, p, XVEL]
              pEq_vTerm_yOp[i, j, k, l] += pEq_vTerm_yp1[l, p] * q[i, j+1, k, p, YVEL] + pEq_vTerm_y[l, p] * q[i, j, k, p, YVEL] + pEq_vTerm_ym1[l, p] * q[i, j-1, k, p, YVEL]
              pEq_pTerm1_yOp[i, j, k, l] += pEq_pTerm1_yp1[l, p] * q[i, j+1, k, p, PRES] + pEq_pTerm1_y[l, p] * q[i, j, k, p, PRES] + pEq_pTerm1_ym1[l, p] * q[i, j-1, k, p, PRES]
              pEq_pTerm2_yOp[i, j, k, l] += pEq_pTerm2_yp1[l, p] * q[i, j+1, k, p, PRES] + pEq_pTerm2_y[l, p] * q[i, j, k, p, PRES] + pEq_pTerm2_ym1[l, p] * q[i, j-1, k, p, PRES]





    # COMPUTATIONS OF THE Y OPERATORS ON THE STENCIL iMin-1:iMax+1 x jMin:jMax
    for i in range(iMin, iMax+2):
      for j in range(jMin, jMax+2):
        for k in range(order): # compute qtties for all dofs
          for l in range(order): # compute qtties for all dofs
            # U equation
            uEq_uTerm_ijkl = uEq_uTerm_xp1[k, 0] * uEq_uTerm_yOp[i+1, j, 0, l] + uEq_uTerm_x[k, 0] * uEq_uTerm_yOp[i, j, 0, l] + uEq_uTerm_xm1[k, 0] * uEq_uTerm_yOp[i-1, j, 0, l]
            uEq_vTerm_ijkl = uEq_vTerm_xp1[k, 0] * uEq_vTerm_yOp[i+1, j, 0, l] + uEq_vTerm_x[k, 0] * uEq_vTerm_yOp[i, j, 0, l] + uEq_vTerm_xm1[k, 0] * uEq_vTerm_yOp[i-1, j, 0, l]
            uEq_pTerm_ijkl = uEq_pTerm_xp1[k, 0] * uEq_pTerm_yOp[i+1, j, 0, l] + uEq_pTerm_x[k, 0] * uEq_pTerm_yOp[i, j, 0, l] + uEq_pTerm_xm1[k, 0] * uEq_pTerm_yOp[i-1, j, 0, l]


            # V equation
            vEq_uTerm_ijkl = vEq_uTerm_xp1[k, 0] * vEq_uTerm_yOp[i+1, j, 0, l] + vEq_uTerm_x[k, 0] * vEq_uTerm_yOp[i, j, 0, l] + vEq_uTerm_xm1[k, 0] * vEq_uTerm_yOp[i-1, j, 0, l]
            vEq_vTerm_ijkl = vEq_vTerm_xp1[k, 0] * vEq_vTerm_yOp[i+1, j, 0, l] + vEq_vTerm_x[k, 0] * vEq_vTerm_yOp[i, j, 0, l] + vEq_vTerm_xm1[k, 0] * vEq_vTerm_yOp[i-1, j, 0, l]
            vEq_pTerm_ijkl = vEq_pTerm_xp1[k, 0] * vEq_pTerm_yOp[i+1, j, 0, l] + vEq_pTerm_x[k, 0] * vEq_pTerm_yOp[i, j, 0, l] + vEq_pTerm_xm1[k, 0] * vEq_pTerm_yOp[i-1, j, 0, l]


            # P equation
            pEq_uTerm_ijkl = pEq_uTerm_xp1[k, 0] * pEq_uTerm_yOp[i+1, j, 0, l] + pEq_uTerm_x[k, 0] * pEq_uTerm_yOp[i, j, 0, l] + pEq_uTerm_xm1[k, 0] * pEq_uTerm_yOp[i-1, j, 0, l]
            pEq_vTerm_ijkl = pEq_vTerm_xp1[k, 0] * pEq_vTerm_yOp[i+1, j, 0, l] + pEq_vTerm_x[k, 0] * pEq_vTerm_yOp[i, j, 0, l] + pEq_vTerm_xm1[k, 0] * pEq_vTerm_yOp[i-1, j, 0, l]
            pEq_pTerm1_ijkl = pEq_pTerm1_xp1[k, 0] * pEq_pTerm1_yOp[i+1, j, 0, l] + pEq_pTerm1_x[k, 0] * pEq_pTerm1_yOp[i, j, 0, l] + pEq_pTerm1_xm1[k, 0] * pEq_pTerm1_yOp[i-1, j, 0, l]
            pEq_pTerm2_ijkl = pEq_pTerm2_xp1[k, 0] * pEq_pTerm2_yOp[i+1, j, 0, l] + pEq_pTerm2_x[k, 0] * pEq_pTerm2_yOp[i, j, 0, l] + pEq_pTerm2_xm1[k, 0] * pEq_pTerm2_yOp[i-1, j, 0, l]
  
            for p in range(1, order): 
              # U equation
              uEq_uTerm_ijkl += uEq_uTerm_xp1[k, p] * uEq_uTerm_yOp[i+1, j, p, l] + uEq_uTerm_x[k, p] * uEq_uTerm_yOp[i, j, p, l] + uEq_uTerm_xm1[k, p] * uEq_uTerm_yOp[i-1, j, p, l]
              uEq_vTerm_ijkl += uEq_vTerm_xp1[k, p] * uEq_vTerm_yOp[i+1, j, p, l] + uEq_vTerm_x[k, p] * uEq_vTerm_yOp[i, j, p, l] + uEq_vTerm_xm1[k, p] * uEq_vTerm_yOp[i-1, j, p, l]
              uEq_pTerm_ijkl += uEq_pTerm_xp1[k, p] * uEq_pTerm_yOp[i+1, j, p, l] + uEq_pTerm_x[k, p] * uEq_pTerm_yOp[i, j, p, l] + uEq_pTerm_xm1[k, p] * uEq_pTerm_yOp[i-1, j, p, l]
  
  
              # V equation
              vEq_uTerm_ijkl += vEq_uTerm_xp1[k, p] * vEq_uTerm_yOp[i+1, j, p, l] + vEq_uTerm_x[k, p] * vEq_uTerm_yOp[i, j, p, l] + vEq_uTerm_xm1[k, p] * vEq_uTerm_yOp[i-1, j, p, l]
              vEq_vTerm_ijkl += vEq_vTerm_xp1[k, p] * vEq_vTerm_yOp[i+1, j, p, l] + vEq_vTerm_x[k, p] * vEq_vTerm_yOp[i, j, p, l] + vEq_vTerm_xm1[k, p] * vEq_vTerm_yOp[i-1, j, p, l]
              vEq_pTerm_ijkl += vEq_pTerm_xp1[k, p] * vEq_pTerm_yOp[i+1, j, p, l] + vEq_pTerm_x[k, p] * vEq_pTerm_yOp[i, j, p, l] + vEq_pTerm_xm1[k, p] * vEq_pTerm_yOp[i-1, j, p, l]
  
  
              # P equation
              pEq_uTerm_ijkl += pEq_uTerm_xp1[k, p] * pEq_uTerm_yOp[i+1, j, p, l] + pEq_uTerm_x[k, p] * pEq_uTerm_yOp[i, j, p, l] + pEq_uTerm_xm1[k, p] * pEq_uTerm_yOp[i-1, j, p, l]
              pEq_vTerm_ijkl += pEq_vTerm_xp1[k, p] * pEq_vTerm_yOp[i+1, j, p, l] + pEq_vTerm_x[k, p] * pEq_vTerm_yOp[i, j, p, l] + pEq_vTerm_xm1[k, p] * pEq_vTerm_yOp[i-1, j, p, l]
              pEq_pTerm1_ijkl += pEq_pTerm1_xp1[k, p] * pEq_pTerm1_yOp[i+1, j, p, l] + pEq_pTerm1_x[k, p] * pEq_pTerm1_yOp[i, j, p, l] + pEq_pTerm1_xm1[k, p] * pEq_pTerm1_yOp[i-1, j, p, l]
              pEq_pTerm2_ijkl += pEq_pTerm2_xp1[k, p] * pEq_pTerm2_yOp[i+1, j, p, l] + pEq_pTerm2_x[k, p] * pEq_pTerm2_yOp[i, j, p, l] + pEq_pTerm2_xm1[k, p] * pEq_pTerm2_yOp[i-1, j, p, l]


            ### ASSEMBLINGS
            divF[i, j, k, l, XVEL] = uEq_uTerm_ijkl + uEq_vTerm_ijkl + uEq_pTerm_ijkl
            divF[i, j, k, l, YVEL] = vEq_uTerm_ijkl + vEq_vTerm_ijkl + vEq_pTerm_ijkl
            divF[i, j, k, l, PRES] = pEq_uTerm_ijkl + pEq_vTerm_ijkl + pEq_pTerm1_ijkl + pEq_pTerm2_ijkl

            physicalTerms[i, j, k, l, XVEL] = uEq_pTerm_ijkl
            physicalTerms[i, j, k, l, YVEL] = vEq_pTerm_ijkl
            physicalTerms[i, j, k, l, PRES] = pEq_uTerm_ijkl + pEq_vTerm_ijkl
            

    return divF, physicalTerms





@njit
def get_evaluation_of_arbitrar_order_SUPG_physical_terms(q, iMin, iMax, jMin, jMax, order, evolCoeffs):
    # allocation de variables pour chaque coeff
    # u equation - p term
    uEq_pTerm_xp1, uEq_pTerm_x, uEq_pTerm_xm1 = evolCoeffs[4]
    uEq_pTerm_yp1, uEq_pTerm_y, uEq_pTerm_ym1 = evolCoeffs[5]

    # v equation - p term
    vEq_pTerm_xp1, vEq_pTerm_x, vEq_pTerm_xm1 = evolCoeffs[10]
    vEq_pTerm_yp1, vEq_pTerm_y, vEq_pTerm_ym1 = evolCoeffs[11]

    # p equation - u term
    pEq_uTerm_xp1, pEq_uTerm_x, pEq_uTerm_xm1 = evolCoeffs[12]
    pEq_uTerm_yp1, pEq_uTerm_y, pEq_uTerm_ym1 = evolCoeffs[13]

    # p equation - v term
    pEq_vTerm_xp1, pEq_vTerm_x, pEq_vTerm_xm1 = evolCoeffs[14]
    pEq_vTerm_yp1, pEq_vTerm_y, pEq_vTerm_ym1 = evolCoeffs[15]

    physicalTerms = np.zeros((np.shape(q))) # dx_p, dy_p and div U in this order
    # INTERMEDIATE QUANTITIES
    shape = np.shape(q)[:-1]
    uEq_pTerm_yOp, vEq_pTerm_yOp, pEq_uTerm_yOp, pEq_vTerm_yOp = np.zeros(((4, ) + shape))


    # COMPUTATIONS OF THE Y OPERATORS ON THE STENCIL iMin-1:iMax+1 x jMin:jMax
    for i in range(iMin-1, iMax+3):
      for j in range(jMin, jMax+2):
        for k in range(order): # compute qtties for all dofs
          for l in range(order): # compute qtties for all dofs
            # U equation
            uEq_pTerm_yOp[i, j, k, l] = uEq_pTerm_yp1[l, 0] * q[i, j+1, k, 0, PRES] + uEq_pTerm_y[l, 0] * q[i, j, k, 0, PRES] + uEq_pTerm_ym1[l, 0] * q[i, j-1, k, 0, PRES]

            # V equation
            vEq_pTerm_yOp[i, j, k, l] = vEq_pTerm_yp1[l, 0] * q[i, j+1, k, 0, PRES] + vEq_pTerm_y[l, 0] * q[i, j, k, 0, PRES] + vEq_pTerm_ym1[l, 0] * q[i, j-1, k, 0, PRES]

            # P equation
            pEq_uTerm_yOp[i, j, k, l] = pEq_uTerm_yp1[l, 0] * q[i, j+1, k, 0, XVEL] + pEq_uTerm_y[l, 0] * q[i, j, k, 0, XVEL] + pEq_uTerm_ym1[l, 0] * q[i, j-1, k, 0, XVEL]
            pEq_vTerm_yOp[i, j, k, l] = pEq_vTerm_yp1[l, 0] * q[i, j+1, k, 0, YVEL] + pEq_vTerm_y[l, 0] * q[i, j, k, 0, YVEL] + pEq_vTerm_ym1[l, 0] * q[i, j-1, k, 0, YVEL]
            
            for p in range(1, order): 
              # U equation
              uEq_pTerm_yOp[i, j, k, l] += uEq_pTerm_yp1[l, p] * q[i, j+1, k, p, PRES] + uEq_pTerm_y[l, p] * q[i, j, k, p, PRES] + uEq_pTerm_ym1[l, p] * q[i, j-1, k, p, PRES]

              # V equation
              vEq_pTerm_yOp[i, j, k, l] += vEq_pTerm_yp1[l, p] * q[i, j+1, k, p, PRES] + vEq_pTerm_y[l, p] * q[i, j, k, p, PRES] + vEq_pTerm_ym1[l, p] * q[i, j-1, k, p, PRES]

              # P equation
              pEq_uTerm_yOp[i, j, k, l] += pEq_uTerm_yp1[l, p] * q[i, j+1, k, p, XVEL] + pEq_uTerm_y[l, p] * q[i, j, k, p, XVEL] + pEq_uTerm_ym1[l, p] * q[i, j-1, k, p, XVEL]
              pEq_vTerm_yOp[i, j, k, l] += pEq_vTerm_yp1[l, p] * q[i, j+1, k, p, YVEL] + pEq_vTerm_y[l, p] * q[i, j, k, p, YVEL] + pEq_vTerm_ym1[l, p] * q[i, j-1, k, p, YVEL]



    # COMPUTATIONS OF THE Y OPERATORS ON THE STENCIL iMin-1:iMax+1 x jMin:jMax
    for i in range(iMin, iMax+2):
      for j in range(jMin, jMax+2):
        for k in range(order): # compute qtties for all dofs
          for l in range(order): # compute qtties for all dofs
            # U equation
            uEq_pTerm_ijkl = uEq_pTerm_xp1[k, 0] * uEq_pTerm_yOp[i+1, j, 0, l] + uEq_pTerm_x[k, 0] * uEq_pTerm_yOp[i, j, 0, l] + uEq_pTerm_xm1[k, 0] * uEq_pTerm_yOp[i-1, j, 0, l]

            # V equation
            vEq_pTerm_ijkl = vEq_pTerm_xp1[k, 0] * vEq_pTerm_yOp[i+1, j, 0, l] + vEq_pTerm_x[k, 0] * vEq_pTerm_yOp[i, j, 0, l] + vEq_pTerm_xm1[k, 0] * vEq_pTerm_yOp[i-1, j, 0, l]

            # P equation
            pEq_uTerm_ijkl = pEq_uTerm_xp1[k, 0] * pEq_uTerm_yOp[i+1, j, 0, l] + pEq_uTerm_x[k, 0] * pEq_uTerm_yOp[i, j, 0, l] + pEq_uTerm_xm1[k, 0] * pEq_uTerm_yOp[i-1, j, 0, l]
            pEq_vTerm_ijkl = pEq_vTerm_xp1[k, 0] * pEq_vTerm_yOp[i+1, j, 0, l] + pEq_vTerm_x[k, 0] * pEq_vTerm_yOp[i, j, 0, l] + pEq_vTerm_xm1[k, 0] * pEq_vTerm_yOp[i-1, j, 0, l]
            
            for p in range(1, order): 
              # U equation
              uEq_pTerm_ijkl += uEq_pTerm_xp1[k, p] * uEq_pTerm_yOp[i+1, j, p, l] + uEq_pTerm_x[k, p] * uEq_pTerm_yOp[i, j, p, l] + uEq_pTerm_xm1[k, p] * uEq_pTerm_yOp[i-1, j, p, l]
  
              # V equation
              vEq_pTerm_ijkl += vEq_pTerm_xp1[k, p] * vEq_pTerm_yOp[i+1, j, p, l] + vEq_pTerm_x[k, p] * vEq_pTerm_yOp[i, j, p, l] + vEq_pTerm_xm1[k, p] * vEq_pTerm_yOp[i-1, j, p, l]
  
              # P equation
              pEq_uTerm_ijkl += pEq_uTerm_xp1[k, p] * pEq_uTerm_yOp[i+1, j, p, l] + pEq_uTerm_x[k, p] * pEq_uTerm_yOp[i, j, p, l] + pEq_uTerm_xm1[k, p] * pEq_uTerm_yOp[i-1, j, p, l]
              pEq_vTerm_ijkl += pEq_vTerm_xp1[k, p] * pEq_vTerm_yOp[i+1, j, p, l] + pEq_vTerm_x[k, p] * pEq_vTerm_yOp[i, j, p, l] + pEq_vTerm_xm1[k, p] * pEq_vTerm_yOp[i-1, j, p, l]
              

            ### ASSEMBLINGS
            physicalTerms[i, j, k, l, XVEL] = uEq_pTerm_ijkl
            physicalTerms[i, j, k, l, YVEL] = vEq_pTerm_ijkl
            physicalTerms[i, j, k, l, PRES] = pEq_uTerm_ijkl + pEq_vTerm_ijkl

    return physicalTerms





@njit
def get_evaluation_of_arbitrar_order_SUPG_weighted_mass_operator(q, iMin, iMax, jMin, jMax, order, massCoeffs):
    # allocation de variables pour chaque coeff
    # u equation - u term
    uEq_uTerm_xp1, uEq_uTerm_x, uEq_uTerm_xm1 = massCoeffs[0]
    uEq_uTerm_yp1, uEq_uTerm_y, uEq_uTerm_ym1 = massCoeffs[1]

    # u equation - v term
    uEq_vTerm_xp1, uEq_vTerm_x, uEq_vTerm_xm1 = massCoeffs[2]
    uEq_vTerm_yp1, uEq_vTerm_y, uEq_vTerm_ym1 = massCoeffs[3]

    # u equation - p term
    uEq_pTerm_xp1, uEq_pTerm_x, uEq_pTerm_xm1 = massCoeffs[4]
    uEq_pTerm_yp1, uEq_pTerm_y, uEq_pTerm_ym1 = massCoeffs[5]

    # v equation - u term
    vEq_uTerm_xp1, vEq_uTerm_x, vEq_uTerm_xm1 = massCoeffs[6]
    vEq_uTerm_yp1, vEq_uTerm_y, vEq_uTerm_ym1 = massCoeffs[7]

    # v equation - v term
    vEq_vTerm_xp1, vEq_vTerm_x, vEq_vTerm_xm1 = massCoeffs[8]
    vEq_vTerm_yp1, vEq_vTerm_y, vEq_vTerm_ym1 = massCoeffs[9]

    # v equation - p term
    vEq_pTerm_xp1, vEq_pTerm_x, vEq_pTerm_xm1 = massCoeffs[10]
    vEq_pTerm_yp1, vEq_pTerm_y, vEq_pTerm_ym1 = massCoeffs[11]

    # p equation - u term
    pEq_uTerm_xp1, pEq_uTerm_x, pEq_uTerm_xm1 = massCoeffs[12]
    pEq_uTerm_yp1, pEq_uTerm_y, pEq_uTerm_ym1 = massCoeffs[13]

    # p equation - v term
    pEq_vTerm_xp1, pEq_vTerm_x, pEq_vTerm_xm1 = massCoeffs[14]
    pEq_vTerm_yp1, pEq_vTerm_y, pEq_vTerm_ym1 = massCoeffs[15]

    # p equation - p term (cas spécial avec mass et second_der)
    pEq_pTerm_xp1, pEq_pTerm_x, pEq_pTerm_xm1 = massCoeffs[16]
    pEq_pTerm_yp1, pEq_pTerm_y, pEq_pTerm_ym1 = massCoeffs[17]


    # print("="*30, "\n"*2)
    # print(" "*14, "Test", "\n"*2)
    # print("="*30, "\n"*2)


    
    divF = np.zeros((np.shape(q)))
    # INTERMEDIATE QUANTITIES
    shape = np.shape(q)[:-1]
    uEq_uTerm_yOp, uEq_vTerm_yOp, uEq_pTerm_yOp, vEq_uTerm_yOp, vEq_vTerm_yOp, vEq_pTerm_yOp, \
      pEq_uTerm_yOp, pEq_vTerm_yOp, pEq_pTerm_yOp, pEq_pTerm2_yOp = np.zeros(((10, ) + shape))


    # COMPUTATIONS OF THE Y OPERATORS ON THE STENCIL iMin-1:iMax+1 x jMin:jMax
    for i in range(iMin-1, iMax+3):
      for j in range(jMin, jMax+2):
        for k in range(order): # compute qtties for all dofs
          for l in range(order): # compute qtties for all dofs
            # U equation
            uEq_uTerm_yOp[i, j, k, l] = uEq_uTerm_yp1[l, 0] * q[i, j+1, k, 0, XVEL] + uEq_uTerm_y[l, 0] * q[i, j, k, 0, XVEL] + uEq_uTerm_ym1[l, 0] * q[i, j-1, k, 0, XVEL]
            # uEq_vTerm_yOp[i, j, k, l] = uEq_vTerm_yp1[l, 0] * q[i, j+1, k, 0, YVEL] + uEq_vTerm_y[l, 0] * q[i, j, k, 0, YVEL] + uEq_vTerm_ym1[l, 0] * q[i, j-1, k, 0, YVEL]
            uEq_pTerm_yOp[i, j, k, l] = uEq_pTerm_yp1[l, 0] * q[i, j+1, k, 0, PRES] + uEq_pTerm_y[l, 0] * q[i, j, k, 0, PRES] + uEq_pTerm_ym1[l, 0] * q[i, j-1, k, 0, PRES]


            # V equation
            # vEq_uTerm_yOp[i, j, k, l] = vEq_uTerm_yp1[l, 0] * q[i, j+1, k, 0, XVEL] + vEq_uTerm_y[l, 0] * q[i, j, k, 0, XVEL] + vEq_uTerm_ym1[l, 0] * q[i, j-1, k, 0, XVEL]
            vEq_vTerm_yOp[i, j, k, l] = vEq_vTerm_yp1[l, 0] * q[i, j+1, k, 0, YVEL] + vEq_vTerm_y[l, 0] * q[i, j, k, 0, YVEL] + vEq_vTerm_ym1[l, 0] * q[i, j-1, k, 0, YVEL]
            vEq_pTerm_yOp[i, j, k, l] = vEq_pTerm_yp1[l, 0] * q[i, j+1, k, 0, PRES] + vEq_pTerm_y[l, 0] * q[i, j, k, 0, PRES] + vEq_pTerm_ym1[l, 0] * q[i, j-1, k, 0, PRES]


            # P equation
            pEq_uTerm_yOp[i, j, k, l] = pEq_uTerm_yp1[l, 0] * q[i, j+1, k, 0, XVEL] + pEq_uTerm_y[l, 0] * q[i, j, k, 0, XVEL] + pEq_uTerm_ym1[l, 0] * q[i, j-1, k, 0, XVEL]
            pEq_vTerm_yOp[i, j, k, l] = pEq_vTerm_yp1[l, 0] * q[i, j+1, k, 0, YVEL] + pEq_vTerm_y[l, 0] * q[i, j, k, 0, YVEL] + pEq_vTerm_ym1[l, 0] * q[i, j-1, k, 0, YVEL]
            pEq_pTerm_yOp[i, j, k, l] = pEq_pTerm_yp1[l, 0] * q[i, j+1, k, 0, PRES] + pEq_pTerm_y[l, 0] * q[i, j, k, 0, PRES] + pEq_pTerm_ym1[l, 0] * q[i, j-1, k, 0, PRES]

            for p in range(1, order): 
              # U equation
              uEq_uTerm_yOp[i, j, k, l] += uEq_uTerm_yp1[l, p] * q[i, j+1, k, p, XVEL] + uEq_uTerm_y[l, p] * q[i, j, k, p, XVEL] + uEq_uTerm_ym1[l, p] * q[i, j-1, k, p, XVEL]
              # uEq_vTerm_yOp[i, j, k, l] += uEq_vTerm_yp1[l, p] * q[i, j+1, k, p, YVEL] + uEq_vTerm_y[l, p] * q[i, j, k, p, YVEL] + uEq_vTerm_ym1[l, p] * q[i, j-1, k, p, YVEL]
              uEq_pTerm_yOp[i, j, k, l] += uEq_pTerm_yp1[l, p] * q[i, j+1, k, p, PRES] + uEq_pTerm_y[l, p] * q[i, j, k, p, PRES] + uEq_pTerm_ym1[l, p] * q[i, j-1, k, p, PRES]

              # V equation
              # vEq_uTerm_yOp[i, j, k, l] += vEq_uTerm_yp1[l, p] * q[i, j+1, k, p, XVEL] + vEq_uTerm_y[l, p] * q[i, j, k, p, XVEL] + vEq_uTerm_ym1[l, p] * q[i, j-1, k, p, XVEL]
              vEq_vTerm_yOp[i, j, k, l] += vEq_vTerm_yp1[l, p] * q[i, j+1, k, p, YVEL] + vEq_vTerm_y[l, p] * q[i, j, k, p, YVEL] + vEq_vTerm_ym1[l, p] * q[i, j-1, k, p, YVEL]
              vEq_pTerm_yOp[i, j, k, l] += vEq_pTerm_yp1[l, p] * q[i, j+1, k, p, PRES] + vEq_pTerm_y[l, p] * q[i, j, k, p, PRES] + vEq_pTerm_ym1[l, p] * q[i, j-1, k, p, PRES]

              # P equation
              pEq_uTerm_yOp[i, j, k, l] += pEq_uTerm_yp1[l, p] * q[i, j+1, k, p, XVEL] + pEq_uTerm_y[l, p] * q[i, j, k, p, XVEL] + pEq_uTerm_ym1[l, p] * q[i, j-1, k, p, XVEL]
              pEq_vTerm_yOp[i, j, k, l] += pEq_vTerm_yp1[l, p] * q[i, j+1, k, p, YVEL] + pEq_vTerm_y[l, p] * q[i, j, k, p, YVEL] + pEq_vTerm_ym1[l, p] * q[i, j-1, k, p, YVEL]
              pEq_pTerm_yOp[i, j, k, l] += pEq_pTerm_yp1[l, p] * q[i, j+1, k, p, PRES] + pEq_pTerm_y[l, p] * q[i, j, k, p, PRES] + pEq_pTerm_ym1[l, p] * q[i, j-1, k, p, PRES]





    # COMPUTATIONS OF THE Y OPERATORS ON THE STENCIL iMin-1:iMax+1 x jMin:jMax
    for i in range(iMin, iMax+2):
      for j in range(jMin, jMax+2):
        for k in range(order): # compute qtties for all dofs
          for l in range(order): # compute qtties for all dofs
            # U equation
            uEq_uTerm_ijkl = uEq_uTerm_xp1[k, 0] * uEq_uTerm_yOp[i+1, j, 0, l] + uEq_uTerm_x[k, 0] * uEq_uTerm_yOp[i, j, 0, l] + uEq_uTerm_xm1[k, 0] * uEq_uTerm_yOp[i-1, j, 0, l]
            # uEq_vTerm_ijkl = uEq_vTerm_xp1[k, 0] * uEq_vTerm_yOp[i+1, j, 0, l] + uEq_vTerm_x[k, 0] * uEq_vTerm_yOp[i, j, 0, l] + uEq_vTerm_xm1[k, 0] * uEq_vTerm_yOp[i-1, j, 0, l]
            uEq_pTerm_ijkl = uEq_pTerm_xp1[k, 0] * uEq_pTerm_yOp[i+1, j, 0, l] + uEq_pTerm_x[k, 0] * uEq_pTerm_yOp[i, j, 0, l] + uEq_pTerm_xm1[k, 0] * uEq_pTerm_yOp[i-1, j, 0, l]


            # V equation
            # vEq_uTerm_ijkl = vEq_uTerm_xp1[k, 0] * vEq_uTerm_yOp[i+1, j, 0, l] + vEq_uTerm_x[k, 0] * vEq_uTerm_yOp[i, j, 0, l] + vEq_uTerm_xm1[k, 0] * vEq_uTerm_yOp[i-1, j, 0, l]
            vEq_vTerm_ijkl = vEq_vTerm_xp1[k, 0] * vEq_vTerm_yOp[i+1, j, 0, l] + vEq_vTerm_x[k, 0] * vEq_vTerm_yOp[i, j, 0, l] + vEq_vTerm_xm1[k, 0] * vEq_vTerm_yOp[i-1, j, 0, l]
            vEq_pTerm_ijkl = vEq_pTerm_xp1[k, 0] * vEq_pTerm_yOp[i+1, j, 0, l] + vEq_pTerm_x[k, 0] * vEq_pTerm_yOp[i, j, 0, l] + vEq_pTerm_xm1[k, 0] * vEq_pTerm_yOp[i-1, j, 0, l]


            # P equation
            pEq_uTerm_ijkl = pEq_uTerm_xp1[k, 0] * pEq_uTerm_yOp[i+1, j, 0, l] + pEq_uTerm_x[k, 0] * pEq_uTerm_yOp[i, j, 0, l] + pEq_uTerm_xm1[k, 0] * pEq_uTerm_yOp[i-1, j, 0, l]
            pEq_vTerm_ijkl = pEq_vTerm_xp1[k, 0] * pEq_vTerm_yOp[i+1, j, 0, l] + pEq_vTerm_x[k, 0] * pEq_vTerm_yOp[i, j, 0, l] + pEq_vTerm_xm1[k, 0] * pEq_vTerm_yOp[i-1, j, 0, l]
            pEq_pTerm_ijkl = pEq_pTerm_xp1[k, 0] * pEq_pTerm_yOp[i+1, j, 0, l] + pEq_pTerm_x[k, 0] * pEq_pTerm_yOp[i, j, 0, l] + pEq_pTerm_xm1[k, 0] * pEq_pTerm_yOp[i-1, j, 0, l]
  
            for p in range(1, order): 
              # U equation
              uEq_uTerm_ijkl += uEq_uTerm_xp1[k, p] * uEq_uTerm_yOp[i+1, j, p, l] + uEq_uTerm_x[k, p] * uEq_uTerm_yOp[i, j, p, l] + uEq_uTerm_xm1[k, p] * uEq_uTerm_yOp[i-1, j, p, l]
              # uEq_vTerm_ijkl += uEq_vTerm_xp1[k, p] * uEq_vTerm_yOp[i+1, j, p, l] + uEq_vTerm_x[k, p] * uEq_vTerm_yOp[i, j, p, l] + uEq_vTerm_xm1[k, p] * uEq_vTerm_yOp[i-1, j, p, l]
              uEq_pTerm_ijkl += uEq_pTerm_xp1[k, p] * uEq_pTerm_yOp[i+1, j, p, l] + uEq_pTerm_x[k, p] * uEq_pTerm_yOp[i, j, p, l] + uEq_pTerm_xm1[k, p] * uEq_pTerm_yOp[i-1, j, p, l]
  
  
              # V equation
              # vEq_uTerm_ijkl += vEq_uTerm_xp1[k, p] * vEq_uTerm_yOp[i+1, j, p, l] + vEq_uTerm_x[k, p] * vEq_uTerm_yOp[i, j, p, l] + vEq_uTerm_xm1[k, p] * vEq_uTerm_yOp[i-1, j, p, l]
              vEq_vTerm_ijkl += vEq_vTerm_xp1[k, p] * vEq_vTerm_yOp[i+1, j, p, l] + vEq_vTerm_x[k, p] * vEq_vTerm_yOp[i, j, p, l] + vEq_vTerm_xm1[k, p] * vEq_vTerm_yOp[i-1, j, p, l]
              vEq_pTerm_ijkl += vEq_pTerm_xp1[k, p] * vEq_pTerm_yOp[i+1, j, p, l] + vEq_pTerm_x[k, p] * vEq_pTerm_yOp[i, j, p, l] + vEq_pTerm_xm1[k, p] * vEq_pTerm_yOp[i-1, j, p, l]
  
  
              # P equation
              pEq_uTerm_ijkl += pEq_uTerm_xp1[k, p] * pEq_uTerm_yOp[i+1, j, p, l] + pEq_uTerm_x[k, p] * pEq_uTerm_yOp[i, j, p, l] + pEq_uTerm_xm1[k, p] * pEq_uTerm_yOp[i-1, j, p, l]
              pEq_vTerm_ijkl += pEq_vTerm_xp1[k, p] * pEq_vTerm_yOp[i+1, j, p, l] + pEq_vTerm_x[k, p] * pEq_vTerm_yOp[i, j, p, l] + pEq_vTerm_xm1[k, p] * pEq_vTerm_yOp[i-1, j, p, l]
              pEq_pTerm_ijkl += pEq_pTerm_xp1[k, p] * pEq_pTerm_yOp[i+1, j, p, l] + pEq_pTerm_x[k, p] * pEq_pTerm_yOp[i, j, p, l] + pEq_pTerm_xm1[k, p] * pEq_pTerm_yOp[i-1, j, p, l]


            ### ASSEMBLING
            # divF[i, j, k, l, XVEL] = uEq_uTerm_ijkl + uEq_vTerm_ijkl + uEq_pTerm_ijkl
            # divF[i, j, k, l, YVEL] = vEq_uTerm_ijkl + vEq_vTerm_ijkl + vEq_pTerm_ijkl
            # divF[i, j, k, l, PRES] = pEq_uTerm_ijkl + pEq_vTerm_ijkl + pEq_pTerm_ijkl

            divF[i, j, k, l, XVEL] = uEq_uTerm_ijkl + uEq_pTerm_ijkl
            divF[i, j, k, l, YVEL] = vEq_vTerm_ijkl + vEq_pTerm_ijkl
            divF[i, j, k, l, PRES] = pEq_uTerm_ijkl + pEq_vTerm_ijkl + pEq_pTerm_ijkl

    return divF



def getLagElementsStandardOperators(order):
  # STANDARDS OPERATORS
  ### FIRST ORDER FOR LAGRANGE BASIS
  mass_o1 = np.array([[[1]], [[4]], [[1]]], dtype=float) / 6.
  left_der_o1 = np.array([[[-1]], [[0]], [[1]]], dtype=float) / 2.
  right_der_o1 = - left_der_o1
  second_der_o1 = np.array([[[-1]], [[2]], [[-1]]], dtype=float)

  ### SECOND ORDER FOR LAGRANGE BASIS
  mass_o2 = np.array([[[- 1, 0], [2, 0]], [[8, 2], [2, 16]], [[-1, 2], [0, 0]]], dtype=float) / 30.
  left_der_o2 = np.array([[[1, 0], [-4, 0]], [[0, -4], [4, 0]], [[-1, 4], [0, 0]]], dtype=float) / 6.
  right_der_o2 = - left_der_o2
  second_der_o2 = np.array([[[1, 0], [-8, 0]], [[14, -8], [-8, 16]], [[1, -8], [0, 0]]], dtype=float) / 3.

  ### THIRD ORDER FOR LAGRANGE BASIS
  mass_o3 = np.array([[[19, 0, 0], [-36, 0, 0], [99, 0, 0]], [[256, 99, -36], [99, 648, -81], [-36, -81, 648]], [[19, -36, 99], [0, 0, 0], [0, 0, 0]]], dtype=float) / 1680.
  left_der_o3 = np.array([[[-7, 0, 0], [24, 0, 0], [-57, 0, 0]], [[0, -57, 24], [57, 0, -81], [-24, 81, 0]], [[7, -24, 57], [0, 0, 0], [0, 0, 0]]], dtype=float) / 80.
  right_der_o3 = - left_der_o3
  second_der_o3 = np.array([[[-13, 0, 0], [54, 0, 0], [-189, 0, 0]], [[296, -189, 54], [-189, 432, -297], [54, -297, 432]], [[-13, 54, -189], [0, 0, 0], [0, 0, 0]]], dtype=float) / 40.

  Id = np.array([ np.zeros((order, order)), np.eye(order), np.zeros((order, order)) ])

  if order == 1:
    mass = mass_o1
    left_der = left_der_o1
    right_der = right_der_o1
    second_der = second_der_o1
  elif order == 2:
    mass = mass_o2
    left_der = left_der_o2
    right_der = right_der_o2
    second_der = second_der_o2
  elif order == 3:
    mass = mass_o3
    left_der = left_der_o3
    right_der = right_der_o3
    second_der = second_der_o3

  return mass, left_der, right_der, second_der, Id



# LES OPERATEURS SONT DECOMPOSES EN 3 : UN QUI AGIT EN +1, UN AUTRE EN 0 ET LE DERNIER EN -1
def get_arbitrar_order_SUPG__weighted_evolution_operator_coeffs(order, dx, dy):
    evolCoeffs = np.zeros((20, 3, order, order), dtype=np.float64)

    mass, left_der, right_der, second_der, Id = getLagElementsStandardOperators(order)

    # U EQUATION
    ### uEq_uTerm : derivée seconde en x et masse en y
    evolCoeffs[0] = 0.5 * second_der / dx
    evolCoeffs[1] = mass

    ### uEq_vTerm : derivées simples mais diffusion par dx
    evolCoeffs[2] = 0.5 * left_der
    evolCoeffs[3] = right_der / dy

    ### uEq_pTerm : derivée simple en x et masse en y
    evolCoeffs[4] = right_der / dx
    evolCoeffs[5] = mass

    # V EQUATION
    ### vEq_uTerm : derivées simples
    evolCoeffs[6] = right_der / dx
    evolCoeffs[7] = 0.5 * left_der

    ### vEq_vTerm : masse en x et derivée seconde en y
    evolCoeffs[8] = mass
    evolCoeffs[9] = 0.5 * second_der / dy

    ### vEq_pTerm : masse en x et derivée simple en y
    evolCoeffs[10] = mass
    evolCoeffs[11] = right_der / dy

    # P EQUATION
    ### pEq_uTerm : derivée simple en x et masse en y
    evolCoeffs[12] = right_der / dx
    evolCoeffs[13] = mass

    ### pEq_vTerm : masse en x et derivée simple en y
    evolCoeffs[14] = mass
    evolCoeffs[15] = right_der / dy

    ### pEq_pTerms
    evolCoeffs[16] = 0.5 * second_der / dx
    evolCoeffs[17] = mass

    evolCoeffs[18] = mass
    evolCoeffs[19] = 0.5 * second_der / dy


    # LUMPED VERSION : on divise chaque ligne de chaque opérateur par la somme des coeff de la même ligne de la masse
    # C'est en fait cet opérateur qui apparaît dans l'algorithme
    for i in range(order):
      somme_ligne_i_masse = np.sum(mass[:, i, :])
      evolCoeffs[:, :, i, :] = evolCoeffs[:, :, i, :] / somme_ligne_i_masse

    return evolCoeffs



# LES OPERATEURS SONT DECOMPOSES EN 3 : UN QUI AGIT EN +1, UN AUTRE EN 0 ET LE DERNIER EN -1
def get_arbitrar_order_SUPG_weighted_mass_operator_coeffs(order, dx, dy):
    massCoeffs = np.zeros((18, 3, order, order), dtype=np.float64)

    mass, left_der, right_der, second_der, Id = getLagElementsStandardOperators(order)

    # U EQUATION
    ### uEq_uTerm : masses
    massCoeffs[0] = mass
    massCoeffs[1] = mass

    ### uEq_vTerm : rien
    massCoeffs[2] = 0.
    massCoeffs[3] = 0.

    ### uEq_pTerm : derivée simple en x et masse en y
    massCoeffs[4] = 0.5 * left_der
    massCoeffs[5] = mass

    # V EQUATION
    ### vEq_uTerm : rien
    massCoeffs[6] = 0.
    massCoeffs[7] = 0.

    ### vEq_vTerm : masses
    massCoeffs[8] = mass
    massCoeffs[9] = mass

    ### vEq_pTerm : masse en x et derivée simple en y
    massCoeffs[10] = mass
    massCoeffs[11] = 0.5 * right_der

    # P EQUATION
    ### pEq_uTerm : derivée simple en x et masse en y
    massCoeffs[12] = 0.5 * right_der
    massCoeffs[13] = mass

    ### pEq_vTerm : masse en x et derivée simple en y
    massCoeffs[14] = mass
    massCoeffs[15] = 0.5 * right_der

    ### pEq_pTerm
    massCoeffs[16] = mass
    massCoeffs[17] = mass


    # LUMPED VERSION : on divise chaque ligne de chaque opérateur par la somme des coeff de la même ligne de la masse
    # C'est en fait cet opérateur qui apparaît dans l'algorithme
    for i in range(order):
      somme_ligne_i_masse = np.sum(mass[:, i, :])
      massCoeffs[:, :, i, :] = massCoeffs[:, :, i, :] / somme_ligne_i_masse

    return massCoeffs





####################################################################################################################

#                                                 CLASSICAL UPWIND                                                 #

####################################################################################################################

