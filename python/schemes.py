# Modules généraux
import numpy as np

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
      (    uEq_uTerm["x"][0] * (  uEq_uTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, XVEL] \
                                + uEq_uTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
                                + uEq_uTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, XVEL]) \
        +  uEq_uTerm["x"][1] * (  uEq_uTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, XVEL] \
                                + uEq_uTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, XVEL] \
                                + uEq_uTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, XVEL]) \
        +  uEq_uTerm["x"][2] * (  uEq_uTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, XVEL] \
                                + uEq_uTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, XVEL] \
                                + uEq_uTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, XVEL]) \
      ) * (1. / dx) * (dy / sum(uEq_uTerm["y"])) \
    \
    + dx/2. * \
      (    uEq_vTerm["x"][0] * (  uEq_vTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                                + uEq_vTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, YVEL] \
                                + uEq_vTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, YVEL]) \
        +  uEq_vTerm["x"][1] * (  uEq_vTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
                                + uEq_vTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, YVEL] \
                                + uEq_vTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, YVEL]) \
        +  uEq_vTerm["x"][2] * (  uEq_vTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
                                + uEq_vTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, YVEL] \
                                + uEq_vTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, YVEL]) \
      ) \
    \
    + 1. * \
      (    uEq_pTerm["x"][0] * (  uEq_pTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                + uEq_pTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                + uEq_pTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, PRES]) \
        +  uEq_pTerm["x"][1] * (  uEq_pTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                                + uEq_pTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                                + uEq_pTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, PRES]) \
        +  uEq_pTerm["x"][2] * (  uEq_pTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                + uEq_pTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                + uEq_pTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, PRES]) \
      ) * (dy / sum(uEq_pTerm["y"]) )  \
    ) 


    ### v equation 

    divF[iMin:iMax+1, jMin:jMax+1, YVEL] =  \
    1. / (dx * dy) * \
    ( \
    dy/2. * \
      (    vEq_uTerm["x"][0] * (  vEq_uTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, XVEL] \
                                + vEq_uTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
                                + vEq_uTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, XVEL]) \
        +  vEq_uTerm["x"][1] * (  vEq_uTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, XVEL] \
                                + vEq_uTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, XVEL] \
                                + vEq_uTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, XVEL]) \
        +  vEq_uTerm["x"][2] * (  vEq_uTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, XVEL] \
                                + vEq_uTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, XVEL] \
                                + vEq_uTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, XVEL]) \
      ) \
    \
    + dy/2. * \
      (    vEq_vTerm["x"][0] * (  vEq_vTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                                + vEq_vTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, YVEL] \
                                + vEq_vTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, YVEL]) \
        +  vEq_vTerm["x"][1] * (  vEq_vTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
                                + vEq_vTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, YVEL] \
                                + vEq_vTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, YVEL]) \
        +  vEq_vTerm["x"][2] * (  vEq_vTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
                                + vEq_vTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, YVEL] \
                                + vEq_vTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, YVEL]) \
      ) * (dx / sum(vEq_vTerm["x"])) * (1. / dy) \
    \
    + 1. * \
      (    vEq_pTerm["x"][0] * (  vEq_pTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                + vEq_pTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                + vEq_pTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, PRES]) \
        +  vEq_pTerm["x"][1] * (  vEq_pTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                                + vEq_pTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                                + vEq_pTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, PRES]) \
        +  vEq_pTerm["x"][2] * (  vEq_pTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                + vEq_pTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                + vEq_pTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, PRES]) \
      ) * (dx / sum(vEq_pTerm["x"]) )  \
    ) 


    ### p equation 

    divF[iMin:iMax+1, jMin:jMax+1, PRES] =  \
    1. / (dx * dy) * \
    ( \
    1. * \
      (    pEq_uTerm["x"][0] * (  pEq_uTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, XVEL] \
                                + pEq_uTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, XVEL] \
                                + pEq_uTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, XVEL]) \
        +  pEq_uTerm["x"][1] * (  pEq_uTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, XVEL] \
                                + pEq_uTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, XVEL] \
                                + pEq_uTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, XVEL]) \
        +  pEq_uTerm["x"][2] * (  pEq_uTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, XVEL] \
                                + pEq_uTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, XVEL] \
                                + pEq_uTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, XVEL]) \
      ) * (dy / sum(pEq_uTerm["y"]) )  \
    \
    + 1. * \
      (    pEq_vTerm["x"][0] * (  pEq_vTerm["y"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, YVEL] \
                                + pEq_vTerm["y"][1] * q[iMin+1:iMax+2, jMin:jMax+1, YVEL] \
                                + pEq_vTerm["y"][2] * q[iMin+1:iMax+2, jMin-1:jMax, YVEL]) \
        +  pEq_vTerm["x"][1] * (  pEq_vTerm["y"][0] * q[iMin:iMax+1, jMin+1:jMax+2, YVEL] \
                                + pEq_vTerm["y"][1] * q[iMin:iMax+1, jMin:jMax+1, YVEL] \
                                + pEq_vTerm["y"][2] * q[iMin:iMax+1, jMin-1:jMax, YVEL]) \
        +  pEq_vTerm["x"][2] * (  pEq_vTerm["y"][0] * q[iMin-1:iMax, jMin+1:jMax+2, YVEL] \
                                + pEq_vTerm["y"][1] * q[iMin-1:iMax, jMin:jMax+1, YVEL] \
                                + pEq_vTerm["y"][2] * q[iMin-1:iMax, jMin-1:jMax, YVEL]) \
      ) * (dx / sum(pEq_vTerm["x"]) )  \
    \
    + dx/2. * \
      (    pEq_pTerm["second_der"][0] * (  pEq_pTerm["mass"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                         + pEq_pTerm["mass"][1] * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                         + pEq_pTerm["mass"][2] * q[iMin+1:iMax+2, jMin-1:jMax, PRES]) \
        +  pEq_pTerm["second_der"][1] * (  pEq_pTerm["mass"][0] * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                                         + pEq_pTerm["mass"][1] * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                                         + pEq_pTerm["mass"][2] * q[iMin:iMax+1, jMin-1:jMax, PRES]) \
        +  pEq_pTerm["second_der"][2] * (  pEq_pTerm["mass"][0] * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                         + pEq_pTerm["mass"][1] * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                         + pEq_pTerm["mass"][2] * q[iMin-1:iMax, jMin-1:jMax, PRES]) \
      ) * (1. / dx) * (dy / sum(pEq_pTerm["mass"])) \
    + dy/2. * \
      (    pEq_pTerm["mass"][0] * (  pEq_pTerm["second_der"][0] * q[iMin+1:iMax+2, jMin+1:jMax+2, PRES] \
                                   + pEq_pTerm["second_der"][1] * q[iMin+1:iMax+2, jMin:jMax+1, PRES] \
                                   + pEq_pTerm["second_der"][2] * q[iMin+1:iMax+2, jMin-1:jMax, PRES]) \
        +  pEq_pTerm["mass"][1] * (  pEq_pTerm["second_der"][0] * q[iMin:iMax+1, jMin+1:jMax+2, PRES] \
                                   + pEq_pTerm["second_der"][1] * q[iMin:iMax+1, jMin:jMax+1, PRES] \
                                   + pEq_pTerm["second_der"][2] * q[iMin:iMax+1, jMin-1:jMax, PRES]) \
        +  pEq_pTerm["mass"][2] * (  pEq_pTerm["second_der"][0] * q[iMin-1:iMax, jMin+1:jMax+2, PRES] \
                                   + pEq_pTerm["second_der"][1] * q[iMin-1:iMax, jMin:jMax+1, PRES] \
                                   + pEq_pTerm["second_der"][2] * q[iMin-1:iMax, jMin-1:jMax, PRES]) \
      ) * (dx / sum(pEq_pTerm["mass"])) * (1. / dy) \
    )

    # print("max(abs(divF))", np.max(abs(divF)))

    return divF






####################################################################################################################

#                                                 CLASSICAL UPWIND                                                 #

####################################################################################################################



def getApproxDivFlux(q, grid, schemeChoice, operators):
    if schemeChoice == 1 :
        return get_UPWIND_divFlux(q, grid)
    if schemeChoice == 2 :
        return get_SUPG_developped_divFlux(q, grid)
    if schemeChoice == 3 :
        return get_modif_SUPG_developped_divFlux(q, grid, operators)