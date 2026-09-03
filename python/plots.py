# Modules généraux
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import imageio.v2 as imageio
import os

# Modules perso
from config import XVEL, YVEL, PRES 
import solutions


# SMALL ANNEXE ROUINES
def getSchemeKeys(params):
    schemeChoice = params["scheme_choice"]
    
    if (schemeChoice == -1):
        scheme_name = "TEST"
        scheme_short = "test_"
    if (schemeChoice == 1):
        scheme_name = "Upwind"
        scheme_short = "UPW_"
    elif (schemeChoice == 2):
        scheme_name = "SUPG"
        scheme_short = "SUPG_"
    elif (schemeChoice == 3):
        scheme_name = "SUPG modifié"
        scheme_short = "modSUPG_"
    elif (schemeChoice == 4):
        scheme_name = "SUPG modifié optim"
        scheme_short = "optModSUPG_"
    
    return scheme_name, scheme_short



def getSimulKeys(params):
    
    simulationChoice = params["simulation_choice"]

    if (simulationChoice == 1):
        simul_name =  "etat constant"
        simul_short = "cst_"
    elif (simulationChoice == 2):
        degree = params["solution_parameters"]["analytical_periodic"]["theta"]
        simul_name =  "smooth flow ("+str(round(degree, 3))+"°)"
        simul_short = "smooth_flow_"+str(round(degree, 3))+"_"
    elif (simulationChoice == 3):
        simul_name =  "vortex"
        simul_short = "vortex_"
    elif (simulationChoice == 4):
        simul_name =  "constant + perturbation"
        simul_short = "cst+pert_"
    elif (simulationChoice == 5):
        simul_name =  "vortex + perturbation"
        simul_short = "vortex+pert_"
    elif (simulationChoice == 6):
        simul_name =  "random noise"
        simul_short = "noise_"
    elif (simulationChoice == 7):
        simul_name =  "checkerboard"
        simul_short = "check_"

    return simul_name, simul_short
    

def getObsKeys(i_obs_list):
    obs_name_table = ["", "figure(s) 1D", "figure(s) 2D", "ecart des solutions", "Norme sup", "Erreur sup", "Masse totale", \
                      "Norme des termes physiques", "Etude de convergence", "Série de tests"]
    obs_short_table = ["", " ", " ", " ", "norm_sup_", "err_sup_", "tot_mass_", "norm_divU_", "conv_", " "]
    
    obs_name = ""
    obs_short = ""

    if len(i_obs_list) > 1:
        for i in i_obs_list:
            obs_name = f"{obs_name} + {obs_name_table[i]}" 
            obs_short = f"{obs_name} + {obs_short_table[i]}"
        return obs_name[3:], obs_short[3:]
    elif len(i_obs_list) == 1:
        obs_name = obs_name_table[i_obs_list[0]]
        obs_short = obs_short_table[i_obs_list[0]]
    else :
        obs_name = "aucune observable"


    return obs_name, obs_short



def getDoObservablesTable(i_obs_list):

    observablesTable = [False for i in range(9+1)]
    for i in i_obs_list :
        observablesTable[i] = True

    # On rajoute des valeurs à la liste pour garder la même structure que dans le fichier paramètres
    return observablesTable


def choicePrints(params):
    # getting parameters to print
    scheme_name = getSchemeKeys(params)[0]
    simulation_name = getSimulKeys(params)[0]
    obs_name = getObsKeys(params["observables_choices"])[0]
    plotLoc = params["plot_parameters"]["plot_loc"]
    Nx = params["grid_parameters"]["mesh_parameters"]["Nx"]
    Ny = params["grid_parameters"]["mesh_parameters"]["Ny"]
    finalTime, CFL = params["time_parameters"].values()

    # prints
    print("")
    print(f"===================================================================")
    print(f"                                                                   ")
    print(f"      Choix de simulation : {simulation_name}                      ")
    print(f"          Choix de schéma : {scheme_name}                          ")
    print(f"                 Maillage : {Nx} x {Ny}                            ")
    print(f"              Temps final : {finalTime}                            ")
    print(f"                      CFL : {CFL}                                  ")
    print(f"        Observable testée : {obs_name}                             ")
    print(f"Emplacement des résultats : {plotLoc}                              ")
    print(f"                                                                   ")
    print(f"===================================================================")
    print("")


####################################################################################################################

#                                                 SOLUTIONS PLOTS                                                  #

####################################################################################################################



def getVisualisationGrid(params):
    order = params["order"]
    xL, xR, yL, yR = params["grid_parameters"]["domain_parameters"].values()
    Nx, Ny, nGhost = params["grid_parameters"]["mesh_parameters"].values()
    nVis = params["number_visualisation_points"]

    # Grille de visualisation
    if order == 0:
        nVis = 1

    xCoord = np.linspace(xL, xR, Nx * nVis + 1)
    yCoord = np.linspace(yL, yR, Ny * nVis + 1)
    X, Y = np.meshgrid(xCoord, yCoord, indexing="ij")

    return X, Y





def makeSolutionsPlots(X, Y, q, q_prep, time, i_obs, params, h, pdf_writers, gif_writers):
    simulationChoice = params["simulation_choice"]

    # Plot ou non de la solution exacte
    do_exact_pdf_plot = False
    do_exact_gif_plot = False
    if (simulationChoice == 2 or (simulationChoice == 3 and time < 1.e-14)):
        do_exact_pdf_plot = (params["plot_parameters"]["do_exact_pdf_plot"] == "y")
        do_exact_gif_plot = (params["plot_parameters"]["do_exact_gif_plot"] == "y")

    q_exact = []
    if do_exact_pdf_plot or do_exact_gif_plot or i_obs == 3 :
        q_exact = solutions.getSolution(time, X, Y, simulationChoice, params["solution_parameters"])

    do_exact_plot = len(q_exact) > 0

    if (i_obs == 1):
        makeSolutions1DPlots(X, Y, q, do_exact_plot, q_exact, time, params, h, pdf_writers["1D"], gif_writers["1D"])

    if (i_obs == 2):
        if (simulationChoice == 4 or simulationChoice == 5):
            perturbedSolution2DPlots(X, Y, q, q_prep, time, params, h, pdf_writers["2D"], gif_writers["2D"])
        else :
            makeSolutions2DPlots(X, Y, q, do_exact_plot, q_exact, time, params, h, pdf_writers["2D"], gif_writers["2D"])

    if (i_obs == 3):
        makeSolutions2DDiffNormPlots(X, Y, q, q_exact, time, params, h, pdf_writers["2D_diff"], gif_writers["2D_diff"])



# 1D Solutions
def makeSolutions1DPlots(X, Y, q, do_exact_plot, q_exact, time, params, h, pdf_writers, gif_writers):
    scheme_name, scheme_short = getSchemeKeys(params)
    simul_name, simul_short = getSimulKeys(params)
    do_pdf_plot = (params["plot_parameters"]["do_pdf_plot"] == "y")
    do_gif_plot = (params["plot_parameters"]["do_gif_plot"] == "y")
    CFL = params["time_parameters"]["CFL_number"]

    i_section = params["plot_parameters"]["section"]

    Nx, Ny, nGhost = params["grid_parameters"]["mesh_parameters"].values()
    i_slice, j_slice = params["plot_parameters"]["slice_indices"]

    if (i_section == 1):
        domaine_silce = X[:, j_slice]
        q_slice = q[:, j_slice]
        slice_name = "coupe horizontale"

        if do_exact_plot :
            q_exact_slice = q_exact[:, j_slice]
    
    elif (i_section == 2):
        domaine_silce = Y[i_slice, :]
        q_slice = q[i_slice, :]
        slice_name = "coupe verticale"

        if do_exact_plot :
            q_exact_slice = q_exact[i_slice, :]

    elif (i_section == 3):
        if (Nx != Ny):
            print("section diagonale mais malliage non adapté")
        else :
            slice_name = "coupe oblique"
            domaine_silce = np.sqrt( X[:, 0]**2 + Y[0, :]**2 )

            q_slice = np.diagonal(q, axis1=0, axis2=1).T
            
            if do_exact_plot :
                q_exact_slice = np.diagonal(q_exact, axis1=0, axis2=1).T
                
    plots = [
        (XVEL, "U", "u approchée", "u exacte", "u"),
        (YVEL, "V", "v approchée", "v exacte", "v"),
        (PRES, "P", "p approchée", "p exacte", "p"),
    ]

    plots = [
        (t[0], f"{simul_short}{scheme_short}{t[1]}_1D_h{round(h, 5)}_CFL{CFL}") + t[2:]
        for t in plots
    ]

    # FIGURES FOR APPROXIMATE SOLUTIONS
    for var, filename, approx_label, exact_label, var_symbol in plots:
        fig, ax = plt.subplots()
        ax.plot(domaine_silce, q_slice[:, var], label=approx_label)
        if do_exact_plot :
            ax.plot(domaine_silce, q_exact_slice[:, var], label=exact_label)
        ax.set_title(f"{simul_name} avec {scheme_name} : {slice_name} de {var_symbol} à t={round(time, 1)} \n et h = {round(h, 5)}, CFL = {CFL}")
        ax.set_xlabel("x")
        ax.set_ylabel(var_symbol)
        ax.legend()
        plt.tight_layout()

        if (do_pdf_plot) :
            pdf_writers[filename].savefig(fig)
        
        if (do_gif_plot) :
            # On écrit directement dans le gif
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            gif_writers[filename].append_data(frame)
        
        plt.close()



# 2D Plots
def makeSolutions2DPlots(X, Y, q, do_exact_plot, q_exact, time, params, h, pdf_writers, gif_writers):
    plot_params = params["plot_parameters"]
    scheme_name, scheme_short = getSchemeKeys(params)
    simul_name, simul_short = getSimulKeys(params)
    do_pdf_plot = (plot_params["do_pdf_plot"] == "y")
    do_gif_plot = (plot_params["do_gif_plot"] == "y")
    CFL = params["time_parameters"]["CFL_number"]

    numLevels = plot_params["levels"]

    plots = [
        (XVEL, "U", "Vitesse U", "RdBu_r"),
        (YVEL, "V", "Vitesse V", "RdBu_r"),
        (PRES, "P", "Pression",  "viridis"),
    ]

    plots = [
        (t[0], f"{simul_short}{scheme_short}{t[1]}_h{round(h, 5)}_CFL{CFL}" ) + t[2:]
        for t in plots
    ]

    # FIGURES FOR APPROXIMATE SOLUTIONS
    for var, filename, title, cmap in plots:
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, q[:, :, var], levels=numLevels, cmap=cmap)
        ax.contour(X, Y, q[:, :, var], levels=numLevels, colors="k", linewidths=0.3)
        # plt.colorbar(cf, ax=ax, label=title, format="%.2f")
        plt.colorbar(cf, ax=ax, label=title)
        ax.set_title(f"{simul_name} avec {scheme_name} : {title} à t={round(time, 1)} \n et h = {round(h, 5)}, CFL = {CFL}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.tight_layout()

        if (do_pdf_plot) :
            pdf_writers[filename].savefig(fig)
        
        if (do_gif_plot) :
            # On écrit directement dans le gif
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            gif_writers[filename].append_data(frame)
        
        plt.close()

    

    # FIGURES FOR EXACT SOLUTIONS
    if (do_exact_plot) :
        exact_plots = [
            (XVEL, "U_exact", "Vitesse U exacte", "RdBu_r"),
            (YVEL, "V_exact", "Vitesse V exacte", "RdBu_r"),
            (PRES, "P_exact", "Pression exacte",  "viridis"),
        ]

        exact_plots = [
            (t[0], simul_short + t[1] + "_h"+str(round(h, 5))) + t[2:]
            for t in exact_plots
        ]

        for var, filename_exact, title, cmap in exact_plots:
            fig, ax = plt.subplots()
            cf = ax.contourf(X, Y, q_exact[:, :, var], levels=numLevels, cmap=cmap)
            ax.contour(X, Y, q_exact[:, :, var], levels=numLevels, colors="k", linewidths=0.3)
            plt.colorbar(cf, ax=ax, label=title, format="%.2f")
            ax.set_title(f"{simul_name} : {title} à t={round(time, 2)} \n et h = {round(h, 5)}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            plt.tight_layout()

            if (do_pdf_plot) :
                pdf_writers[filename_exact].savefig(fig)
            
            if (do_gif_plot) :
                # On écrit directement dans le gif
                fig.canvas.draw()
                frame = np.array(fig.canvas.renderer.buffer_rgba())
                gif_writers[filename_exact].append_data(frame)

            plt.close()



def makeSolutions2DDiffNormPlots(X, Y, q, q_exact, time, params, h, pdf_writers, gif_writers):
    plot_params = params["plot_parameters"]
    simul_name, simul_short = getSimulKeys(params)
    scheme_name, scheme_short = getSchemeKeys(params)
    do_pdf_plot = (plot_params["do_pdf_plot"] == "y")
    do_gif_plot = (plot_params["do_gif_plot"] == "y")
    CFL = params["time_parameters"]["CFL_number"]

    numLevels = plot_params["levels"]



    plots = [
        (XVEL, "|U-Uex|", "|U-Uex|", "RdBu_r"),
        (YVEL, "|V-Vex|", "|V-Vex|", "RdBu_r"),
        (PRES, "|P-Pex|", "|P-Pex|",  "viridis"),
    ]

    plots = [
        (t[0], f"{simul_short}{scheme_short}{t[1]}_h{round(h, 5)}_CFL{CFL}") + t[2:]
        for t in plots
    ]

    ecart = abs(q - q_exact)

    log_ecart = -16 * np.ones((np.shape(q)))
    log_ecart[ecart > 1.e-16] = np.log(ecart[ecart > 1.e-16]) / np.log(10.)
    plots = [
        (t[0], t[1], f"log({t[2]})") + t[3:]
        for t in plots
    ]

    # FIGURES FOR APPROXIMATE SOLUTIONS
    for var, filename, title, cmap in plots:
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, log_ecart[:, :, var], levels=numLevels, cmap=cmap)
        ax.contour(X, Y, log_ecart[:, :, var], levels=numLevels, colors="k", linewidths=0.3)
        # cf = ax.contourf(X, Y, ecart[:, :, var], levels=numLevels, cmap=cmap)
        # ax.contour(X, Y, ecart[:, :, var], levels=numLevels, colors="k", linewidths=0.3)
        # plt.colorbar(cf, ax=ax, label=title, format="%.2f")
        plt.colorbar(cf, ax=ax, label=title, format="%.2f")
        ax.set_title(f"{simul_name} : avec {scheme_name} \n {title} à t={round(time, 3)} et h = {round(h, 5)}, CFL = {CFL} \n")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.tight_layout()

        if (do_pdf_plot) :
            pdf_writers[filename].savefig(fig)
        
        if (do_gif_plot) :
            # On écrit directement dans le gif
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            gif_writers[filename].append_data(frame)
        
        plt.close()


# 2D Plots of || U - U_eq || (perturbed vortex)
def perturbedSolution2DPlots(X, Y, q, q_prep, time, params, h, pdf_writers, gif_writers):
    plot_params = params["plot_parameters"]
    simul_name, simul_short = getSimulKeys(params)
    scheme_name, scheme_short = getSchemeKeys(params)
    do_pdf_plot = (plot_params["do_pdf_plot"] == "y")
    do_gif_plot = (plot_params["do_gif_plot"] == "y")
    CFL = params["time_parameters"]["CFL_number"]

    numLevels = plot_params["levels"]
    
    ecart = np.linalg.norm(q[:, :, :PRES] - q_prep[:, :, :PRES], ord=2, axis=2)

    filename = simul_short + scheme_short + "_h"+str(round(h, 5)) +"_CFL"+str(CFL)

    # FIGURES 
    fig, ax = plt.subplots()
    cf = ax.contourf(X, Y, ecart, levels=numLevels, cmap="viridis")
    ax.contour(X, Y, ecart, levels=numLevels, colors="k", linewidths=0.3)
    # plt.colorbar(cf, ax=ax, label=title, format="%.2f")
    plt.colorbar(cf, ax=ax, label=f"$||\\mathbf{{U}} - \\mathbf{{U_eq}}||$")
    ax.set_title(f"{simul_name} avec {scheme_name} : $||\\mathbf{{U}} - \\mathbf{{U_{{eq}}}}||$ à t={round(time, 2)} \n et h = {round(h, 5)}, CFL = {CFL}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.tight_layout()

    if (do_pdf_plot) :
        pdf_writers[filename].savefig(fig)
    
    if (do_gif_plot) :
        # On écrit directement dans le gif
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        gif_writers[filename].append_data(frame)
    
    plt.close()




# AUXILIARY ROUTINES 



# GIF Open and Close
def openGifWriters(do_obs_table, params):
    plot_loc = params["plot_parameters"]["plot_loc"]
    gif_time = 1000. * params["plot_parameters"]["gif_time"] # Conversion s to ms !!
    simulationChoice = params["simulation_choice"]
    simul_short = getSimulKeys(params)[1]
    scheme_short = getSchemeKeys(params)[1]

    CFL = params["time_parameters"]["CFL_number"]
    Nx, Ny, nGhost = params["grid_parameters"]["mesh_parameters"].values()
    xL, xR, yL, yR = params["grid_parameters"]["domain_parameters"].values()
    h = max( (xR - xL) / (1. * Nx), (yR - yL) / (1. * Ny) )

    gif_writers_1D = {}
    gif_writers_2D = {}
    gif_writers_2D_diff = {}

    if do_obs_table[1] :
        if simulationChoice == 5 :
            filename =  simul_short + scheme_short + "_1D" +"_h"+str(round(h, 5)) +"_CFL"+str(CFL)
            gif_writers_1D = \
                {filename: imageio.get_writer(
                    f"{plot_loc}{filename}.gif",
                    mode="I",
                    duration=gif_time,
                    loop=0
                )}
        else :
            plots = [("U", ), ("V", ), ("P", ), ("U_exact", ), ("V_exact", ), ("P_exact", )]
            plots = [(f"{simul_short}{scheme_short}{var}_1D_h{round(h, 5)}_CFL{CFL}",) for (var,) in plots[:3]] + plots[3:]
            plots = plots[3:] + [(f"{simul_short}{var}_1D_h{round(h, 5)}",) for (var,) in plots[3:]]

            gif_writers_2D = \
            {
                filename: imageio.get_writer(
                    f"{plot_loc}{filename}.gif",
                    mode="I",
                    duration=gif_time,
                    loop=0
                )
                for filename, *_ in plots
            }
    
    if do_obs_table[2] : 
        if simulationChoice == 5 :
            filename =  simul_short + scheme_short +"_h"+str(round(h, 5)) +"_CFL"+str(CFL)
            gif_writers_2D = \
                {filename: imageio.get_writer(
                    f"{plot_loc}{filename}.gif",
                    mode="I",
                    duration=gif_time,
                    loop=0
                )}
        else :
            plots = [("U", ), ("V", ), ("P", ), ("U_exact", ), ("V_exact", ), ("P_exact", )]
            # plots = [(system_short + simul_short + var + "_h"+str(round(h, 5)),) for (var,) in plots]
            # plots = [(var +"_CFL"+str(CFL),) for (var,) in plots[:3]] + plots[3:]

            plots = [(f"{simul_short}{scheme_short}{var}_h{round(h, 5)}_CFL{CFL}",) for (var,) in plots[:3]] + plots[3:]
            plots = plots[3:] + [(f"{simul_short}{var}_h{round(h, 5)}",) for (var,) in plots[3:]]

            gif_writers_2D = \
            {
                filename: imageio.get_writer(
                    f"{plot_loc}{filename}.gif",
                    mode="I",
                    duration=gif_time,
                    loop=0
                )
                for filename, *_ in plots
            }
    
    if do_obs_table[3] :
        plots = [("|U-Uex|", ), ("|V-Vex|", ), ("|P-Pex|", )]
        plots = [(f"{simul_short}{scheme_short}{var}_h{round(h, 5)}_CFL{CFL}",) for (var,) in plots]

        gif_writers_2D_diff = \
        {
            filename: imageio.get_writer(
                f"{plot_loc}{filename}.gif",
                mode="I",
                duration=gif_time,
                loop=0
            )
            for filename, *_ in plots
        }
    
    return {"1D" : gif_writers_1D, "2D" : gif_writers_2D, "2D_diff" : gif_writers_2D_diff}


def closeGifWriters(gif_writers):
    for writer in gif_writers["1D"].values():
        writer.close()
    for writer in gif_writers["2D"].values():
        writer.close()
    for writer in gif_writers["2D_diff"].values():
        writer.close()
    print("GIFs sauvegardés !")



# PDF open and close
def openPdfWriters(do_obs_table, params):
    plot_loc = params["plot_parameters"]["plot_loc"]
    simulationChoice = params["simulation_choice"]
    simul_short = getSimulKeys(params)[1]
    scheme_short = getSchemeKeys(params)[1]

    CFL = params["time_parameters"]["CFL_number"]
    Nx, Ny, nGhost = params["grid_parameters"]["mesh_parameters"].values()
    xL, xR, yL, yR = params["grid_parameters"]["domain_parameters"].values()
    h = max( (xR - xL) / (1. * Nx), (yR - yL) / (1. * Ny) )

    pdf_writers_1D = {}
    pdf_writers_2D = {}
    pdf_writers_2D_diff = {}

    if do_obs_table[1] :
        if simulationChoice == 5 :
            filename = simul_short + scheme_short + "_1D_h"+str(round(h, 5)) +"_CFL"+str(CFL)
            pdf_writers_1D = {filename : PdfPages(f"{plot_loc}{filename}.pdf")}
        else :
            plots = ["U", "V", "P", "U_exact", "V_exact", "P_exact"]
            plots = [f"{simul_short}{scheme_short}{var}_1D_h{round(h, 5)}_CFL{CFL}" for var in plots[:3]] + plots[3:]
            plots = plots[:3] + [f"{simul_short}{var}_1D_h{round(h, 5)}" for var in plots[3:]]

            pdf_writers_1D = \
            {
                filename: PdfPages(f"{plot_loc}{filename}.pdf")
                for filename in plots
            }

    if do_obs_table[2] :
        if simulationChoice == 4 or simulationChoice == 5 :
            filename = simul_short + scheme_short + "_h"+str(round(h, 5)) +"_CFL"+str(CFL)
            pdf_writers_2D = {filename : PdfPages(f"{plot_loc}{filename}.pdf")}
        else :
            plots = ["U", "V", "P", "U_exact", "V_exact", "P_exact"]
            plots = [f"{simul_short}{scheme_short}{var}_h{round(h, 5)}_CFL{CFL}" for var in plots[:3]] + plots[3:]
            plots = plots[:3] + [f"{simul_short}{var}_h{round(h, 5)}" for var in plots[3:]]

            pdf_writers_2D = \
            {
                filename: PdfPages(f"{plot_loc}{filename}.pdf")
                for filename in plots
            }
    
    if do_obs_table[3] :
        plots = ["|U-Uex|", "|V-Vex|", "|P-Pex|"]
        plots = [f"{simul_short}{scheme_short}{var}_h{round(h, 5)}_CFL{CFL}" for var in plots]

        pdf_writers_2D_diff = \
        {
            filename: PdfPages(f"{plot_loc}{filename}.pdf")
            for filename in plots
        }
    
    return {"1D" : pdf_writers_1D, "2D" : pdf_writers_2D, "2D_diff" : pdf_writers_2D_diff}




def closePdfWriters(pdf_writers):
    for writer in pdf_writers["1D"].values():
        writer.close()
    for writer in pdf_writers["2D"].values():
        writer.close()
    for writer in pdf_writers["2D_diff"].values():
        writer.close()
    print("PDFs sauvegardés !")



####################################################################################################################

#                                              ERRORS AND NORMS PLOTS                                              #

####################################################################################################################



def makeObservableTablePlots(i_obs, iterations_of_obs_table, observableTable, params, h):
    plot_loc = params["plot_parameters"]["plot_loc"]
    simul_name, simul_short = getSimulKeys(params)
    scheme_name, scheme_short = getSchemeKeys(params)
    same_plot = params["plot_parameters"]["same_plot"]
    finalTime, CFL = params["time_parameters"].values()

    # if (i_obs == [4]): # norms 
    #     firstIteration = 0
    #     lastIteration = lastCompletedIteration
    # elif (i_obs == [5]): # errors : no computation for initial datum
    #     firstIteration = 1
    #     lastIteration = lastCompletedIteration
    # elif (i_obs == [6]): # total masses
    #     firstIteration = 0
    #     lastIteration = lastCompletedIteration
    # elif (i_obs == [7]): # divU
    #     firstIteration = 0
    #     lastIteration = lastCompletedIteration

    obs_name, obs_short = getObsKeys(i_obs)

    obsNumber = len(iterations_of_obs_table)
    # iterations_of_obs_table = np.arange(firstIteration, lastIteration+1)
    # observableTable = observableTable[firstIteration : lastIteration+1]

    if i_obs == [7] :
        with PdfPages(f"{plot_loc}{simul_short}{scheme_short}{obs_short}h{round(h, 5)}_CFL{CFL}.pdf") as pdf:
            obs_name = "Norme des termes physiques"
            obs_label = "physcial terms norm"
            plt.figure(1)
            fig, ax = plt.subplots()
            if abs(np.max(observableTable) / max(np.min(observableTable), 1.e-16)) \
                        > 100 :
                        plt.yscale("log")
            ax.set_title(f"{simul_name} avec {scheme_name} \n {obs_name} en fonction " \
                    f"de l'itération \n Tf = {finalTime}, CFL = {CFL} et h = {round(h, 5)}")
            ax.plot(iterations_of_obs_table, observableTable[:, 0], label=f"${{|| \\widetilde{{\\mathsf{{grad}}}} \\, P ||}}_{{\\infty}}$")
            ax.plot(iterations_of_obs_table, observableTable[:, 1], label=f"${{|| \\widetilde{{\\mathsf{{DIV}}}} \\, U ||}}_{{\\infty}}$")
            ax.set_xlabel("n")
            ax.set_ylabel(obs_label)
            ax.legend()
            fig.tight_layout()  # Ajuste automatiquement les marges
            pdf.savefig(fig)
            plt.close(fig)


    else :
        plt.figure(3)
        var = [(XVEL, "u"), (YVEL, "v"), (PRES, "p")]
        with PdfPages(f"{plot_loc}{simul_short}{scheme_short}{obs_short}h{round(h, 5)}_CFL{CFL}.pdf") as pdf:
            if (same_plot == "y"):
                fig, ax = plt.subplots()
                if abs(np.max(observableTable) / max(np.min(observableTable), 1.e-16)) \
                    > 100 :
                    ax.set_yscale("log")

                if (i_obs == [4]):
                    u_label = f"$|| u_{{n}} ||_{{\\infty}}$"
                    v_label = f"$|| v_{{n}} ||_{{\\infty}}$"
                    p_label = f"$|| p_{{n}} ||_{{\\infty}}$"
                if (i_obs == [5]):
                    u_label = f"$|| u_{{n}} - u_{{\\text{{ex}}, n}} ||_{{\\infty}}$"
                    v_label = f"$|| v_{{n}} - v_{{\\text{{ex}}, n}} ||_{{\\infty}}$"
                    p_label = f"$|| p_{{n}} - p_{{\\text{{ex}}, n}} ||_{{\\infty}}$"
                if (i_obs == [6]):
                    u_label = f"$\\int_{{\\Omega}} u \\, dX$"
                    v_label = f"$\\int_{{\\Omega}} v \\, dX$"
                    p_label = f"$\\int_{{\\Omega}} p \\, dX$"

                ax.set_title(f"{simul_name} avec {scheme_name} \n {obs_name} des composantes de q fonction " \
                                f"de l'itération \n Tf = {finalTime}, CFL = {CFL} et h = {round(h, 5)}")
                ax.plot(iterations_of_obs_table, observableTable[:, 0], label=u_label, marker="o", alpha = 0.5, markevery=obsNumber//10)
                ax.plot(iterations_of_obs_table, observableTable[:, 1], label=v_label, marker="^", alpha = 0.5, markevery=obsNumber//10)
                ax.plot(iterations_of_obs_table, observableTable[:, 2], label=p_label, marker="v", alpha = 0.5, markevery=obsNumber//10)
                ax.set_xlabel("n")
                # ax.set_xscale("log")
                ax.legend()
                fig.tight_layout()  # Ajuste automatiquement les marges
                pdf.savefig(fig)
                plt.close(fig)
            else :
                for k, varName in var:
                    fig, ax = plt.subplots()
                    if abs(np.max(observableTable[:, k]) / max(np.min(observableTable[:, k]), 1.e-16)) \
                        > 100 :
                        ax.set_yscale("log")

                    if (i_obs == [4]):
                        obs_label = f"$|| {varName}_{{n}} ||_{{\\infty}}$"
                    if (i_obs == [5]):
                        obs_label = f"$|| {varName}_{{n}} - {varName}_{{\\text{{ex}}, n}} ||_{{\\infty}}$"
                    if (i_obs == [6]):
                        obs_label = f"$\\int_{{\\Omega}} {varName} \\, dX$"

                    ax.plot(iterations_of_obs_table, observableTable[:, k])
                    ax.set_title(f"{simul_name} avec {scheme_name} \n {obs_name} de {varName} en fonction " \
                                    f"de l'itération \n Tf = {finalTime}, CFL = {CFL} et h = {round(h, 5)}")
                    ax.set_xlabel("n")
                    ax.set_ylabel(obs_label)
                    fig.tight_layout()  # Ajuste automatiquement les marges
                    pdf.savefig(fig)
                    plt.close(fig)



def makeConvTestPlots(hList, supErrorsList, L2ErrorsList, params):
    plot_loc = params["plot_parameters"]["plot_loc"]
    scheme_name, scheme_short = getSchemeKeys(params)
    simul_name, simul_short = getSimulKeys(params)
    finalTime, CFL = params["time_parameters"].values()

    obs_name, obs_short = getObsKeys([7])

    plt.figure(3)
    var = [(XVEL, "u"), (YVEL, "v"), (PRES, "p")]
    with PdfPages(f"{plot_loc}{simul_short}{scheme_short}{obs_short}{CFL}.pdf") as pdf:
        for k, varName in var:
            # Regression de pente 1 pour les log des deux jeux de données :
            coeffSup = np.exp(np.sum(np.log(supErrorsList[:, k]) - np.log(hList)) / len(hList))
            coeffL2 = np.exp(np.sum(np.log(L2ErrorsList[:, k]) - np.log(hList)) / len(hList))

            # Plots
            fig, ax = plt.subplots()
            ax.set_title(f"{simul_name} avec {scheme_name} : Erreurs sur {varName} " \
                            f"\n en fct du pas d'espace avec Tf = {finalTime}, CFL = {CFL}")
            ax.plot(hList, supErrorsList[:, k], label="Erreur $L^{{\\infty}}$", color="blue")
            ax.plot(hList, coeffSup * hList, label="pente 1", color="blue", linestyle="--", alpha=0.5)
            ax.plot(hList, L2ErrorsList[:, k], label="Erreur $L^2$", color="orange")
            ax.plot(hList, coeffL2 * hList, label="pente 1", color="orange",  linestyle="--", alpha=0.5)
            ax.set_xscale("log")
            ax.set_xlabel("h")
            ax.set_yscale("log")
            ax.set_ylabel(f"$|| {varName} - {varName}_{{\\text{{ex}}}} ||$")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)



####################################################################################################################

#                                              HIGH ORDER VISUALISATION                                            #

####################################################################################################################



# GET LAGRANGIAN REF COEFFICIENTS :
# returns the matrix (a_ij) with 
# Li = a_i0 + a_i1 X + ... + a_iK X^K
def getRefEvenlySpacedLagInterpPolynCoefs(N):
    X = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            X[i, j] = ((1. * j) / (1. * N))**i

    coeffMat = np.linalg.inv(X)
    return coeffMat



def getEvaluationOfBasisPolynomials(N, nVis):

    # 1 : CALCUL DES COEFFICIENTS DES POLYNOMES SUR LA CELLULE DE REF [0, 1]
    coeffMat = getRefEvenlySpacedLagInterpPolynCoefs(N)

    # 2 : CALCUL DE LA MATRICE DE VANDERMONDE DES POINTS DE VISUALISATIONS
    coordVis = np.linspace(0, 1, nVis, endpoint=False)
    vanDerMat = np.ones((N+1, nVis))
    for i in range(1, N+1):
        vanDerMat[i, :] = coordVis**i

    # 3 : MATRICE DES EVALUATIONS DES POLYNOMES EN CES POINTS :
    evalMat = coeffMat @ vanDerMat

    return evalMat



def getVisualisationData(q, params):
    Nx, Ny, nGhost = params["grid_parameters"]["mesh_parameters"].values()
    xL, xR, yL, yR = params["grid_parameters"]["domain_parameters"].values()
    order = params["order"]
    nVis = params["number_visualisation_points"]

    shape = np.shape(q)
    # print("shape of q : ", np.shape(q))
    if len(shape) == 4:
        qVis = np.zeros((Nx * nVis + 1, Ny * nVis + 1))
    else :
        nVar = np.shape(q)[-1]
        qVis = np.zeros((Nx * nVis + 1, Ny * nVis + 1, nVar))

    # print("shape of qVis : ", np.shape(qVis))


    if nVis == order : # LA GRILLE DE VISUALISATION COÏNCIDE AVEC CELLE D'ORDRE ELEVE : ON APPLATIT JUSTE LES DONNES DU MAILLAGE VALIDE
        for i in range(Nx):
          for j in range(Ny):
            for k in range(order):
              for l in range(order):
                qVis[i * order + k, j * order + l] = q[i, j, k, l]
    
        for j in range(Ny):
            qVis[Nx * order, j*order:(j+1)*order] = q[Nx, j, 0, :]
    
        for i in range(Nx):
            qVis[i*order:(i+1)*order, Ny * order] = q[i, Ny, :, 0]
    
        qVis[Nx * order, Ny * order] = q[Nx, Ny, 0, 0]


    else : # LA GRILLE DE VISUALISATION NE COÏNCIDE PAS AVEC CELLE D'ORDRE ELEVE : ON DOIT INTERPOLER
        # 2 : PROJECTION DES DONNEES SUR LA GRILLE CELLULE PAR CELLULE
        ### A : CALCUL DE LA MATRICE DES EVALUATION DES POLYNOMES DE BASE
        evalMat = getEvaluationOfBasisPolynomials(order, nVis)
        evalMat_transpose = np.transpose(evalMat)

        ### B : INTERPOLATIONS DE LA SOLUTION SUR [xL, xR[ x [yL, yR[
        q_cell_ij = np.zeros((order+1, order+1, 3))
        for i in range(Nx):
            for j in range(Ny):
                q_cell_ij[:-1, :-1] = q[i, j]
                q_cell_ij[order, :-1] = q[i + 1, j, 0, :]
                q_cell_ij[:-1, order] = q[i, j + 1, :, 0]
                q_cell_ij[order, order] = q[i + 1, j + 1, 0, 0]

                for var in range(3):
                    qVis[i * nVis : (i+1) * nVis, j * nVis : (j+1) * nVis, var] = evalMat_transpose @ q_cell_ij[:, :, var] @ evalMat


        # ### C : VALEURS SUR LE BORD RESTANT
        # for j in range(Ny):
        #     q_cell_ij[:-1, :-1] = q[iMax+1, j]
        #     q_cell_ij[order, :-1] = q[iMax+1+1, j, 0, :]
        #     q_cell_ij[:-1, order] = q[i, j+1, :, 0]
        #     q_cell_ij[order, order] = q[i+1, j+1, 0, 0]

        #     for var in range(3):
        #         val = evalMat_transpose @ q_cell_ij[:, :, var] @ evalMat
        #         qVis[Nx * nVis, j * nVis : (j+1) * nVis, :] = val[0, :]

        # for i in range(Nx):
        #     q_cell_ij[:-1, :-1] = q[i, j]
        #     q_cell_ij[order, :-1] = q[i+1, j, 0, :]
        #     q_cell_ij[:-1, order] = q[i, j+1, :, 0]
        #     q_cell_ij[order, order] = q[i+1, j+1, 0, 0]

        #     for var in range(3):
        #         val = evalMat_transpose @ q_cell_ij[:, :, var] @ evalMat
        #         qVis[i * nVis : (i+1) * nVis, Ny * nVis, :] = val[:, 0]


        # CL PERIODIQUES :
        qVis[Nx * nVis, :] = qVis[0, :]
        qVis[:, Ny * nVis] = qVis[:, 0]


    return qVis
