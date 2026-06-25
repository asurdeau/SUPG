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
    systemChoice = params["system_choice"]

    if systemChoice == 1 : # 2D ACOUSTIC SIMULATIONS
        simulationChoice = params["solution_parameters"]["simulation_choice"]

        if (simulationChoice == 1):
            simul_name =  "etat constant"
            simul_short = "cst_"
        elif (simulationChoice == 2):
            simul_name =  "random noise"
            simul_short = "noise_"
        elif (simulationChoice == 3):
            simul_name =  "cst + small gaussian"
            simul_short = "cst+pert_"
        elif (simulationChoice == 4):
            simul_name =  "checkerboard"
            simul_short = "check_"
        elif (simulationChoice == 5):
            degree = params["solution_parameters"]["analytical_periodic"]["theta"]
            simul_name =  "smooth oblique flow ("+str(round(degree, 3))+"°)"
            simul_short = "smooth_flow_"+str(round(degree, 3))+"_"
        elif (simulationChoice == 6):
            simul_name =  "stationary vortex"
            simul_short = "vortex_"
        elif (simulationChoice == 7):
            simul_name =  "stationary vortex + small gaussian perturbation"
            simul_short = "vortex+pert_"
    
    return simul_name, simul_short
    

def getObsKeys(obsChoice):
    if (obsChoice == 1):
        obs_name = "figure(s) 2D"
        obs_short = ""
    elif (obsChoice == 2):
        obs_name = "figure(s) 1D"
        obs_short = ""
    elif (obsChoice == 3):
        obs_name = "Norme sup"
        obs_short = "norm_sup_"
    elif (obsChoice == 4):
        obs_name = "Erreur sup"
        obs_short = "err_sup_"
    elif (obsChoice == 5):
        obs_name = "test de convergence"
        obs_short = ""
    
    return obs_name, obs_short
    


def choicePrints(params):
    # getting parameters to print
    scheme_name = getSchemeKeys(params)[0]
    simulation_name = getSimulKeys(params)[0]
    obs_name = getObsKeys(params["observables"])[0]
    plotLoc = params["plot_parameters"]["plot_loc"]
    CFL = params["time_parameters"]["CFL_number"]

    # prints
    print("")
    print("===================================================================")
    print("")
    print("             Schéma testé : "+scheme_name                           )
    print("      Choix de simulation : "+simulation_name                       )
    print("                      CFL : "+str(CFL)                              )
    print("        Observable testée : "+obs_name                              )
    print("Emplacement des résultats : "+plotLoc                               )
    print("")
    print("===================================================================")
    print("")


####################################################################################################################

#                                                 SOLUTIONS PLOTS                                                  #

####################################################################################################################



def makeSolutionsPlots(q, time, params, grid, pdf_writers, gif_writers):
    i_obs = params["observables"]
    simulationChoice = params["solution_parameters"]["simulation_choice"]
    if (i_obs == 1):
        if (simulationChoice == 7):
            perturbedVortex2DPlots(q, time, params, grid, pdf_writers, gif_writers)
        else :
            makeSolutions2DPlots(q, time, params, grid, pdf_writers, gif_writers)
    if (i_obs == 2):
        makeSolutions1DPlots(q, time, params, grid, pdf_writers, gif_writers)



# 1D Solutions
def makeSolutions1DPlots(q, time, params, grid, pdf_writers, gif_writers):
    plot_params = params["plot_parameters"]
    simulChoice = params["solution_parameters"]["simulation_choice"]
    scheme_name, scheme_short = getSchemeKeys(params)
    simul_name, simul_short = getSimulKeys(params)
    do_pdf_plot = (plot_params["do_pdf_plot"] == "y")
    do_gif_plot = (plot_params["do_gif_plot"] == "y")
    h = max( grid.steps )
    CFL = params["time_parameters"]["CFL_number"]

    i_section = plot_params["section"]

    # On ne prend en compte une éventuelle solution exacte que si on la connait 
    # ET que c'est pertinent (pas les vortex stat. après t = 0 par ex)
    do_exact_pdf_plot = False
    do_exact_gif_plot = False
    if (simulChoice == 5 or (simulChoice == 6 and time < 1.e-14)):
        do_exact_pdf_plot = (plot_params["do_exact_pdf_plot"] == "y")
        do_exact_gif_plot = (plot_params["do_exact_gif_plot"] == "y")

    iMin, iMax, jMin, jMax = grid.valid_grid
    X = grid.xGrid[iMin:iMax+1, jMin:jMax+1]
    Y = grid.yGrid[iMin:iMax+1, jMin:jMax+1]

    Nx = iMax - iMin + 1
    Ny = jMax - jMin + 1

    if (i_section == 1):
        jMid = int( 0.5 * Ny )
        domaine_silce = X[:, jMid]
        q_slice = q[:, jMid]
        slice_name = "coupe horizontale"

        if (do_exact_pdf_plot or do_exact_gif_plot):
            q_exact = solutions.getSolution(time, grid, params["solution_parameters"])
            q_exact_slice = q_exact[:, jMid]
    
    elif (i_section == 2):
        iMid = int( 0.5 * Nx )
        domaine_silce = Y[iMid]
        q_slice = q[iMid]
        slice_name = "coupe verticale"

        if (do_exact_pdf_plot or do_exact_gif_plot):
            q_exact = solutions.getSolution(time, grid, params["solution_parameters"])
            q_exact_slice = q_exact[iMid]

    elif (i_section == 3):
        if (Nx != Ny):
            print("section diagonale mais malliage non adapté")
        else :
            slice_name = "coupe oblique"

            iMid = int( 0.5 * Nx )
            jMid = int( 0.5 * Ny )
            domaine_silce = np.sqrt( X[:, 0]**2 + Y[0, :]**2 )

            q_slice = np.diagonal(q, axis1=0, axis2=1).T
            
            if (do_exact_pdf_plot or do_exact_gif_plot):
                q_exact = solutions.getSolution(time, grid, params["solution_parameters"])
                q_exact_slice = np.diagonal(q_exact, axis1=0, axis2=1).T
                
    plots = [
        (XVEL, "U", "u approchée", "u exacte", "u"),
        (YVEL, "V", "v approchée", "v exacte", "v"),
        (PRES, "P", "p approchée", "p exacte", "p"),
    ]

    plots = [
        (t[0], simul_short + scheme_short + t[1] + "_h"+str(round(h, 5)) +"_CFL"+str(CFL) ) + t[2:]
        for t in plots
    ]

    # FIGURES FOR APPROXIMATE SOLUTIONS
    for var, filename, approx_label, exact_label, var_symbol in plots:
        fig, ax = plt.subplots()
        ax.plot(domaine_silce, q_slice[:, var], label=approx_label)
        if (do_exact_pdf_plot or do_exact_gif_plot):
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
def makeSolutions2DPlots(q, time, params, grid, pdf_writers, gif_writers):
    plot_params = params["plot_parameters"]
    simulChoice = params["solution_parameters"]["simulation_choice"]
    scheme_name, scheme_short = getSchemeKeys(params)
    simul_name, simul_short = getSimulKeys(params)
    do_pdf_plot = (plot_params["do_pdf_plot"] == "y")
    do_gif_plot = (plot_params["do_gif_plot"] == "y")
    h = max( grid.steps )
    CFL = params["time_parameters"]["CFL_number"]

    numLevels = plot_params["levels"]

    # On ne prend en compte une éventuelle solution exacte que si on la connait 
    # ET que c'est pertinent (pas les vortex stat. après t = 0 par ex)
    do_exact_pdf_plot = False
    do_exact_gif_plot = False
    if (simulChoice == 5 or (simulChoice == 6 and time < 1.e-14)):
        do_exact_pdf_plot = (plot_params["do_exact_pdf_plot"] == "y")
        do_exact_gif_plot = (plot_params["do_exact_gif_plot"] == "y")

    X = grid.xValidGrid
    Y = grid.yValidGrid

    plots = [
        (XVEL, "U", "Vitesse U", "RdBu_r"),
        (YVEL, "V", "Vitesse V", "RdBu_r"),
        (PRES, "P", "Pression",  "viridis"),
    ]

    plots = [
        (t[0], simul_short + scheme_short + t[1] + "_h"+str(round(h, 5)) +"_CFL"+str(CFL) ) + t[2:]
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
    if (do_exact_pdf_plot or do_exact_gif_plot) :
        exact_plots = [
            (XVEL, "U_exact", "Vitesse U exacte", "RdBu_r"),
            (YVEL, "V_exact", "Vitesse V exacte", "RdBu_r"),
            (PRES, "P_exact", "Pression exacte",  "viridis"),
        ]

        exact_plots = [
            (t[0], simul_short + t[1] + "_h"+str(round(h, 5))) + t[2:]
            for t in exact_plots
        ]

        q_exact = solutions.getSolution(time, grid, params["solution_parameters"])

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

            if (do_exact_pdf_plot) :
                pdf_writers[filename_exact].savefig(fig)
            
            if (do_exact_gif_plot) :
                # On écrit directement dans le gif
                fig.canvas.draw()
                frame = np.array(fig.canvas.renderer.buffer_rgba())
                gif_writers[filename_exact].append_data(frame)

            plt.close()




# 2D Plots of || U - U_eq || (perturbed vortex)
def perturbedVortex2DPlots(q, time, params, grid, pdf_writers, gif_writers):
    plot_params = params["plot_parameters"]
    scheme_short, scheme_name = getSchemeKeys(params)
    simul_name, simul_short = getSimulKeys(params)
    do_pdf_plot = (plot_params["do_pdf_plot"] == "y")
    do_gif_plot = (plot_params["do_gif_plot"] == "y")
    h = max( grid.steps )
    CFL = params["time_parameters"]["CFL_number"]

    numLevels = plot_params["levels"]

    X = grid.xValidGrid
    Y = grid.yValidGrid

    q_eq = solutions.getSolution(time, grid, params["solution_parameters"])
    grid.dirichlet(q_eq)
    
    ecart = np.linalg.norm(q[:, :, :PRES] - q_eq[:, :, :PRES], ord=2, axis=2)

    filename = simul_short + scheme_short + "diff_to_eq" + "_h"+str(round(h, 5)) +"_CFL"+str(CFL)

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
def openGifWriters(params):
    plot_loc = params["plot_parameters"]["plot_loc"]
    gif_time = 1000. * params["plot_parameters"]["gif_time"] # Conversion s to ms !!
    simulationChoice = params["solution_parameters"]["simulation_choice"]
    simul_short = getSimulKeys(params)[1]
    scheme_short = getSchemeKeys(params)[1]

    CFL = params["time_parameters"]["CFL_number"]
    Nx, Ny = params["grid_parameters"]["mesh_parameters"].values()[:2]
    xL, xR, yL, yR = params["grid_parameters"]["domain_parameters"].values()
    h = max( (xR - xL) / (1. * Nx), (yR - yL) / (1. * Ny) )

    if simulationChoice == 7 :
        filename = simul_short + scheme_short + "diff_to_eq" + "_h"+str(round(h, 5)) +"_CFL"+str(CFL)
        return {filename : PdfPages(f"{plot_loc}{filename}.pdf")}
    else :
        plots = [("U", ), ("V", ), ("P", ), ("U_exact", ), ("V_exact", ), ("P_exact", )]
        plots = [(simul_short + scheme_short + var + "_h"+str(round(h, 5)),) for (var,) in plots]
        plots = [(var +"_CFL"+str(CFL),) for (var,) in plots[:3]] + plots[3:]

        return {
            filename: imageio.get_writer(
                f"{plot_loc}{filename}.gif",
                mode="I",
                duration=gif_time,
                loop=0
            )
            for filename, *_ in plots
        }


def closeGifWriters(gif_writers):
    for writer in gif_writers.values():
        writer.close()
    print("GIFs sauvegardés !")



# PDF open and close
def openPdfWriters(params):
    plot_loc = params["plot_parameters"]["plot_loc"]
    simulationChoice = params["solution_parameters"]["simulation_choice"]
    simul_short = getSimulKeys(params)[1]
    scheme_short = getSchemeKeys(params)[1]

    CFL = params["time_parameters"]["CFL_number"]
    Nx, Ny = params["grid_parameters"]["mesh_parameters"].values()[:2]
    xL, xR, yL, yR = params["grid_parameters"]["domain_parameters"].values()
    h = max( (xR - xL) / (1. * Nx), (yR - yL) / (1. * Ny) )

    if simulationChoice == 7 :
        filename = simul_short + scheme_short + "diff_to_eq" + "_h"+str(round(h, 5)) +"_CFL"+str(CFL)
        return {filename+"_"+scheme_short : PdfPages(f"{plot_loc}{filename+"_"+scheme_short}.pdf")}
    else :
        plots = ["U", "V", "P", "U_exact", "V_exact", "P_exact"]
        plots = [simul_short + scheme_short + var+"_h"+str(round(h, 5)) for var in plots]
        plots = [var +"_CFL"+str(CFL) for var in plots[:3]] + plots[3:]

        return {
            filename: PdfPages(f"{plot_loc}{filename}.pdf")
            for filename in plots
        }


def closePdfWriters(pdf_writers):
    for writer in pdf_writers.values():
        writer.close()
    print("PDFs sauvegardés !")



####################################################################################################################

#                                              ERRORS AND NORMS PLOTS                                              #

####################################################################################################################



def makeNormPlots(norms, nb_iter, params, grid):
    plot_loc = params["plot_parameters"]["plot_loc"]
    scheme_short, scheme_name = getSchemeKeys(params)
    simul_name, simul_short = getSimulKeys(params)
    i_obs = params["observables"]
    finalTime, CFL = params["time_parameters"].values()
    h = max( grid.steps )

    obs_name, obs_short = getObsKeys(i_obs)

    plt.figure(3)
    var = [(XVEL, "u"), (YVEL, "v"), (PRES, "p")]
    with PdfPages(plot_loc + simul_short + scheme_short + obs_short + "_h" + str(round(h, 5)) + "_CFL" + str(CFL) + ".pdf") as pdf:
        for k, varName in var:
            if (i_obs == 3):
                obs_label = f"$|| {varName}_{{n}} ||$"
            if (i_obs == 4):
                obs_label = f"$|| {varName}_{{n}} - {varName}_{{\\text{{ex}}, n}} ||$"

            # fig, ax = plt.subplots(figsize=(5, 5))
            fig, ax = plt.subplots()
            ax.plot(np.arange(1, nb_iter+1), norms[:nb_iter, k])
            ax.set_title(f"{simul_name} avec {scheme_name} : {obs_name} de {varName} en fonction " \
                            f"\n de l'itération; pour Tf = {finalTime}, CFL = {CFL} et h = {round(h, 5)}")
            ax.set_xlabel("n")
            ax.set_yscale("log")
            ax.set_ylabel(obs_label)
            fig.tight_layout()  # Ajuste automatiquement les marges
            pdf.savefig(fig)
            plt.close(fig)



def makeConvTestPlots(hList, supErrorsList, L2ErrorsList, params):
    plot_loc = params["plot_parameters"]["plot_loc"]
    scheme_short, scheme_name = getSchemeKeys(params)
    simul_name, simul_short = getSimulKeys(params)
    finalTime, CFL = params["time_parameters"].values()

    plt.figure(3)
    var = [(XVEL, "u"), (YVEL, "v"), (PRES, "p")]
    with PdfPages(plot_loc + simul_short + scheme_short + "test_conv_" + ".pdf") as pdf:
        for k, varName in var:
            # Regression de pente 1 pour les log des deux jeux de données :
            coeffSup = np.exp(np.sum(np.log(supErrorsList[:, k]) - np.log(hList)) / len(hList))
            coeffL2 = np.exp(np.sum(np.log(L2ErrorsList[:, k]) - np.log(hList)) / len(hList))

            # Plots
            fig, ax = plt.subplots()
            ax.set_title(f"{simul_name} avec {scheme_name} : Erreurs sur {varName} en fonction " \
                            f"\n de l'itération; pour Tf = {finalTime}, CFL = {CFL}")
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