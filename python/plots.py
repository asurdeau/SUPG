# Modules généraux
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import imageio.v2 as imageio
import os

# Modules perso
from config import XVEL, YVEL, PRES 
import solutions

def getSchemeTitles(params):
    schemeChoice = params["scheme_choice"]

    if (schemeChoice == 1):
        schemeShort = "UPW"
        schemeName = "Upwind"
    elif (schemeChoice == 2):
        schemeShort = "SUPG"
        schemeName = "SUPG"
    elif (schemeChoice == 3):
        schemeShort = "modSUPG"
        schemeName = "SUPG modifié"
    
    return schemeShort, schemeName



def makeSolutionsPlots(q, time, params, grid, pdf_writers, gif_writers):
    i_obs = params["plot_parameters"]["observables"]
    if (i_obs == 1):
        makeSolutions2DPlots(q, time, params, grid, pdf_writers, gif_writers)
    if (i_obs == 2):
        makeSolutions1DPlots(q, time, params, grid, pdf_writers, gif_writers)



# 1D Solutions
def makeSolutions1DPlots(q, time, params, grid, pdf_writers, gif_writers):
    plot_params = params["plot_parameters"]["sol_plots"]
    do_pdf_plot = (plot_params["do_pdf_plot"] == "y")
    do_gif_plot = (plot_params["do_gif_plot"] == "y")

    i_section = plot_params["section"]

    # On ne prend en compte une éventuelle solution exacte que si on la connait 
    do_exact_pdf_plot = False
    do_exact_gif_plot = False
    if (params["solution_parameters"]["simulation_choice"] == 5):
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

        if (do_exact_pdf_plot or do_exact_gif_plot):
            q_exact = solutions.getSolution(time, grid, params["solution_parameters"])
            q_exact_slice = q_exact[:, jMid]
    
    elif (i_section == 2):
        iMid = int( 0.5 * Nx )
        domaine_silce = Y[iMid]
        q_slice = q[iMid]

        if (do_exact_pdf_plot or do_exact_gif_plot):
            q_exact = solutions.getSolution(time, grid, params["solution_parameters"])
            q_exact_slice = q_exact[iMid]

    elif (i_section == 3):
        if (Nx != Ny):
            print("section diagonale mais malliage non adapté")
        else :
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

    schemeShort, schemeName = getSchemeTitles(params)

    # FIGURES FOR APPROXIMATE SOLUTIONS
    for var, filename, approx_label, exact_label, var_symbol in plots:
        fig, ax = plt.subplots()
        ax.plot(domaine_silce, q_slice[:, var], label=approx_label)
        if (do_exact_pdf_plot or do_exact_gif_plot):
            ax.plot(domaine_silce, q_exact_slice[:, var], label=exact_label)
        ax.set_title(f"{var_symbol} avec le schéma "+schemeName+f" à t={round(time, 1)}")
        ax.set_xlabel("x")
        ax.set_ylabel(var_symbol)
        ax.legend()
        plt.tight_layout()

        if (do_pdf_plot) :
            pdf_writers[filename+"_"+schemeShort].savefig(fig)
        
        if (do_gif_plot) :
            # On écrit directement dans le gif
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            gif_writers[filename+"_"+schemeShort].append_data(frame)
        
        plt.close()


# 2D Plots
def makeSolutions2DPlots(q, time, params, grid, pdf_writers, gif_writers):
    plot_params = params["plot_parameters"]["sol_plots"]
    do_pdf_plot = (plot_params["do_pdf_plot"] == "y")
    do_gif_plot = (plot_params["do_gif_plot"] == "y")

    numLevels = plot_params["levels"]

    # On ne prend en compte une éventuelle solution exacte que si on la connait 
    do_exact_pdf_plot = False
    do_exact_gif_plot = False
    if (params["solution_parameters"]["simulation_choice"] == 5):
        do_exact_pdf_plot = (plot_params["do_exact_pdf_plot"] == "y")
        do_exact_gif_plot = (plot_params["do_exact_gif_plot"] == "y")

    iMin, iMax, jMin, jMax = grid.valid_grid
    X = grid.xGrid[iMin:iMax+1, jMin:jMax+1]
    Y = grid.yGrid[iMin:iMax+1, jMin:jMax+1]

    plots = [
        (XVEL, "U", "Vitesse U", "RdBu_r"),
        (YVEL, "V", "Vitesse V", "RdBu_r"),
        (PRES, "P", "Pression",  "viridis"),
    ]

    schemeShort, schemeName = getSchemeTitles(params)

    # FIGURES FOR APPROXIMATE SOLUTIONS
    for var, filename, title, cmap in plots:
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, q[:, :, var], levels=numLevels, cmap=cmap)
        # cf = ax.imshow(q[:, :, var].T, origin="lower", cmap=cmap, extent=[X.min(), X.max(), Y.min(), Y.max()], aspect="equal", interpolation="bilinear")
        ax.contour(X, Y, q[:, :, var], levels=numLevels, colors="k", linewidths=0.3)
        # plt.colorbar(cf, ax=ax, label=title, format="%.2f")
        plt.colorbar(cf, ax=ax, label=title)
        ax.set_title(f"{title} avec le schéma "+schemeName+f" à t={round(time, 1)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.tight_layout()

        if (do_pdf_plot) :
            pdf_writers[filename+"_"+schemeShort].savefig(fig)
        
        if (do_gif_plot) :
            # On écrit directement dans le gif
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            gif_writers[filename+"_"+schemeShort].append_data(frame)
        
        plt.close()
    
    

    # FIGURES FOR EXACT SOLUTIONS
    if (do_exact_pdf_plot or do_exact_gif_plot) :
        exact_plots = [
            (XVEL, "U_exact", "Vitesse U exacte", "RdBu_r"),
            (YVEL, "V_exact", "Vitesse V exacte", "RdBu_r"),
            (PRES, "P_exact", "Pression exacte",  "viridis"),
        ]

        q_exact = solutions.getSolution(time, grid, params["solution_parameters"])

        for var, filename_exact, title, cmap in exact_plots:
            fig, ax = plt.subplots()
            cf = ax.contourf(X, Y, q_exact[:, :, var], levels=50, cmap=cmap, rasterized=True)
            # cf = ax.imshow(q[:, :, var].T, origin="lower", cmap=cmap, extent=[X.min(), X.max(), Y.min(), Y.max()], aspect="equal", interpolation="bilinear")
            ax.contour(X, Y, q_exact[:, :, var], levels=50, colors="k", linewidths=0.3, rasterized=True)
            plt.colorbar(cf, ax=ax, label=title, format="%.2f")
            ax.set_title(f"{title} à t={time}")
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



# AUXILIARY ROUTINES 



# GIF Open and Close
def openGifWriters(params):
    plot_loc = params["plot_parameters"]["sol_plots"]["plot_loc"]
    gif_time = 1000. * params["plot_parameters"]["sol_plots"]["gif_time"] # Conversion s to ms !!
    schemeShort = getSchemeTitles(params)[0]
    plots = [("U_"+schemeShort, ), ("V_"+schemeShort, ), ("P_"+schemeShort, ), ("U_exact", ), ("V_exact", ), ("P_exact", )]
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
    plot_loc = params["plot_parameters"]["sol_plots"]["plot_loc"]
    schemeShort = getSchemeTitles(params)[0]
    plots = ["U_"+schemeShort, "V_"+schemeShort, "P_"+schemeShort, "U_exact", "V_exact", "P_exact"]
    return {
        filename: PdfPages(f"{plot_loc}{filename}.pdf")
        for filename in plots
    }


def closePdfWriters(pdf_writers):
    for writer in pdf_writers.values():
        writer.close()
    print("PDFs sauvegardés !")