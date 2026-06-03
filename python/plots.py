# Modules généraux
import numpy as np
import matplotlib.pyplot as plt
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



def makePlots(q, time, params, grid, gif_writers):
    plot_loc = params["plot_parameters"]["plot_loc"]
    do_pdf_plot = params["plot_parameters"]["do_pdf_plot"]
    do_gif_plot = params["plot_parameters"]["do_gif_plot"]
    do_exact_plot = params["plot_parameters"]["do_exact_plot"]
    iMin, iMax, jMin, jMax = grid.valid_grid
    X = grid.xGrid[iMin:iMax+1, jMin:jMax+1]
    Y = grid.yGrid[iMin:iMax+1, jMin:jMax+1]

    plots = [
        (XVEL, "U", "Vitesse U", "RdBu_r"),
        (YVEL, "V", "Vitesse V", "RdBu_r"),
        (PRES, "P", "Pression",  "viridis"),
    ]

    schemeShort, schemeName = getSchemeTitles(params)

    for var, filename, title, cmap in plots:
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, q[:, :, var], levels=50, cmap=cmap)
        ax.contour(X, Y, q[:, :, var], levels=50, colors="k", linewidths=0.3)
        # plt.colorbar(cf, ax=ax, label=title, format="%.2f")
        plt.colorbar(cf, ax=ax, label=title)
        ax.set_title(f"{title} avec le schéma "+schemeName+f" à t={round(time, 1)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.tight_layout()
        
        if (do_gif_plot) :
            # On écrit directement dans le gif
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            gif_writers[filename+"_"+schemeShort].append_data(frame)

        if (do_pdf_plot) :
            plt.savefig(f"{plot_loc}{filename}{time}.pdf")
        
        plt.close()
    
    
    if (do_exact_plot) :
        exact_plots = [
            (XVEL, "U_exact", "Vitesse U exacte", "RdBu_r"),
            (YVEL, "V_exact", "Vitesse V exacte", "RdBu_r"),
            (PRES, "P_exact", "Pression exacte",  "viridis"),
        ]

        q_exact = solutions.getSolution(time, X, Y, params["solution_parameters"])

        for var, filename_exact, title, cmap in exact_plots:
            fig, ax = plt.subplots()
            cf = ax.contourf(X, Y, q_exact[:, :, var], levels=50, cmap=cmap)
            ax.contour(X, Y, q_exact[:, :, var], levels=50, colors="k", linewidths=0.3)
            plt.colorbar(cf, ax=ax, label=title, format="%.2f")
            ax.set_title(f"{title} à t={time}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            plt.tight_layout()
            
            if (do_gif_plot) :
                # On écrit directement dans le gif
                fig.canvas.draw()
                frame = np.array(fig.canvas.renderer.buffer_rgba())
                gif_writers[filename_exact].append_data(frame)

            if (do_pdf_plot) :
                plt.savefig(f"{plot_loc}{filename_exact}{time}.pdf")

            plt.close()


def openGifWriters(params):
    plot_loc = params["plot_parameters"]["plot_loc"]
    gif_time = 1000. * params["plot_parameters"]["gif_time"] # Conversion s to ms !!
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