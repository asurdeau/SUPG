# Modules généraux
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# Modules perso
from config import XVEL, YVEL, PRES 

def makePlots(q, time, params, grid, gif_frames):
    plot_loc = params["plot_parameters"]["plot_loc"]
    do_pdf_plot = params["plot_parameters"]["do_pdf_plot"]
    do_gif_plot = params["plot_parameters"]["do_gif_plot"]
    X = grid.xGrid
    Y = grid.yGrid

    plots = [
        (XVEL, "U", "Vitesse U", "RdBu_r"),
        (YVEL, "V", "Vitesse V", "RdBu_r"),
        (PRES, "P", "Pression",  "viridis"),
    ]

    for var, filename, title, cmap in plots:
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, q[:, :, var], levels=50, cmap=cmap)
        ax.contour(X, Y, q[:, :, var], levels=50, colors="k", linewidths=0.3)
        plt.colorbar(cf, ax=ax, label=title, format="%.2f")
        ax.set_title(f"{title} à t={time}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.tight_layout()
        
        if (do_gif_plot) :
            png_path = f"{plot_loc}{filename}{time}.png"
            plt.savefig(png_path)
            gif_frames[filename].append(png_path)

        if (do_pdf_plot) :
            plt.savefig(f"{plot_loc}{filename}{time}.pdf")

        plt.close()

def saveGifs(gif_frames, params):
    plot_loc = params["plot_loc"]
    gif_time = 1000 * params["gif_time"] # Conversions en ms !
    for filename, frames in gif_frames.items():
        images = [imageio.imread(frame) for frame in frames]
        imageio.mimwrite(
            f"{plot_loc}{filename}.gif",
            images,
            duration=gif_time,
            loop=0
        )
        for frame in frames:
            os.remove(frame)
    print("GIFs sauvegardés !")