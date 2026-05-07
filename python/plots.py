# Modules généraux
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# Modules perso
from config import XVEL, YVEL, PRES 

def makePlots(q, time, params, grid, gif_frames):
    grid_params = params["grid_parameters"]
    Nx, Ny, _ = grid_params["mesh_parameters"].values()
    xL, xR, yL, yR = grid_params["domain_parameters"].values()
    iMin, iMax, jMin, jMax = grid.valid_grid
    q_valid = q[iMin:iMax+2, jMin:jMax+2]
    plot_loc = params["plot_parameters"]["plot_loc"]
    XX = np.linspace(xL, xR, Nx+1)
    YY = np.linspace(yL, yR, Ny+1)
    X, Y = np.meshgrid(XX, YY)

    plots = [
        (XVEL, "u_t", "Vitesse U", "RdBu_r"),
        (YVEL, "v_t", "Vitesse V", "RdBu_r"),
        (PRES, "p_t", "Pression",  "viridis"),
    ]

    for var, filename, title, cmap in plots:
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, q_valid[:, :, var], levels=50, cmap=cmap)
        ax.contour(X, Y, q_valid[:, :, var], levels=50, colors="k", linewidths=0.3)
        plt.colorbar(cf, ax=ax, label=title, format="%.2f")
        ax.set_title(f"{title} à t={time}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.tight_layout()
        
        
        png_path = f"{plot_loc}{filename}{time}.png"
        plt.savefig(png_path)
        gif_frames[filename].append(png_path)

        plt.savefig(f"{plot_loc}{filename}{time}.pdf")
        plt.close()


def saveGifs(gif_frames, plot_params):
    plot_loc = plot_params["plot_loc"]
    duration = plot_params["gif_time"]
    for filename, frames in gif_frames.items():
        with imageio.get_writer(f"{plot_loc}{filename}.gif", mode="I", duration=duration) as writer:
            for frame in frames:
                writer.append_data(imageio.imread(frame))
                os.remove(frame)
    print("GIFs sauvegardés !")