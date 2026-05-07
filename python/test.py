import numpy as np
import matplotlib.pyplot as plt
# import sys
# import yaml
# import grid_mod

# params = yaml.load(open("parameters.yaml"),Loader=yaml.SafeLoader)

# print(params["mesh_parameters"])
# Nx, Ny, nGhost = params["mesh_parameters"].values()
# print(Nx, Ny, nGhost)



X = np.linspace(0, 1, 100+1, endpoint=True)
Y = X**2

plt.figure()
plt.plot(X, Y)
plt.show()

