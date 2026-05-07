import numpy as np
import grid_mod

a = 1 \
+2

print(a)

xL = 0.
xR = 1.
yL = 0.
yR = 1.

nGhost = 2
Nx = 10
Ny = 8

grid = grid_mod.gridOperator(xL, xR, yL, yR, Nx, Ny, nGhost)

print("Indices valides en x : ")
print(grid.iMin, grid.iMax)
print("")
print("Indices valides en y")
print(grid.jMin, grid.jMax)


u = np.zeros((Nx+2*nGhost, Ny+2*nGhost))
for i in range(grid.iMin, grid.iMax+1) :
    for j in range(grid.jMin, grid.jMax+1) :
        u[i, j] = i - grid.iMin + j - grid.jMin + 1

print(u)

grid.periodize(nGhost, u)
print(u)