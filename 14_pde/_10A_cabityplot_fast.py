import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

nx = 41
ny = 41
nt = 500

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots()

def update_plot(n):
    data = np.loadtxt(f"results_{n}.txt")
    p = data[:, 2].reshape(ny, nx)
    u = data[:, 3].reshape(ny, nx)
    v = data[:, 4].reshape(ny, nx)

    ax.clear()
    ax.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
    ax.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])

ani = FuncAnimation(fig, update_plot, frames=nt, interval=1)
plt.show()
