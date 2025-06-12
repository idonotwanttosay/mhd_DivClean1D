import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

import postprocess as pp

steps = pp.available_steps("rho")
if not steps:
    raise SystemExit("no output")

xs, ys = pp.read_grid("rho")
X, Y = np.meshgrid(xs, ys)

fig, ax = plt.subplots(figsize=(6,5))


def update(step):
    ax.clear()
    rho = pp.load_field("rho", step, xs, ys)
    c = ax.contourf(X, Y, rho, levels=40, cmap="viridis")
    grid_file = f"Result/grid_{step}.csv"
    try:
        data = np.loadtxt(grid_file, delimiter=",")
        if data.ndim == 1:
            data = data[None, :]
        for lvl, x0, y0, dx, dy, nx, ny in data:
            rect = Rectangle((x0, y0), dx * (nx - 1), dy * (ny - 1),
                             fill=False, edgecolor="red", lw=1)
            ax.add_patch(rect)
    except OSError:
        pass
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_title(f"step {step}")
    ax.set_aspect("equal")
    return c.collections

ani = FuncAnimation(fig, update, frames=steps, blit=False)
ani.save("Result/animation.mp4", fps=3)
print("Saved Result/animation.mp4")
