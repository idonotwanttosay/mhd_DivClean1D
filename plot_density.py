import matplotlib.pyplot as plt
import numpy as np

import postprocess as pp

steps = pp.available_steps("rho")
if not steps:
    raise SystemExit("no rho")
xs, ys = pp.read_grid("rho")
nx, ny = len(xs), len(ys)
for s in steps:
    rho = pp.load_field("rho", s, xs, ys)
    X, Y = np.meshgrid(xs, ys)
    plt.figure(figsize=(6,5))
    plt.contourf(X, Y, rho, levels=40, cmap='viridis')
    plt.colorbar()
    plt.title(f"rho step {s}")
    plt.gca().set_aspect('equal')
    out = f"Result/rho_{s}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")
