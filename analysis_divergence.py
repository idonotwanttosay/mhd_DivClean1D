"""Compute L2 norm of magnetic field divergence over time."""

import matplotlib.pyplot as plt
import numpy as np

import postprocess as pp

steps = pp.available_steps("bx")
if not steps:
    raise RuntimeError("No B field output found. Did you recompile & rerun the solver?")

xs, ys = pp.read_grid("bx")
dx = xs[1] - xs[0]
dy = ys[1] - ys[0]

def load(step: int, prefix: str) -> np.ndarray:
    return pp.load_field(prefix, step, xs, ys)

l2 = []
for s in steps:
    bx = load(s, "bx")
    by = load(s, "by")
    div = (np.roll(bx, -1, 1) - np.roll(bx, 1, 1)) / (2 * dx) + (
        np.roll(by, -1, 0) - np.roll(by, 1, 0)
    ) / (2 * dy)
    l2.append(np.sqrt(np.mean(div ** 2)))

plt.semilogy(steps,l2,'o-')
plt.xlabel("step"); plt.ylabel("L2(∇·B)")
plt.title("Magnetic Divergence Error")
plt.tight_layout()
plt.savefig("Result/divB_error.png",dpi=200)
print("Saved Result/divB_error.png")
