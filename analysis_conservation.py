"""Check mass and energy conservation from output CSV files."""

import matplotlib.pyplot as plt
import numpy as np

import postprocess as pp

steps = pp.available_steps("rho")
if not steps:
    raise SystemExit("No output found")

xs, ys = pp.read_grid("rho")
DX = xs[1] - xs[0]
DY = ys[1] - ys[0]

def load(step: int, prefix: str) -> np.ndarray:
    return pp.load_field(prefix, step, xs, ys)

mass=[]; energy=[]
for s in steps:
    rho = load(s, "rho")
    e   = load(s, "e")
    mass.append(rho.sum()*DX*DY)
    energy.append(e.sum()*DX*DY)

plt.plot(steps, mass, label='mass')
plt.plot(steps, energy, label='energy')
plt.xlabel('step'); plt.ylabel('total')
plt.title('Conservation check')
plt.legend()
plt.tight_layout()
plt.savefig('Result/conservation.png', dpi=200)
print('Saved Result/conservation.png')
