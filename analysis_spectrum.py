"""Compute kinetic energy spectrum and save a PNG."""

import matplotlib.pyplot as plt
import numpy as np

import postprocess as pp

steps = pp.available_steps("u")
if not steps:
    raise SystemExit("No velocity output found.")

xs, ys = pp.read_grid("u")
nx = len(xs)
ny = len(ys)

def load(step: int, comp: str) -> np.ndarray:
    return pp.load_field(comp, step, xs, ys)

def spectrum(u,v):
    u = u - u.mean(); v = v - v.mean()
    uhat = np.fft.fft2(u); vhat = np.fft.fft2(v)
    E = 0.5*(np.abs(uhat)**2 + np.abs(vhat)**2)/(nx*ny)**2
    kx = np.fft.fftfreq(nx)*(nx)
    ky = np.fft.fftfreq(ny)*(ny)
    KX,KY = np.meshgrid(kx,ky)
    kmag = np.sqrt(KX**2+KY**2)
    k_bins = np.arange(0.5, min(nx,ny)//2 +1 ,1)
    Ek=np.zeros_like(k_bins)
    for i in range(len(k_bins)):
        mask = (kmag>=k_bins[i]-0.5)&(kmag<k_bins[i]+0.5)
        Ek[i]=E[mask].sum()
    return k_bins,Ek

plt.figure(figsize=(8,4))
for s in steps:
    u=load(s,"u"); v=load(s,"v")
    k,Ek = spectrum(u,v)
    plt.loglog(k,Ek,label=f"s{s}")
plt.loglog([1,10],[1e-2,1e-2*10**(-5/3)],'k--',label='k^-5/3')
plt.xlabel("k"); plt.ylabel("E(k)")
plt.title("Kinetic Energy Spectrum")
plt.legend(fontsize=7,ncol=2)
plt.tight_layout()
plt.savefig("Result/energy_spectrum.png",dpi=200)
print("Saved Result/energy_spectrum.png")
