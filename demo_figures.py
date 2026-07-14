"""
Regenerates the "gallery" figures used in the README:

  figures/wavefunctions_harmonic2d.png   - anisotropic 2D HO eigenfunctions (heatmaps)
  figures/surface_harmonic2d.png         - anisotropic 2D HO, one state as a 3D surface
  figures/energylevels_harmonic2d.png    - anisotropic 2D HO energy-level diagram
  figures/wavefunctions_doublewell2d.png - quartic-x / harmonic-y double well eigenfunctions
  figures/energylevels_doublewell2d.png  - double well energy-level diagram

The other two README figures (convergence_isw2d.png and deltawell_divergence2d.png) come
from validate_2d.py instead, since they are direct byproducts of the numerical validation,
not just demo eigenfunctions.

Both potentials here use anisotropic/non-square parameters on purpose (omega_x != omega_y
for the oscillator, Vy far from the x-well's tunneling scale for the double well): the
isotropic case has exact eigenvalue degeneracies, and eigsh - like any single-vector
Krylov method - is not guaranteed to resolve a degenerate eigenspace into clean, separable
(nx,ny) states. See the README for the numerical evidence.

Run: python demo_figures.py
"""

import matplotlib.pyplot as plt
from functools import partial

from Eigensolver_2Dimensions import (
    Schrodinger_solver_2D,
    V_HarmonicOscillator_2D,
    V_DoubleWell_2D,
    plot_eigenfunction_grid,
    plot_eigenfunction_surface,
    plot_energy_levels_2d,
)

if __name__ == "__main__":
    # --- Anisotropic harmonic oscillator ---
    x, y, eigvals, eigvecs = Schrodinger_solver_2D(
        V_pot=partial(V_HarmonicOscillator_2D, omega_x=1.0, omega_y=1.7, m=1.0),
        x_min=-8.0, x_max=8.0, y_min=-8.0, y_max=8.0,
        Nx=80, Ny=80, num_eigvals=6,
    )
    ho_labels = ["(0,0)", "(1,0)", "(0,1)", "(2,0)", "(1,1)", "(3,0)"]
    ho_titles = [f"n={n}  (nx,ny)={ho_labels[n]}" for n in range(6)]

    fig, axes = plot_eigenfunction_grid(
        x, y, eigvecs, n_states=6, ncols=3, titles=ho_titles,
        suptitle="Anisotropic 2D harmonic oscillator: eigenfunctions (omega_x=1.0, omega_y=1.7)",
    )
    fig.savefig("figures/wavefunctions_harmonic2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ax = plot_eigenfunction_surface(x, y, eigvecs, n=4, title="n=4, (nx,ny)=(1,1)")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.view_init(elev=28, azim=-60)
    fig = ax.figure
    fig.tight_layout()
    fig.savefig("figures/surface_harmonic2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ax = plot_energy_levels_2d(
        eigvals, n_states=6, labels=[f"(nx,ny)={l}" for l in ho_labels],
        title="Anisotropic 2D harmonic oscillator: energy levels",
    )
    fig = ax.figure
    fig.savefig("figures/energylevels_harmonic2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Anisotropic HO lowest energies:")
    for n, En in enumerate(eigvals):
        print(f"  n={n} (nx,ny)={ho_labels[n]}: E = {En:.6f}")

    # --- Quartic double well (x) times harmonic (y) ---
    x, y, eigvals, eigvecs = Schrodinger_solver_2D(
        V_pot=partial(V_DoubleWell_2D, a=1.5, Vx=1.0, Vy=3.0),
        x_min=-4.0, x_max=4.0, y_min=-4.0, y_max=4.0,
        Nx=80, Ny=80, num_eigvals=6,
    )
    dw_labels = ["(0,0)", "(1,0)", "(0,1)", "(1,1)", "(2,0)", "(3,0)"]
    dw_titles = [f"n={n}  (nx,ny)={dw_labels[n]}" for n in range(6)]

    fig, axes = plot_eigenfunction_grid(
        x, y, eigvecs, n_states=6, ncols=3, titles=dw_titles,
        suptitle="Quartic double well (x) times harmonic (y): eigenfunctions",
    )
    fig.savefig("figures/wavefunctions_doublewell2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ax = plot_energy_levels_2d(
        eigvals, n_states=6, labels=[f"(nx,ny)={l}" for l in dw_labels],
        title="Double well: energy levels",
    )
    fig = ax.figure
    fig.savefig("figures/energylevels_doublewell2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nDouble well lowest energies:")
    for n, En in enumerate(eigvals):
        print(f"  n={n} (nx,ny)={dw_labels[n]}: E = {En:.6f}")
    print(f"\nTunneling splitting, ny=0 doublet (n=0,1): {eigvals[1]-eigvals[0]:.5e}")
    print(f"Tunneling splitting, ny=1 doublet (n=2,3): {eigvals[3]-eigvals[2]:.5e}")

    print("\nSaved 5 figures to figures/")
