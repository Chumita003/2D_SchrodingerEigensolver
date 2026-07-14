"""
Validation for Eigensolver_2Dimensions.py

Compares the numerical eigenvalues against potentials that have a closed-form spectrum,
and reports the relative error per level:

  - Infinite square well (rectangular): E_{nx,ny} = (pi^2 hbar^2 / 2m) (nx^2/Lx^2 + ny^2/Ly^2)
  - Harmonic oscillator (anisotropic):  E_{nx,ny} = hbar*omega_x*(nx+1/2) + hbar*omega_y*(ny+1/2)

Both are checked with Lx != Ly (well) and omega_x != omega_y (oscillator) on purpose.
The isotropic cases (square well, omega_x == omega_y) have exact degeneracies, and
eigsh/ARPACK - like any single-vector Krylov method - is not guaranteed to resolve a
degenerate eigenspace correctly: it can return fewer distinct directions than the
requested num_eigvals, or an arbitrary mixed basis within a degenerate subspace instead
of the separable (nx,ny) states. This is a property of Krylov subspace methods, not a
bug in this code - see the README for a hands-on demonstration. Using anisotropic/
rectangular parameters for validation sidesteps the issue entirely.

It also reproduces the convergence-order plot referenced in the README: for the (square)
infinite square well, the ground-state relative error is measured as a function of the
number of grid points N and compared against an O(1/N) reference line. This is the direct
numerical evidence for the boundary-stencil limitation documented in d2dx2_matrix's
docstring in Eigensolver_2Dimensions.py - the same limitation as the 1D code, now along
both x and y boundary rows.

Finally, it runs a divergence study for the discrete 2D delta well: unlike the 1D delta
well (which has an exact closed-form bound-state energy that the discretization
converges to), the 2D delta potential has no continuum limit without an explicit
regularization scale. The measured ground-state energy grows without bound as the grid is
refined - this is a real feature of the continuum 2D delta potential, not a numerical bug.

Run: python validate_2d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from Eigensolver_2Dimensions import (
    Schrodinger_solver_2D,
    V_HarmonicOscillator_2D,
    V_InfiniteSquareWell_2D,
    V_DeltaDiscrete_2D,
)


def validate_infinite_square_well_2d(Lx=12.0, Ly=8.0, N=90, num_eigvals=6, hbar=1.0, m=1.0):
    x, y, eigvals, eigvecs = Schrodinger_solver_2D(
        V_pot=V_InfiniteSquareWell_2D,
        x_min=-0.5 * Lx, x_max=0.5 * Lx,
        y_min=-0.5 * Ly, y_max=0.5 * Ly,
        Nx=N, Ny=N, hbar=hbar, m=m,
        num_eigvals=num_eigvals,
    )
    levels = sorted(
        (np.pi ** 2 * hbar ** 2 / (2 * m)) * (nx ** 2 / Lx ** 2 + ny ** 2 / Ly ** 2)
        for nx in range(1, 9) for ny in range(1, 9)
    )
    analytic = np.array(levels[:num_eigvals])
    rel_err = np.abs(eigvals - analytic) / analytic
    return np.arange(num_eigvals), eigvals, analytic, rel_err


def validate_harmonic_oscillator_2d(omega_x=1.0, omega_y=1.7, m=1.0, hbar=1.0,
                                     x_min=-8.0, x_max=8.0, y_min=-8.0, y_max=8.0,
                                     N=90, num_eigvals=6):
    x, y, eigvals, eigvecs = Schrodinger_solver_2D(
        V_pot=partial(V_HarmonicOscillator_2D, omega_x=omega_x, omega_y=omega_y, m=m),
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        Nx=N, Ny=N, hbar=hbar, m=m,
        num_eigvals=num_eigvals,
    )
    levels = sorted(
        hbar * omega_x * (nx + 0.5) + hbar * omega_y * (ny + 0.5)
        for nx in range(6) for ny in range(6)
    )
    analytic = np.array(levels[:num_eigvals])
    rel_err = np.abs(eigvals - analytic) / analytic
    return np.arange(num_eigvals), eigvals, analytic, rel_err


def print_table(name, ns, numeric, analytic, rel_err):
    print(f"\n{name}")
    print(f"{'n':>3} {'numeric':>14} {'analytic':>14} {'rel. error':>12}")
    for n, num, an, err in zip(ns, numeric, analytic, rel_err):
        print(f"{n:>3} {num:>14.8f} {an:>14.8f} {err:>12.3e}")


def convergence_study_isw_2d(L=10.0, Ns=(30, 45, 65, 90), hbar=1.0, m=1.0):
    '''
    Ground state only (always non-degenerate, so there's no Krylov-degeneracy subtlety
    here regardless of the square domain). Same idea as the 1D convergence study: fit the
    measured error against an O(1/N) reference line to confirm the boundary-stencil
    limitation caps global convergence at first order.
    '''
    errs = []
    analytic_gs = (np.pi ** 2 * hbar ** 2 / (2 * m)) * (1 / L ** 2 + 1 / L ** 2)
    for N in Ns:
        _, _, eigvals, _ = Schrodinger_solver_2D(
            V_pot=V_InfiniteSquareWell_2D,
            x_min=-0.5 * L, x_max=0.5 * L, y_min=-0.5 * L, y_max=0.5 * L,
            Nx=N, Ny=N, hbar=hbar, m=m, num_eigvals=1,
        )
        errs.append(abs(eigvals[0] - analytic_gs) / analytic_gs)
    return np.array(Ns), np.array(errs)


def delta_well_2d_divergence_study(alpha=2.0, Ns=(30, 50, 70, 90), hbar=1.0, m=1.0):
    '''
    NOT a convergence check - there is nothing for this to converge to. The discrete 2D
    delta well's ground-state energy grows in magnitude as N increases (dx shrinks),
    because the continuum 2D delta potential has no bound-state energy scale without an
    explicit regularization/cutoff. This function just measures and returns that growth.
    '''
    E0s = []
    for N in Ns:
        _, _, eigvals, _ = Schrodinger_solver_2D(
            V_pot=partial(V_DeltaDiscrete_2D, alpha=alpha, x0=0.0, y0=0.0),
            x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0,
            Nx=N, Ny=N, hbar=hbar, m=m, num_eigvals=1,
        )
        E0s.append(eigvals[0])
    return np.array(Ns), np.array(E0s)


if __name__ == "__main__":
    ns, num, an, err = validate_infinite_square_well_2d()
    print_table("Infinite square well, rectangular (Lx=12, Ly=8, hbar=m=1, N=90)", ns, num, an, err)

    ns, num, an, err = validate_harmonic_oscillator_2d()
    print_table("Harmonic oscillator, anisotropic (omega_x=1, omega_y=1.7, N=90)", ns, num, an, err)

    Ns, errs = convergence_study_isw_2d()
    print("\n2D infinite square well, ground-state convergence vs N:")
    for N, e in zip(Ns, errs):
        print(f"N={N:>5}   rel. error={e:.3e}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(Ns, errs, "o-", label="ground state, measured")
    ax.loglog(Ns, errs[0] * Ns[0] / Ns, "--", color="gray", label=r"$O(1/N)$ reference")
    ax.set_xlabel("N (grid points per axis)")
    ax.set_ylabel("relative error")
    ax.set_title("2D infinite square well: convergence order")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/convergence_isw2d.png", dpi=150)
    print("\nSaved figures/convergence_isw2d.png")

    Ns_d, E0s = delta_well_2d_divergence_study()
    print("\n2D discrete delta well, ground-state energy vs N (does NOT converge):")
    for N, E0 in zip(Ns_d, E0s):
        print(f"N={N:>5}   E0={E0:.6f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(Ns_d, E0s, "o-", color="crimson")
    ax.set_xlabel("N (grid points per axis)")
    ax.set_ylabel(r"$E_0$ (ground-state energy)")
    ax.set_title("2D discrete delta well: no continuum limit")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/deltawell_divergence2d.png", dpi=150)
    print("Saved figures/deltawell_divergence2d.png")
