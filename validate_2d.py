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
number of grid points N and compared against an O(1/N^4) reference line. This is the
direct numerical evidence for the boundary closure documented in d2dx2_matrix's docstring
in Eigensolver_2Dimensions.py - the same closure as the 1D code, applied along both the x
and y boundary rows. Before it, the well sat at 1.07e-2 / 7.06e-3 / 4.85e-3 / 3.48e-3 for
N = 30 / 45 / 65 / 90 with a measured slope of about 1.0 (i.e. global O(1/N)); it now
gives 1.53e-6 / 2.89e-7 / 6.45e-8 / 1.73e-8 with a slope of about 4.1.

Note that the potentials with a jump discontinuity (V_FiniteSquareWell_2D_*) are still
capped near first order by pointwise sampling of the step onto the mesh. That is a
separate issue from the boundary rows and is not addressed here.

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
    V_FiniteSquareWell_2D_Separable,
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


def _bisect(f, a, b, tol=1e-13, maxit=300):
    fa, fb = f(a), f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0:
        return None
    for _ in range(maxit):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < tol or (b - a) < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


def _finite_well_1d_levels(L, V0, m=1.0, hbar=1.0, n_scan=200000):
    '''
    Bound states of the 1D symmetric finite well of width L and depth V0, from the usual
    transcendental conditions k*tan(kL/2) = kappa (even) and k*cot(kL/2) = -kappa (odd),
    with k = sqrt(2mE)/hbar and kappa = sqrt(2m(V0-E))/hbar. Same hand-rolled bisection
    used in validate_1d.py. For L=4, V0=40 this returns eps_1 = 0.276598377 as its first
    entry.
    '''
    def k_(E): return np.sqrt(2 * m * E) / hbar
    def kap_(E): return np.sqrt(2 * m * (V0 - E)) / hbar

    def f_even(E):
        k, kap = k_(E), kap_(E)
        return k * np.tan(k * L / 2) - kap

    def f_odd(E):
        k, kap = k_(E), kap_(E)
        return k / np.tan(k * L / 2) + kap

    eps = 1e-9
    Es = np.linspace(eps, V0 - eps, n_scan)
    roots = []
    for f in (f_even, f_odd):
        vals = f(Es)
        for i in range(len(Es) - 1):
            v0, v1 = vals[i], vals[i + 1]
            if np.isfinite(v0) and np.isfinite(v1) and v0 * v1 < 0 and abs(v0) < 50 and abs(v1) < 50:
                r = _bisect(f, Es[i], Es[i + 1])
                if r is not None:
                    roots.append(r)
    return np.array(sorted(roots))


def validate_finite_square_well_2d_separable(Lx=4.0, Ly=5.0, V0=40.0, N=160,
                                             num_eigvals=4, hbar=1.0, m=1.0):
    '''
    Quantitative benchmark for a finite well in 2D.

    This only works because V_FiniteSquareWell_2D_Separable is built as Vx(x) + Vy(y),
    so the Hamiltonian separates and E_{nx,ny} = eps_nx(Lx,V0) + eps_ny(Ly,V0) with eps
    the 1D bound states. The plain rectangular well V_FiniteSquareWell_2D_NonSeparable is
    NOT separable (it is V0 in the corners where the separable sum would be 2*V0), so its
    spectrum does not factorize and it must not be compared against this formula.

    Lx != Ly on purpose, to avoid exact degeneracies (see the module docstring).

    Accuracy: expect ~2e-2, not 1e-8, and do not read that as a failure of the solver.
    The jump in V is sampled pointwise onto the mesh, which caps convergence near first
    order regardless of the boundary closure, and 2D resolutions are much coarser than 1D
    ones at equal cost: N=160 per axis here means dx = 0.1, against dx = 0.008 for the 1D
    finite-well check in validate_1d.py, which lands at 8e-4. Scaling 8e-4 by 0.1/0.008
    gives ~1e-2, so the number below is exactly what first-order behaviour predicts. The
    error is also a nearly uniform shift across levels (~1.8e-2 on all four), which is the
    signature of a systematic potential-sampling bias rather than a stencil problem.

    The residual is somewhat sensitive to where the step lands relative to the grid: at
    N=157 (which puts the x-edges midway between grid points) it drops to ~1.1e-2. A
    proper fix is cell-averaged sampling of V instead of pointwise np.where; that is not
    implemented here.
    '''
    eps_x = _finite_well_1d_levels(Lx, V0, m=m, hbar=hbar)
    eps_y = _finite_well_1d_levels(Ly, V0, m=m, hbar=hbar)
    levels = sorted(ex + ey for ex in eps_x for ey in eps_y)
    analytic = np.array(levels[:num_eigvals])

    x, y, eigvals, eigvecs = Schrodinger_solver_2D(
        V_pot=partial(V_FiniteSquareWell_2D_Separable, Lx=Lx, Ly=Ly, V0=V0),
        x_min=-8.0, x_max=8.0, y_min=-8.0, y_max=8.0,
        Nx=N, Ny=N, hbar=hbar, m=m, num_eigvals=num_eigvals,
    )
    rel_err = np.abs(eigvals - analytic) / analytic
    return np.arange(num_eigvals), eigvals, analytic, rel_err


def print_table(name, ns, numeric, analytic, rel_err):
    print(f"\n{name}")
    print(f"{'n':>3} {'numeric':>14} {'analytic':>14} {'rel. error':>12}")
    for n, num, an, err in zip(ns, numeric, analytic, rel_err):
        print(f"{n:>3} {num:>14.8f} {an:>14.8f} {err:>12.3e}")


def convergence_study_isw_2d(L=10.0, Ns=(30, 45, 65, 90, 120), hbar=1.0, m=1.0):
    '''
    Ground state only (always non-degenerate, so there's no Krylov-degeneracy subtlety
    here regardless of the square domain). Same idea as the 1D convergence study: compare
    the measured error against an O(1/N^4) reference line to confirm that the
    odd-extension boundary closure restores the design order of the stencil. Measured
    local slopes on this grid: 4.11, 4.08, 4.05, 4.04.
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

    ns, num, an, err = validate_finite_square_well_2d_separable()
    print_table("Finite square well, SEPARABLE (Lx=4, Ly=5, V0=40, N=160)", ns, num, an, err)

    Ns, errs = convergence_study_isw_2d()
    slopes = -np.diff(np.log(errs)) / np.diff(np.log(Ns))
    print("\n2D infinite square well, ground-state convergence vs N:")
    for i, (N, e) in enumerate(zip(Ns, errs)):
        slope = f"   local slope p={slopes[i-1]:.2f}" if i > 0 else ""
        print(f"N={N:>5}   rel. error={e:.3e}{slope}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(Ns, errs, "o-", label="ground state, measured")
    ax.loglog(Ns, errs[0] * (Ns[0] / Ns) ** 4, "--", color="gray", label=r"$O(1/N^4)$ reference")
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
