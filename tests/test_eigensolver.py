"""
Regression tests for Eigensolver_2Dimensions.py.

These don't prove the solver is bug-free, they just pin down the accuracy levels
documented in the README (and verified by hand in validate_2d.py) so that a future
change to d2dx2_matrix, Schrodinger_solver_2D, or any of the potentials gets caught
immediately instead of silently drifting. Thresholds are set with headroom above the
measured errors - see the README for what each number actually means and why the
infinite-square-well threshold is looser than the harmonic oscillator's (same boundary-
stencil story as the 1D repo, just now along both x and y).

The delta-well test is not an accuracy check - there is nothing for it to converge to in
2D. It just pins down the direction of the (documented, expected) divergence, so that a
future change that accidentally makes it converge - which would actually be suspicious,
not an improvement - doesn't slip by unnoticed.

Run: pytest
"""

import numpy as np
from functools import partial

from Eigensolver_2Dimensions import Schrodinger_solver_2D, V_HarmonicOscillator_2D
from Eigensolver_2Dimensions import (
    V_FiniteSquareWell_2D_NonSeparable,
    V_FiniteSquareWell_2D_Separable,
)
from validate_2d import (
    validate_infinite_square_well_2d,
    validate_harmonic_oscillator_2d,
    validate_finite_square_well_2d_separable,
    convergence_study_isw_2d,
    delta_well_2d_divergence_study,
)


def test_separable_finite_well_is_separable_and_the_other_one_is_not():
    # Guards the distinction the two potentials exist to make. In the corner region
    # (|x| > Lx/2 AND |y| > Ly/2) the rectangular well is V0, while a genuine Vx(x)+Vy(y)
    # sum is 2*V0. That is exactly why only the separable one may be compared against
    # E_{nx,ny} = eps_nx + eps_ny.
    Lx = Ly = 4.0
    V0 = 40.0
    X, Y = np.meshgrid([0.0, 3.0], [0.0, 3.0], indexing='ij')  # (0,0) inside, (3,3) corner

    V_ns = V_FiniteSquareWell_2D_NonSeparable(X, Y, Lx=Lx, Ly=Ly, V0=V0, centered=True)
    V_s = V_FiniteSquareWell_2D_Separable(X, Y, Lx=Lx, Ly=Ly, V0=V0)

    assert V_ns[0, 0] == 0.0 and V_s[0, 0] == 0.0        # both vanish inside
    assert V_ns[1, 1] == V0                              # rectangular well: V0 in the corner
    assert V_s[1, 1] == 2 * V0                           # separable sum: 2*V0 there
    assert V_ns[0, 1] == V0 and V_s[0, 1] == V0          # and they agree on the edge strips


def test_separable_finite_well_2d_matches_sum_of_1d_levels():
    # Quantitative benchmark that only exists because the potential separates. Measured
    # ~1.8e-2, capped by pointwise sampling of the step in V (first order) at the coarse
    # dx=0.1 a 2D grid can afford - not by the stencil. See the docstring in validate_2d.
    # Loose threshold on purpose; this pins the structure of the spectrum, not precision.
    _, _, _, err = validate_finite_square_well_2d_separable(N=160)
    assert np.all(err < 5e-2)


def test_infinite_square_well_2d_matches_analytic():
    # Measured ~2e-8 to 2e-7 at N=90. Before the odd-extension boundary closure this sat
    # at ~3.5e-3, flat across levels, so the threshold here is deliberately tight: it is
    # the regression guard for that closure.
    _, _, _, err = validate_infinite_square_well_2d(N=90)
    assert np.all(err < 1e-5)


def test_harmonic_oscillator_2d_matches_analytic():
    # Measured ~5e-5 to 1.7e-4 at N=90. The oscillator is essentially unaffected by the
    # boundary defect (wavefunctions decay before reaching the domain edge).
    _, _, _, err = validate_harmonic_oscillator_2d(N=90)
    assert np.all(err < 5e-4)


def test_isw_2d_convergence_is_fourth_order():
    # The 5-point stencil is 4th order in the interior, and with the odd-extension closure
    # on the boundary rows that order now survives globally along both x and y. Measured
    # local slopes: 4.11, 4.08, 4.05, 4.04. Before the closure they sat at ~1.0, which is
    # what this test used to assert.
    Ns, errs = convergence_study_isw_2d()
    slopes = -np.log(errs[1:] / errs[:-1]) / np.log(Ns[1:] / Ns[:-1])
    assert np.all(slopes > 3.5)
    assert np.all(slopes < 4.6)


def test_delta_well_2d_does_not_converge():
    # Not a bug: the continuum 2D delta potential has no bound-state energy scale without
    # an explicit regularization. The discrete ground-state energy should grow in
    # magnitude, monotonically, as the grid is refined - see the README.
    Ns, E0s = delta_well_2d_divergence_study()
    assert np.all(np.diff(E0s) < 0)  # E0 gets more negative as N increases
    assert abs(E0s[-1]) > 3 * abs(E0s[0])  # growth is substantial, not a rounding wobble


def test_eigenvalues_are_sorted_and_nondegenerate_ordering():
    _, eigvals, _, _ = validate_harmonic_oscillator_2d(N=60)
    assert np.all(np.diff(eigvals) > 0)


def test_eigenfunctions_are_normalized():
    x, y, eigvals, eigvecs = Schrodinger_solver_2D(
        V_pot=partial(V_HarmonicOscillator_2D, omega_x=1.0, omega_y=1.7, m=1.0),
        x_min=-8.0, x_max=8.0, y_min=-8.0, y_max=8.0,
        Nx=60, Ny=60, num_eigvals=4,
    )
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    for n in range(eigvals.size):
        norm = np.sum(eigvecs[:, :, n] ** 2) * dx * dy
        assert abs(norm - 1.0) < 1e-8


def test_anisotropic_oscillator_node_counts_match_quantum_numbers():
    # Structural sanity check, not an accuracy check: for a well-separated (anisotropic,
    # non-degenerate) 2D harmonic oscillator, the n-th state's node count along x and
    # along y should match its (nx,ny) quantum numbers, same idea as the 1D code's
    # node-counting test but split into the two separable directions.
    x, y, eigvals, eigvecs = Schrodinger_solver_2D(
        V_pot=partial(V_HarmonicOscillator_2D, omega_x=1.0, omega_y=1.7, m=1.0),
        x_min=-8.0, x_max=8.0, y_min=-8.0, y_max=8.0,
        Nx=80, Ny=80, num_eigvals=6,
    )
    expected_nx_ny = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (3, 0)]

    def count_nodes(v):
        significant = v[np.abs(v) > 1e-3 * np.max(np.abs(v))]
        return int(np.sum(np.sign(significant)[1:] != np.sign(significant)[:-1]))

    iy0 = eigvecs.shape[0] // 2
    ix0 = eigvecs.shape[1] // 2
    for n, (nx_expected, ny_expected) in enumerate(expected_nx_ny):
        psi = eigvecs[:, :, n]
        nodes_x = count_nodes(psi[iy0, :])
        nodes_y = count_nodes(psi[:, ix0])
        assert nodes_x == nx_expected
        assert nodes_y == ny_expected
