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
from validate_2d import (
    validate_infinite_square_well_2d,
    validate_harmonic_oscillator_2d,
    convergence_study_isw_2d,
    delta_well_2d_divergence_study,
)


def test_infinite_square_well_2d_matches_analytic():
    # Measured ~3.5e-3 at N=90, flat across levels (boundary-stencil limitation, see
    # README). Generous margin.
    _, _, _, err = validate_infinite_square_well_2d(N=90)
    assert np.all(err < 6e-3)


def test_harmonic_oscillator_2d_matches_analytic():
    # Measured ~5e-5 to 1.7e-4 at N=90. The oscillator is essentially unaffected by the
    # boundary defect (wavefunctions decay before reaching the domain edge).
    _, _, _, err = validate_harmonic_oscillator_2d(N=90)
    assert np.all(err < 5e-4)


def test_isw_2d_convergence_is_first_order():
    # The 5-point stencil is 4th order in the interior, but the boundary rows cap global
    # convergence at O(1/N), not O(1/N^4) - same mechanism as the 1D code, verified here
    # by fitting the measured slope. Loose bounds since this is only 4 points.
    Ns, errs = convergence_study_isw_2d()
    slopes = np.log(errs[1:] / errs[:-1]) / np.log(Ns[1:] / Ns[:-1])
    assert np.all(slopes < -0.8)
    assert np.all(slopes > -1.3)


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
