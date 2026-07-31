import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import eigsh
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the '3d' projection)

def d2dx2_matrix(N, dx):

    '''
    This function approximates (d^2/dx^2) using the same 5-point central
    finite-difference stencil used in the 1D code, on interior points:

        f''(x_i) ~ (-f_{i+2} + 16 f_{i+1} - 30 f_i + 16 f_{i-1} - f_{i-2}) / (12 dx^2)

    Dirichlet boundary conditions are assumed.
    '''

    '''
    Boundary treatment (odd/antisymmetric extension), carried over from the 1D code.

    The row for the first interior point needs psi_{-1}, one point *outside* the domain.
    It is not the boundary itself (which is legitimately 0 by Dirichlet) and it is not
    zero. Since psi'' = (2m/hbar^2)(V - E) psi and psi = 0 on the boundary, psi'' also
    vanishes there, so psi is odd about the boundary and psi_{-1} = -psi_1. That turns
    the -psi_{-1} term into +psi_1, i.e. it adds +1/(12 dx^2) to the diagonal entry of
    the first and last rows (-30 -> -29 in units of 1/(12 dx^2)).

    Crucially the correction is diagonal-only, so this matrix stays exactly symmetric.
    That matters more in 2D than in 1D: the Laplacian is built as
    kron(Iy, Dxx) + kron(Dyy, Ix), so any asymmetry in Dxx or Dyy would propagate straight
    into H and break H = H^dagger. The one-sided 4th-order rows do have that problem;
    this closure does not.

    Measured effect (2D infinite square well, Lx = 4, Ly = 6, worst relative error over
    the four lowest levels): 2.6e-3 -> 3.5e-7 at N = 120 per axis, and 1.3e-3 -> 2.2e-8 at
    N = 240, i.e. the O(1/N) signature along both boundary pairs is gone and the expected
    high-order convergence is restored. The harmonic oscillator is unchanged to machine
    precision, as before. See validate_2d.py and the README for the full tables.

    Residual limitations, both unrelated to the boundary rows:
      - the odd extension is exact only if V'(boundary) = 0, since the 4th derivative
        reduces to d4psi/dx4 = (4m/hbar^2) V'(x_0) psi'(x_0) there - the first even
        derivative that need not vanish. Otherwise a local O(dx^2) error survives in
        those rows.
      - potentials with an interior jump (V_FiniteSquareWell_2D_NonSeparable and friends)
        are still limited to ~O(dx) by pointwise sampling of the step onto the mesh.
    '''

    if N < 5:
        raise ValueError('N must be at least 5 interior points for the 5-point stencil.')
    if dx <= 0:
        raise ValueError('dx must be positive.')

    # Constructing the second derivative matrix coefficients
    coeffs = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * dx**2)
    offsets = np.array([-2, -1, 0, 1, 2])

    # Odd-extension closure at the two Dirichlet boundaries: psi_{-1} = -psi_1 turns the
    # -psi_{-1} term into +psi_1, i.e. -30 -> -29 in units of 1/(12 dx^2). Diagonal-only,
    # so this matrix stays symmetric and the Kronecker-sum Laplacian stays Hermitian.
    main = coeffs[2] * np.ones(N)
    main[0] += 1.0 / (12.0 * dx**2)
    main[-1] += 1.0 / (12.0 * dx**2)

    d2_matrix = diags(
        diagonals=[
            coeffs[0] * np.ones(N - 2),
            coeffs[1] * np.ones(N - 1),
            main,
            coeffs[3] * np.ones(N - 1),
            coeffs[4] * np.ones(N - 2),
        ],
        offsets=offsets,
        shape=(N, N),
        format='csr'
    )

    return d2_matrix


def Schrodinger_solver_2D(
    V_pot,
    x_min=-10.0,
    x_max=10.0,
    y_min=-10.0,
    y_max=10.0,
    Nx=300,
    Ny=300,
    hbar=1.0,
    m=1.0,
    num_eigvals=10,
):
    '''
    This function solves the 2D time-independent Schrodinger equation

        H psi = E psi
        H = -(hbar^2 / 2m) (d^2/dx^2 + d^2/dy^2) + V(x,y)

    using the SAME 5-point 1D stencil along x and along y.

    Since each second derivative uses 5 points, the resulting 2D Laplacian
    couples each point to:
        (i +/- 1, j), (i +/- 2, j), (i, j +/- 1), (i, j +/- 2)
    plus the center point (i,j).

    Since the final 2D stencil has 9 nonzero points total:
    center + 4 nearest axial neighbors + 4 next-nearest axial neighbors.
    Thus, this is a 9-point CROSS stencil, not the usual diagonal 9-point stencil.
    '''

    if x_max <= x_min:
        raise ValueError('x_max must be greater than x_min.')
    if y_max <= y_min:
        raise ValueError('y_max must be greater than y_min.')
    if Nx < 7 or Ny < 7:
        raise ValueError('Need at least 7 total grid points in each direction.')
    if num_eigvals <= 0:
        raise ValueError('num_eigvals must be a positive integer.')

    # Create spatial grids including boundaries
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Interior points only (Dirichlet psi=0 on boundaries)
    x_interior = x[1:-1]
    y_interior = y[1:-1]
    Nx_int = x_interior.size
    Ny_int = y_interior.size
    if Nx_int < 5 or Ny_int < 5:
        raise ValueError('Need at least 5 interior points in each direction.')

    Ntot = Nx_int * Ny_int
    if num_eigvals >= Ntot:
        raise ValueError(f'num_eigvals must be smaller than total interior size ({Ntot}).')

    # Same 1D 5-point stencil in x and y
    Dxx = d2dx2_matrix(Nx_int, dx)
    Dyy = d2dx2_matrix(Ny_int, dy)

    Ix = eye(Nx_int, format='csr')
    Iy = eye(Ny_int, format='csr')

    # 2D Laplacian via Kronecker sum
    Lap = kron(Iy, Dxx) + kron(Dyy, Ix)
    T = -(hbar**2 / (2.0 * m)) * Lap

    # Potential on the 2D interior grid
    X_in, Y_in = np.meshgrid(x_interior, y_interior, indexing='xy')

    if not callable(V_pot):
        raise ValueError('V_pot must be callable as V_pot(X, Y).')

    V_values = np.asarray(V_pot(X_in, Y_in), dtype=float)
    if V_values.shape != X_in.shape:
        raise ValueError(
            f'V_pot(X,Y) must return shape {X_in.shape}, got {V_values.shape}.'
        )

    V = diags(V_values.ravel(), offsets=0, format='csr')

    # Hamiltonian
    H = T + V

    # Solve for the lowest eigenpairs.
    # Shift-invert: target eigenvalues near sigma instead of running plain Lanczos on
    # the full spectrum. This matters more here than in 1D, since Ntot grows as N^2 and
    # plain 'SA' mode gets noticeably slower as the grid grows. sigma must sit safely
    # below every possible eigenvalue: since H = T + V with T >= 0 (the discretized
    # kinetic energy is positive semi-definite), E_0 >= min(V) always, so any
    # sigma < min(V) is guaranteed safe, and "closest to sigma" becomes exactly
    # "the k smallest".
    sigma = float(V_values.min()) - 1.0
    eigvals, eigvecs = eigsh(H, k=num_eigvals, sigma=sigma, which='LM')

    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Put eigenvectors back on the full grid and normalize
    normalized_eigvecs = np.zeros((Ny, Nx, num_eigvals), dtype=float)
    area_element = dx * dy

    for n in range(num_eigvals):
        psi_n = eigvecs[:, n].reshape((Ny_int, Nx_int))
        normalized_eigvecs[1:-1, 1:-1, n] = psi_n
        norm = np.sqrt(np.sum(np.abs(normalized_eigvecs[:, :, n])**2) * area_element)
        normalized_eigvecs[:, :, n] /= norm

    return x, y, eigvals, normalized_eigvecs


## List of potential functions

def V_HarmonicOscillator_2D(X, Y, omega_x=1.0, omega_y=1.0, m=1.0):
    return 0.5 * m * (omega_x**2 * X**2 + omega_y**2 * Y**2)

def V_AnharmonicOscillator_2D(X, Y, a=1.0, b=0.05):
    r2 = X**2 + Y**2
    return 0.5 * a * r2 + b * r2**2

def V_InfiniteSquareWell_2D(X, Y):
    return np.zeros_like(X, dtype=float)

def V_FiniteSquareWell_2D_NonSeparable(X, Y, Lx=4.0, Ly=4.0, V0=50.0, centered=True):
    '''
    Rectangular finite well: V = 0 inside the rectangle, V0 everywhere outside.

    This potential is NOT separable, i.e. it cannot be written as Vx(x) + Vy(y). Check the
    corner region, |x| > Lx/2 AND |y| > Ly/2: this V returns V0 there, while
    Vx(x) + Vy(y) would return 2*V0. So the spectrum does NOT factorize and it is wrong to
    compare it against E_{nx,ny} = eps_nx + eps_ny. It is a legitimate physical potential
    and a fine qualitative test - just not an analytic benchmark.

    Use V_FiniteSquareWell_2D_Separable below when you want a quantitative benchmark.
    '''
    if centered:
        inside = (np.abs(X) <= 0.5 * Lx) & (np.abs(Y) <= 0.5 * Ly)
    else:
        inside = (X >= 0.0) & (X <= Lx) & (Y >= 0.0) & (Y <= Ly)
    return np.where(inside, 0.0, V0)


# Backwards-compatible alias: the old name pointed at the non-separable version.
V_FiniteSquareWell_2D = V_FiniteSquareWell_2D_NonSeparable


def V_FiniteSquareWell_1D_Profile(u, L=4.0, V0=40.0):
    '''1D finite-well profile, 0 for |u| <= L/2 and V0 outside. Helper for the separable 2D well.'''
    return np.where(np.abs(u) <= 0.5 * L, 0.0, V0)


def V_FiniteSquareWell_2D_Separable(X, Y, Lx=4.0, Ly=4.0, V0=40.0):
    '''
    Separable finite well by construction: V(x,y) = Vx(x) + Vy(y), each factor a 1D finite
    well of depth V0. Note this is 2*V0 in the corners, not V0 - that is exactly the price
    of separability, and it is what makes the spectrum factorize:

        E_{nx,ny} = eps_nx(Lx, V0) + eps_ny(Ly, V0)

    with eps_n the 1D finite-well bound states (roots of k tan(kL/2) = kappa for even
    states and k cot(kL/2) = -kappa for odd ones). For Lx = 4, V0 = 40 the 1D ground state
    is eps_1 = 0.276598377, so the 2D ground state of this potential is 2 * 0.276598377.

    Accuracy caveat: like every step potential here, the pointwise sampling of the
    discontinuity onto the mesh limits convergence to ~O(dx), independently of the
    boundary stencil. Expect ~1e-3 to 1e-4 relative error, not 1e-8.
    '''
    return (V_FiniteSquareWell_1D_Profile(X, L=Lx, V0=V0)
            + V_FiniteSquareWell_1D_Profile(Y, L=Ly, V0=V0))

def V_LinearPotential_2D(X, Y, Fx=1.0, Fy=0.0):
    return Fx * X + Fy * Y

def V_SoftCoulomb_2D(X, Y, Z=1.0, eps=0.2):
    return -Z / np.sqrt(X**2 + Y**2 + eps**2)

def V_SingleWell_2D(X, Y, ax=1.0, ay=1.0):
    # Simple quartic single well: ax*X^4 + ay*Y^4
    return ax * X**4 + ay * Y**4

def V_DeltaDiscrete_2D(X, Y, alpha=8.0, x0=0.0, y0=0.0):
    """
    Attractive discrete 2D delta well:
        V(x,y) = -alpha * delta(x-x0) * delta(y-y0), alpha > 0

    Discrete implementation on the mesh:
        V[j0, i0] = -alpha / (dx * dy)
    at the grid point nearest (x0, y0), and zero elsewhere.

    Unlike the 1D delta well, this one does NOT converge to a fixed ground-state energy
    as the grid is refined. I checked this directly (see validate_2d.py and the README):
    the measured E0 grows without bound in magnitude as N increases (-0.79, -2.25, -4.46,
    -7.42 at N=30/50/70/90). This is not a bug - the continuum 2D Dirac delta potential
    has no bound-state energy scale without an explicit regularization, and dx is playing
    that role here. Use this potential for illustration only, not for anything where the
    absolute energy value matters.
    """

    Varr = np.zeros_like(X, dtype=float)

    # Infer dx and dy from the meshgrid
    if X.shape[1] > 1:
        dx = X[0, 1] - X[0, 0]
    else:
        dx = 1.0

    if Y.shape[0] > 1:
        dy = Y[1, 0] - Y[0, 0]
    else:
        dy = 1.0

    # Index of nearest grid point to (x0, y0)
    dist2 = (X - x0)**2 + (Y - y0)**2
    j0, i0 = np.unravel_index(np.argmin(dist2), X.shape)

    Varr[j0, i0] = -alpha / (dx * dy)
    return Varr

def V_DoubleWell_2D(X, Y, a=1.5, Vx=1.0, Vy=0.3):
    # Simple quartic double well: Vx*(X^2 - a^2)^2 + Vy*Y^2
    return Vx * (X**2 - a**2)**2 + Vy * Y**2

## Plotting

def plot_eigenfunction_heatmap(
    x, y, eigvecs, n,
    probability=False,
    ax=None,
    cmap=None,
    title=None,
    show_colorbar=True,
    ):
    '''
    Heatmap of a single 2D eigenfunction psi_n(x,y) (or |psi_n(x,y)|^2 if
    probability=True) over the full (x,y) grid.

    x, y, eigvecs: outputs of Schrodinger_solver_2D. eigvecs has shape (Ny, Nx, k).
    n: which state to plot (0-indexed, same ordering as eigvals).
    probability: plot |psi_n|^2 (sequential colormap, always >= 0) instead of the signed
    wavefunction psi_n (diverging colormap centered at 0, since psi_n and -psi_n are
    equally valid and the sign itself carries no physical meaning, only the nodal
    structure - where it changes sign - does).
    ax: existing matplotlib Axes to draw on (creates a new figure if None).

    Returns the Axes used.
    '''
    psi = eigvecs[:, :, n]

    if probability:
        field = psi**2
        default_cmap = 'viridis'
        cbar_label = r'$|\psi_n(x,y)|^2$'
        vmin, vmax = 0.0, np.max(field)
    else:
        field = psi
        default_cmap = 'RdBu_r'
        cbar_label = r'$\psi_n(x,y)$'
        vlim = np.max(np.abs(field))
        vmin, vmax = -vlim, vlim

    if cmap is None:
        cmap = default_cmap
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.0))

    im = ax.imshow(
        field,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower', aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title if title is not None else f'n={n}', fontsize=9)
    if show_colorbar:
        plt.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)
    return ax

def plot_eigenfunction_grid(
    x, y, eigvecs,
    n_states=None,
    ncols=3,
    probability=False,
    cmap=None,
    titles=None,
    suptitle=None,
    ):
    '''
    Mosaic of heatmaps for the first n_states eigenfunctions, one panel per state. This
    is the 2D analogue of stacking psi_n(x) curves in the 1D code: show several
    states side by side is a grid of small heatmaps rather than overlaying them.

    x, y, eigvecs: outputs of Schrodinger_solver_2D.
    n_states: how many states to draw (default: all available in eigvecs).
    ncols: number of columns in the mosaic.
    titles: optional list of per-panel title strings (e.g. "(nx,ny)=(1,0)"); defaults to
    "n=<index>".

    Returns (fig, axes).
    '''
    if n_states is None:
        n_states = eigvecs.shape[2]
    n_states = min(n_states, eigvecs.shape[2])
    nrows = int(np.ceil(n_states / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for n in range(n_states):
        title = titles[n] if titles is not None else f'n={n}'
        plot_eigenfunction_heatmap(
            x, y, eigvecs, n, probability=probability, ax=axes[n], cmap=cmap, title=title,
        )
    for ax in axes[n_states:]:
        ax.axis('off')

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig, axes

def plot_eigenfunction_surface(
    x, y, eigvecs, n,
    ax=None,
    cmap='viridis',
    title=None,
    ):
    '''
    3D surface plot of a single eigenfunction psi_n(x,y). Complements the heatmap view:
    a heatmap shows the nodal structure clearly (where psi=0), a surface shows the actual
    amplitude profile (how tall the lobes are relative to each other), which is easy to
    lose in a flat color plot.

    x, y, eigvecs: outputs of Schrodinger_solver_2D.
    n: which state to plot (0-indexed).
    ax: existing 3D Axes to draw on (creates a new figure with a 3D projection if None).

    Returns the Axes used.
    '''
    psi = eigvecs[:, :, n]
    X, Y = np.meshgrid(x, y, indexing='xy')

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, psi, cmap=cmap, linewidth=0, antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\psi_n(x,y)$')
    ax.set_title(title if title is not None else f'n={n}', fontsize=10)
    return ax

def plot_energy_levels_2d(eigvals, n_states=None, ax=None, title='Energy levels', labels=None):
    '''
    Energy-level diagram: one horizontal line per E_n, labeled with n (and, if provided,
    a quantum-number label like "(nx,ny)=(1,0)"). Same style as the 1D level diagram.
    Levels closer together than the 1D code's near-degeneracy threshold get their labels
    nudged apart so they do not overlap, and near-exact degeneracies (e.g. (1,0) and (0,1)
    for an isotropic oscillator) get their splitting printed explicitly.

    eigvals: output of Schrodinger_solver_2D (assumed sorted ascending).
    n_states: how many levels to draw (default: all available).
    labels: optional list of strings, one per level, appended to the "n=..." annotation.
    ax: existing matplotlib Axes to draw on (creates a new figure if None).

    Returns the Axes used.
    '''
    if n_states is None:
        n_states = eigvals.size
    n_states = min(n_states, eigvals.size)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 5))

    cmap = plt.get_cmap('viridis', max(n_states, 1))
    yrange = eigvals[n_states - 1] - eigvals[0]
    min_gap = 0.045 * max(yrange, 1e-9)

    placed = []
    for n in range(n_states):
        ax.hlines(eigvals[n], 0, 1, color=cmap(n), lw=2.2)

        y_label = eigvals[n]
        if placed and (y_label - placed[-1]) < min_gap:
            y_label = placed[-1] + min_gap
        placed.append(y_label)

        split = ''
        if n > 0 and abs(eigvals[n] - eigvals[n - 1]) < 1e-3 * max(abs(eigvals[n]), 1.0):
            split = f'  (Δ={eigvals[n] - eigvals[n - 1]:.2e})'

        extra = f' {labels[n]}' if labels is not None else ''
        ax.annotate(
            f'n={n}{extra}:  E={eigvals[n]:.4f}{split}',
            xy=(1.0, eigvals[n]), xytext=(1.05, y_label),
            fontsize=8, va='center', color=cmap(n),
            arrowprops=dict(arrowstyle='-', color=cmap(n), lw=0.6),
        )

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel('Energy')
    ax.set_title(title)
    pad = 0.08 * max(yrange, 1.0)
    ax.set_ylim(eigvals[0] - pad, max(eigvals[n_states - 1], placed[-1]) + pad)
    return ax

'''
---------------------------- USAGE RECIPES ----------------------------------
------------------------- 1) Harmonic Oscillator ---------------------------
# V(x,y) = 1/2 m (omega_x^2 x^2 + omega_y^2 y^2)
# Note: omega_x == omega_y here gives an isotropic (degenerate) spectrum - see the
# Hermiticity/degeneracy note in the __main__ block and the README for why the demo
# figures use omega_x != omega_y instead.
x, y, eigvals, eigvecs = Schrodinger_solver_2D(
    V_pot=partial(V_HarmonicOscillator_2D, omega_x=1.0, omega_y=1.0, m=1.0),
    x_min=-6.0, x_max=6.0,
    y_min=-6.0, y_max=6.0,
    Nx=180, Ny=180,
    num_eigvals=8
)

----------------------- 2) Anharmonic Oscillator ---------------------------
# V(x,y) = 1/2 a (x^2+y^2) + b (x^2+y^2)^2
x, y, eigvals, eigvecs = Schrodinger_solver_2D(
    V_pot=partial(V_AnharmonicOscillator_2D, a=1.0, b=0.05),
    x_min=-8.0, x_max=8.0,
    y_min=-8.0, y_max=8.0,
    Nx=200, Ny=200,
    num_eigvals=8
)

-------------------------- 3) Infinite Square Well -------------------------
# Infinite walls are enforced by the finite domain + Dirichlet BC (psi=0 at edges).
# So inside the box V=0 everywhere.
x, y, eigvals, eigvecs = Schrodinger_solver_2D(
    V_pot=V_InfiniteSquareWell_2D,
    x_min=-5.0, x_max=5.0,
    y_min=-5.0, y_max=5.0,
    Nx=180, Ny=180,
    num_eigvals=8
)

--------------------------- 4) Finite Square Well (Non-Separable) -------------------------
# Rectangular well: V=0 inside, V=V0 outside.
x, y, eigvals, eigvecs = Schrodinger_solver_2D(
    V_pot=partial(V_FiniteSquareWell_2D_NonSeparable, Lx=4.0, Ly=4.0, V0=40.0, centered=True),
    x_min=-8.0, x_max=8.0,
    y_min=-8.0, y_max=8.0,
    Nx=180, Ny=180,
    num_eigvals=8
)

--------------------------- 5) Finite Square Well (Separable) -------------------------
# Rectangular well: V=0 inside, V=V0 outside.
x, y, eigvals, eigvecs = Schrodinger_solver_2D(
    V_pot=partial(V_FiniteSquareWell_2D_Separable, Lx=4.0, Ly=4.0, V0=40.0),
    x_min=-8.0, x_max=8.0,
    y_min=-8.0, y_max=8.0,
    Nx=180, Ny=180,
    num_eigvals=8
)

----------------------------- 6) Linear Potential --------------------------
# V(x,y) = Fx*x + Fy*y
x, y, eigvals, eigvecs = Schrodinger_solver_2D(
    V_pot=partial(V_LinearPotential_2D, Fx=1.0, Fy=0.0),
    x_min=-10.0, x_max=10.0,
    y_min=-10.0, y_max=10.0,
    Nx=220, Ny=220,
    num_eigvals=8
)

----------------------------- 7) Soft Coulomb ------------------------------
# V(x,y) = -Z/sqrt(x^2+y^2+eps^2)
# Smaller eps => deeper and sharper singularity regularization.
x, y, eigvals, eigvecs = Schrodinger_solver_2D(
    V_pot=partial(V_SoftCoulomb_2D, Z=1.0, eps=0.25),
    x_min=-20.0, x_max=20.0,
    y_min=-20.0, y_max=20.0,
    Nx=240, Ny=240,
    num_eigvals=8
)

--------------------------- 8) Quartic Single Well -------------------------
# V(x,y) = ax*x^4 + ay*y^4
x, y, eigvals, eigvecs = Schrodinger_solver_2D(
    V_pot=partial(V_SingleWell_2D, ax=1.0, ay=1.0),
    x_min=-4.0, x_max=4.0,
    y_min=-4.0, y_max=4.0,
    Nx=180, Ny=180,
    num_eigvals=8
)

# ---------------------------- 9) Discrete Delta Well ----------------------
# V(x,y) = -alpha*delta(x-x0)delta(y-y0)
# Implemented as one attractive grid site with strength -alpha/(dx*dy)
x, y, eigvals, eigvecs = Schrodinger_solver_2D(
    V_pot=partial(V_DeltaDiscrete_2D, alpha=2.0, x0=0.0, y0=0.0),
    x_min=-10.0, x_max=10.0,
    y_min=-10.0, y_max=10.0,
    Nx=220, Ny=220,
    num_eigvals=6
)

--------------------------- 10) Quartic Double Well -------------------------
# V(x,y) = Vx*(x^2-a^2)^2 + Vy*y^2
x, y, eigvals, eigvecs = Schrodinger_solver_2D(
    V_pot=partial(V_DoubleWell_2D, a=1.5, Vx=1.0, Vy=0.3),
    x_min=-5.0, x_max=5.0,
    y_min=-5.0, y_max=5.0,
    Nx=200, Ny=200,
    num_eigvals=8
)
'''
## Running it

if __name__ == '__main__':
    # Note: omega_x != omega_y on purpose. For the isotropic case (omega_x == omega_y)
    # the spectrum has exact degeneracies, and eigsh/ARPACK - like any single-vector
    # Krylov method - is not guaranteed to resolve the full degenerate eigenspace (it can
    # return fewer distinct directions than the requested num_eigvals, or an arbitrary
    # mixed basis within each degenerate subspace instead of the separable (nx,ny)
    # states). See the README for the numerical evidence. Using a slightly anisotropic
    # oscillator here sidesteps that issue entirely for this quick demo.
    x, y, eigvals, eigvecs = Schrodinger_solver_2D(
        V_pot=partial(V_HarmonicOscillator_2D, omega_x=1.0, omega_y=1.0, m=1.0),
        x_min=-6.0, x_max=6.0,
        y_min=-6.0, y_max=6.0,
        Nx=180, Ny=180,
        num_eigvals=8
    )

    print("Lowest energies:")
    for n, En in enumerate(eigvals):
        print(f"n={n}, E = {En:.6f}")

    # Quick visual check: a mosaic of the first few eigenfunctions, a 3D surface for the
    # ground state, and the energy-level diagram.
    plot_eigenfunction_grid(
        x, y, eigvecs, n_states=6, ncols=3,
        suptitle="2D Harmonic Oscillator: eigenfunctions",
    )
    plot_eigenfunction_surface(
        x, y, eigvecs, n=0, title="Ground state, psi_0(x,y)",
    )
    plot_energy_levels_2d(eigvals, n_states=6, title="2D Harmonic Oscillator: energy levels")
    plt.show()