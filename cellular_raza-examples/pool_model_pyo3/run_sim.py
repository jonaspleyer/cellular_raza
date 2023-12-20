import numpy as np
import scipy as sp

import cr_pool_model_pyo3 as crp_py


def calculate_lattice_points(x_min, x_max, n_agents):
    # Calculate the lowest number possible such that
    # we can create a almost even lattice based on that
    m = np.ceil(np.sqrt(n_agents)) + 1
    dx = (x_max - x_min) / m

    x, y = np.mgrid[x_min+dx:x_max:dx, x_min+dx:x_max:dx]
    if m % 2 == 0:
        z = np.vstack([
            x.reshape(-1),
            y.reshape(-1)]
        ).T
    else:
        for i in range(len(y)):
            y[i] = np.roll(y[i], -(i % 2))
        z = np.vstack([
            x.reshape(-1),
            y.reshape(-1)
        ]).T
    return z


def calculate_index_distributions(n_cells_1, n_cells_2, n_positions, homogenous=True):
    if homogenous:
        indices = np.arange(n_positions)

        ind_even = indices[::2]
        ind_uneven = indices[1::2]

        ind1_even = ind_even[:n_cells_1]
        ind1_uneven = ind_uneven[:n_cells_1-len(ind1_even)]

        ind1 = np.sort(np.hstack([ind1_even, ind1_uneven]))
        ind2 = np.sort(np.setdiff1d(indices, ind1))[:n_cells_2]

        return ind1, ind2
    else:
        indices = np.arange(n_positions)
        ind1 = indices[:n_cells_1]
        ind2 = indices[n_cells_1:]
        return ind1, ind2


def generate_cells(n_cells_1, n_cells_2, domain, randomness=0.0, pad=0.15, seed=0):
    """
    n_cells_1: int
    n_cells_2: int
    uniformity: float
        Floating point number between 0.0 and 1.0
    """
    # Fix numpy random seed
    rng = np.random.default_rng(seed)

    # Get the domain size
    d_min =    pad *domain.size
    d_max = (1-pad)*domain.size
    r = np.clip(randomness, 0, 1)

    positions_random = d_min + rng.random((n_cells_1+n_cells_2, 2))*(d_max - d_min)

    positions_lattice = calculate_lattice_points(0.0, domain.size, n_cells_1 + n_cells_2)
    positions = positions_random*r + (1-r)*positions_lattice
    ind1, ind2 = calculate_index_distributions(n_cells_1, n_cells_2, len(positions), homogenous=False)

    cells = []
    for i in range(n_cells_1 + n_cells_2):
        # Cell Settings
        cell = crp_py.BacteriaTemplate()
        cell.cycle.lag_phase_transition_rate_1 = 0.005
        cell.cycle.lag_phase_transition_rate_2 = 0.0025

        if i < n_cells_1:
            # x = rng.uniform(d_min, (1-u)*d_min + u*d_max)
            cell.mechanics.pos = positions[ind1[i]]
        else:
            # x = rng.uniform(u*d_min + (1-u)*d_max, d_max)
            cell.cellular_reactions.species = crp_py.Species.S2
            cell.mechanics.pos = positions[ind2[i - n_cells_1]]

        # y = rng.uniform(d_min, d_max)
        # cell.mechanics.pos = [x, y]
        # cell.mechanics.pos = positions[i]

        cells.append(cell)
    return cells


if __name__ == "__main__":
    # Domain Settings
    domain = crp_py.Domain()
    domain.size = 1_000
    domain.diffusion_constants = [5.0, 5.0]

    # Meta Parameters
    meta_params = crp_py.MetaParams()
    meta_params.save_interval = 1_000
    meta_params.n_times = 40_001
    meta_params.dt = 0.25
    meta_params.n_threads = 8

    cells = generate_cells(18, 18, domain, randomness=0.0)

    output_path = crp_py.run_simulation(
        cells,
        domain,
        meta_params,
    )
