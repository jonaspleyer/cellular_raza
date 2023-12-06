import numpy as np
import matplotlib.pyplot as plt

import cr_pool_model_pyo3 as crp_py

# simulation_settings = SimulationSettings()

domain = crp_py.Domain()
meta_params = crp_py.MetaParams()

n_cells_1 = 5
n_cells_2 = 5

cells = []
for i in range(n_cells_1 + n_cells_2):
    cell = crp_py.BacteriaTemplate()
    cell.mechanics.pos = np.random.uniform(0.0, domain.size, [2])
    if i > n_cells_1:
        cell.cellular_reactions.species = crp_py.Species.S1

    cells.append(cell)

output_path = crp_py.run_simulation(
    cells,
    domain,
    meta_params,
)
