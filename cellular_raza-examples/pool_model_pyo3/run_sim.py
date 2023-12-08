import numpy as np
import scipy as sp

import cr_pool_model_pyo3 as crp_py







if __name__ == "__main__":
    # Domain Settings
    domain = crp_py.Domain()
    domain.size = 1_000
    domain.diffusion_constants = [15.0, 5.0]

    # Meta Parameters
    meta_params = crp_py.MetaParams()
    meta_params.save_interval = 500
    meta_params.n_times = 20_001
    meta_params.n_threads = 8

    cells = generate_cells(18, 18, domain)

    output_path = crp_py.run_simulation(
        cells,
        domain,
        meta_params,
    )
