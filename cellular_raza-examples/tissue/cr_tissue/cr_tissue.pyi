import numpy as np

class SimulationSettings:
    n_plants: int
    domain_size: float
    n_voxels: int
    n_threads: int
    dt: float
    t_max: float
    save_interval: float
    rng_seed: int
    target_area: float
    force_strength: float
    force_strength_weak: float
    force_strength_species: float
    force_relative_cutoff: float
    potential_stiffness: float
    damping_constant: float
    cell_diffusion_constant: float

def run_simulation(
    sim_settings: SimulationSettings,
    plant_points: np.ndarray,
    plant_species: list[int],
): ...
