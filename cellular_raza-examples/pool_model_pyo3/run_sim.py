from cr_pool_model_pyo3 import *

simulation_settings = SimulationSettings()

# Domain
simulation_settings.domain_size = 200
simulation_settings
simulation_settings.voxel_food_initial_concentration = 10.0
simulation_settings.voxel_food_diffusion_constant = 20.0

# Starting Domain
simulation_settings.starting_domain_x_low = 90
simulation_settings.starting_domain_x_high = 110
simulation_settings.starting_domain_y_low = 90
simulation_settings.starting_domain_y_high = 110

# Bacteria - General
simulation_settings.n_bacteria_initial = 2

# Bacteria - Interaction & Mechanics
simulation_settings.bacteria_mechanics.kb_temperature = 0.01

# Bacteria - Reactions
simulation_settings.bacteria_reactions.uptake_rate = [0.002]
simulation_settings.bacteria_reactions.intracellular_concentrations = [0]

# Bacteria - Cycle
simulation_settings.bacteria_cycle.lack_phase_transition_rate = 0.00001

# General Settings
simulation_settings.n_threads = 1
simulation_settings.dt = 0.2
simulation_settings.n_times = 60_001
simulation_settings.save_interval = 50

if __name__ == "__main__":
    output_path = run_simulation(simulation_settings)
else:
    import os
    from pathlib import Path
    output_path = Path("out/pool_model/") / sorted(os.listdir("out/pool_model/"))[-1]
