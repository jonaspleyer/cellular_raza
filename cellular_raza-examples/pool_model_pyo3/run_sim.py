from cr_pool_model_pyo3 import *

simulation_settings = SimulationSettings()

# Domain
simulation_settings.domain.size = 1_000

simulation_settings.domain.initial_concentrations = [2.0, 0]
simulation_settings.domain.diffusion_constants = [1.0, 0]

# Bacteria - General
simulation_settings.n_bacteria_initial = 250

# Bacteria - Interaction & Mechanics
simulation_settings.bacteria_mechanics.kb_temperature = 0.01
simulation_settings.bacteria_interaction.potential_strength = 0.5

# Bacteria - Reactions
simulation_settings.bacteria_reactions.uptake_rates = [0.01, 0.0]

# Bacteria - Cycle
simulation_settings.bacteria_cycle.lag_phase_transition_rate = 0.0001
simulation_settings.bacteria_cycle.food_to_volume_conversion = 1e-5

# General Settings
simulation_settings.n_threads = 4
simulation_settings.dt = 0.5
simulation_settings.n_times = 40_001
simulation_settings.save_interval = 250

if __name__ == "__main__":
    output_path = run_simulation(simulation_settings)
else:
    import os
    from pathlib import Path
    output_path = Path("out/pool_model/") / sorted(os.listdir("out/pool_model/"))[-1]
