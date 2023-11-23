from cr_autophagy_pyo3 import SimulationSettings, run_simulation

simulation_settings = SimulationSettings()

# Settings Cargo
simulation_settings.n_cells_cargo=100
simulation_settings.particle_template_cargo.interaction.avidity = 1.0

# Settings Atg11/Receptor
simulation_settings.n_cells_atg11_receptor=700
simulation_settings.particle_template_atg11_receptor.interaction.avidity = 1.0
simulation_settings.particle_template_atg11_receptor.mechanics.kb_temperature = 0.0025

# Time
simulation_settings.n_threads = 10
simulation_settings.dt = 2.0
simulation_settings.n_times = 30_001
simulation_settings.save_interval = 1000

# Domain
simulation_settings.domain_size = 40
simulation_settings.domain_size_cargo = 11
simulation_settings.domain_n_voxels = 5

# Other settings
simulation_settings.show_progressbar = True

if __name__ == "__main__":
    output_path = run_simulation(simulation_settings)
