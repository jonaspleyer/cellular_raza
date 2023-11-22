from cr_autophagy_pyo3 import SimulationSettings, run_simulation

simulation_settings = SimulationSettings()

# Settings Cargo
simulation_settings.n_cells_cargo=300
# simulation_settings.particle_template_cargo.interaction.potential_strength = 0.01

# Settings Atg11/Receptor
simulation_settings.n_cells_atg11_receptor=1500
# simulation_settings.particle_template_atg11_receptor.interaction.potential_strength = 0.01
# simulation_settings.particle_template_atg11_receptor.mechanics.kb_temperature = 0.01

# Time
simulation_settings.n_threads = 6
simulation_settings.dt = 1.0
simulation_settings.n_times = 50_001
simulation_settings.save_interval = 50

# Domain
simulation_settings.domain_size = 60
simulation_settings.domain_size_cargo = 16
simulation_settings.domain_n_voxels = 7

# Other settings
simulation_settings.show_progressbar = True

if __name__ == "__main__":
    output_path = run_simulation(simulation_settings)
