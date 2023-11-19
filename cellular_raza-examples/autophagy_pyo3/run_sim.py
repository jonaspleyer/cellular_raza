from cr_autophagy_pyo3 import SimulationSettings, run_simulation

simulation_settings = SimulationSettings()

# Settings Cargo
simulation_settings.n_cells_cargo=10
simulation_settings.particle_template_cargo.interaction.epsilon = 0.1
simulation_settings.particle_template_cargo.interaction.bound = 0.01
simulation_settings.particle_template_cargo.interaction.cutoff = 4.0
simulation_settings.particle_template_cargo.interaction.cell_radius = 1.0
simulation_settings.particle_template_cargo.interaction.clustering_strength = 1.1
simulation_settings.particle_template_cargo.mechanics.damping = 1.5
simulation_settings.particle_template_cargo.mechanics.mass = 4/3*3.1415
simulation_settings.particle_template_cargo.mechanics.kb_temperature = 0.0001

# Settings Atg11/Receptor
simulation_settings.n_cells_atg11_receptor=20
simulation_settings.particle_template_atg11_receptor.interaction.epsilon = 1.0
simulation_settings.particle_template_atg11_receptor.interaction.bound = 0.03
simulation_settings.particle_template_atg11_receptor.interaction.cutoff = 4.0
simulation_settings.particle_template_atg11_receptor.interaction.cell_radius = 1.0
simulation_settings.particle_template_atg11_receptor.interaction.clustering_strength = 1.1
simulation_settings.particle_template_atg11_receptor.mechanics.damping = 1.5
simulation_settings.particle_template_atg11_receptor.mechanics.mass = 4/3*3.1415
simulation_settings.particle_template_atg11_receptor.mechanics.kb_temperature = 0.002

# General Settings
simulation_settings.n_threads = 1
simulation_settings.dt = 0.5
simulation_settings.domain_size = 20
simulation_settings.domain_interaction_range = simulation_settings.domain_size / 8
simulation_settings.n_times = 1_000_001
simulation_settings.save_interval = 500
simulation_settings.show_progressbar = True

if __name__ == "__main__":
    output_path = run_simulation(simulation_settings)
