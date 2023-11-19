from cr_autophagy_pyo3 import SimulationSettings, run_simulation

simulation_settings = SimulationSettings()

# Settings Cargo
simulation_settings.n_cells_cargo=300
simulation_settings.particle_template_cargo.interaction.strength_repell = 0.01
simulation_settings.particle_template_cargo.interaction.strength_attract = 0.01
simulation_settings.particle_template_cargo.interaction.interaction_range = 1.0
simulation_settings.particle_template_cargo.interaction.cell_radius = 1.0
simulation_settings.particle_template_cargo.interaction.clustering_strength = 0.3
simulation_settings.particle_template_cargo.mechanics.damping = 1.5
simulation_settings.particle_template_cargo.mechanics.mass = 4/3*3.1415
simulation_settings.particle_template_cargo.mechanics.kb_temperature = 0.00001

# Settings Atg11/Receptor
simulation_settings.n_cells_atg11_receptor=600
simulation_settings.particle_template_atg11_receptor.interaction.strength_repell = 0.01
simulation_settings.particle_template_atg11_receptor.interaction.strength_attract = 0.01
simulation_settings.particle_template_atg11_receptor.interaction.interaction_range = 0.5
simulation_settings.particle_template_atg11_receptor.interaction.cell_radius = 1.0
simulation_settings.particle_template_atg11_receptor.interaction.clustering_strength = 0.3
simulation_settings.particle_template_atg11_receptor.mechanics.damping = 1.5
simulation_settings.particle_template_atg11_receptor.mechanics.mass = 4/3*3.1415
simulation_settings.particle_template_atg11_receptor.mechanics.kb_temperature = 0.01

# General Settings
simulation_settings.n_threads = 1
simulation_settings.dt = 2.0
simulation_settings.domain_size = 100
simulation_settings.domain_interaction_range = simulation_settings.domain_size / 8
simulation_settings.n_times = 100_001
simulation_settings.save_interval = 250
simulation_settings.show_progressbar = True

if __name__ == "__main__":
    output_path = run_simulation(simulation_settings)
else:
    import os
    from pathlib import Path
    output_path = Path("out/autophagy/") / os.listdir("out/autophagy/")[-1]
