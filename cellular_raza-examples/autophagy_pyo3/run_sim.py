from cr_autophagy_pyo3 import SimulationSettings, run_simulation

simulation_settings = SimulationSettings()

# Settings Cargo
simulation_settings.n_cells_cargo=3
simulation_settings.particle_template_cargo.interaction.bound = 0.02
simulation_settings.particle_template_cargo.interaction.well_depth = 0.01
simulation_settings.particle_template_cargo.interaction.potential_width = 1/10
simulation_settings.particle_template_cargo.interaction.cell_radius = 1.0
simulation_settings.particle_template_cargo.interaction.cutoff = simulation_settings.particle_template_cargo.interaction.cell_radius + simulation_settings.particle_template_cargo.interaction.potential_width
simulation_settings.particle_template_cargo.interaction.avidity = 1.0

simulation_settings.particle_template_cargo.mechanics.damping = 1.5
simulation_settings.particle_template_cargo.mechanics.mass = 4/3*3.1415
simulation_settings.particle_template_cargo.mechanics.kb_temperature = 0.00001

# Settings Atg11/Receptor
simulation_settings.n_cells_atg11_receptor=1
simulation_settings.particle_template_atg11_receptor.interaction.bound = 0.02
simulation_settings.particle_template_atg11_receptor.interaction.well_depth = 0.01
simulation_settings.particle_template_atg11_receptor.interaction.potential_width = 0.5/10
simulation_settings.particle_template_atg11_receptor.interaction.cell_radius = 0.5
simulation_settings.particle_template_atg11_receptor.interaction.cutoff = simulation_settings.particle_template_atg11_receptor.interaction.cell_radius + simulation_settings.particle_template_atg11_receptor.interaction.potential_width
simulation_settings.particle_template_atg11_receptor.interaction.avidity = 1.0

simulation_settings.particle_template_atg11_receptor.mechanics.damping = 0.5
simulation_settings.particle_template_atg11_receptor.mechanics.mass = 4/3*3.1415*0.5**3
simulation_settings.particle_template_atg11_receptor.mechanics.kb_temperature = 0.0075

# General Settings
simulation_settings.n_threads = 1
simulation_settings.dt = 0.5
simulation_settings.domain_size = 20
simulation_settings.domain_n_voxels = 5
simulation_settings.n_times = 10_001
simulation_settings.save_interval = 20
simulation_settings.show_progressbar = True

if __name__ == "__main__":
    output_path = run_simulation(simulation_settings)
else:
    import os
    from pathlib import Path
    output_path = Path("out/autophagy/") / os.listdir("out/autophagy/")[-1]
