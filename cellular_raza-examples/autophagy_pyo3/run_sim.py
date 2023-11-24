from cr_autophagy_pyo3 import SimulationSettings, run_simulation
import cr_autophagy as cra

simulation_settings = SimulationSettings()

# Settings Cargo
simulation_settings.n_cells_cargo=10
simulation_settings.cell_radius_cargo = 1.5

# Settings R11
simulation_settings.n_cells_r11=1
simulation_settings.cell_radius_r11 = 1.0
simulation_settings.potential_strength_r11_r11 = 0.001
simulation_settings.kb_temperature_r11 = 0.0

simulation_settings.interaction_range_r11_cargo = 2 * simulation_settings.cell_radius_cargo
simulation_settings.potential_strength_cargo_r11 = 0.02

# Time
simulation_settings.n_threads = 1
simulation_settings.dt = 1.0
simulation_settings.n_times = 10_001
simulation_settings.save_interval = 100

# Domain
ds = 40
simulation_settings.domain_size = ds
cs = 4
xs = 0.25
simulation_settings.domain_r11_low =    [ds/2 - cs, ds/2 - xs - 2*cs, ds/2 - cs]
simulation_settings.domain_r11_high =   [ds/2 + cs, ds/2 - xs,        ds/2 + cs]
simulation_settings.domain_cargo_low =  [ds/2 - cs, ds/2 + xs,        ds/2 - cs]
simulation_settings.domain_cargo_high = [ds/2 + cs, ds/2 + xs + 2*cs, ds/2 + cs]

# Other settings
simulation_settings.show_progressbar = True

if __name__ == "__main__":
    from pathlib import Path
    import os
    output_path = Path(run_simulation(simulation_settings))

    # print("Saving scatter Snapshots")
    # cra.save_all_scatter_snapshots(output_path, threads=-1)#simulation_settings.n_threads)

    # Create movie with ffmpeg
    # print("Generating Scatter Snapshot Movie")
    # bashcmd = f"ffmpeg -v quiet -stats -y -r 30 -f image2 -pattern_type glob -i '{output_path}/scatterplots/*.png' -c:v h264 -pix_fmt yuv420p -strict -2 {output_path}/scatter_movie.mp4"
    # os.system(bashcmd)

    print("Saving Snapshots")
    cra.save_all_snapshots(output_path, threads=14)#simulation_settings.n_threads)

    # Also create a movie with ffmpeg
    print("Generating Snapshot Movie")
    bashcmd = f"ffmpeg -v quiet -stats -y -r 30 -f image2 -pattern_type glob -i '{output_path}/snapshots/*.png' -c:v h264 -pix_fmt yuv420p -strict -2 {output_path}/snapshot_movie.mp4"
    os.system(bashcmd)
