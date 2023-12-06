from cr_autophagy_pyo3 import SimulationSettings, run_simulation
import cr_autophagy as cra

simulation_settings = SimulationSettings()

# HIGH AFFINITY CONFIGURATION
simulation_settings.potential_strength_cargo_r11 = 0.01
simulation_settings.potential_strength_cargo_r11_avidity = 0.00

# HIGH AVIDITY CONFIGURATION
# simulation_settings.potential_strength_cargo_r11 = 0.0
# simulation_settings.potential_strength_cargo_r11_avidity = 0.01

if __name__ == "__main__":
    from pathlib import Path
    import os
    output_path = Path(run_simulation(simulation_settings))

    print("Saving Snapshots")
    cra.save_all_snapshots(output_path, threads=14)#simulation_settings.n_threads)

    # Also create a movie with ffmpeg
    print("Generating Snapshot Movie")
    bashcmd = f"ffmpeg -v quiet -stats -y -r 30 -f image2 -pattern_type glob -i '{output_path}/snapshots/*.png' -c:v h264 -pix_fmt yuv420p -strict -2 {output_path}/snapshot_movie.mp4"
    os.system(bashcmd)
