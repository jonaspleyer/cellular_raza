from cr_autophagy_pyo3 import SimulationSettings, run_simulation
import cr_autophagy as cra
import numpy as np
import itertools
import multiprocessing as mp
import os
import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import json


def create_default_settings():
    simulation_settings = SimulationSettings()

    # Settings Cargo
    simulation_settings.n_cells_cargo=100
    simulation_settings.n_cells_r11=250
    simulation_settings.cell_radius_cargo = 1.5
    simulation_settings.interaction_range_cargo_cargo = 2.0

    # Settings R11
    simulation_settings.potential_strength_r11_r11 = 0.015
    simulation_settings.potential_strength_cargo_r11 = 0.005
    simulation_settings.potential_strength_cargo_r11_avidity = 1.5
    simulation_settings.kb_temperature_r11 = 0.015

    simulation_settings.interaction_range_r11_cargo = 1 * simulation_settings.cell_radius_cargo

    # Time
    simulation_settings.n_threads = 1
    simulation_settings.dt = 1.0
    simulation_settings.n_times = 30_001
    simulation_settings.save_interval = 2_000

    # Domain
    simulation_settings.domain_size = 30
    simulation_settings.domain_cargo_low = [10]*3
    simulation_settings.domain_cargo_high = [20]*3

    # Other settings
    simulation_settings.show_progressbar = False

    return simulation_settings


def run_single_simulation(i, potential_strength_r11_r11, potential_strength_cargo_r11, potential_strength_cargo_r11_avidity, kb_temperature_r11):
    simulation_settings = create_default_settings()
    simulation_settings.potential_strength_r11_r11 = potential_strength_r11_r11
    simulation_settings.potential_strength_cargo_r11 = potential_strength_cargo_r11
    simulation_settings.potential_strength_cargo_r11_avidity = potential_strength_cargo_r11_avidity
    simulation_settings.kb_temperature_r11 = kb_temperature_r11

    simulation_settings.show_progressbar = False
    simulation_settings.storage_name = f"out/autophagy/explore_parameter_space_{i:08}/"

    output_path = run_simulation(simulation_settings)
    return Path(output_path)


def combine_plots(output_path):
    number = str(output_path).split("/")[-2].split("_")[-1]
    f = open(output_path / "simulation_settings.json")
    simulation_settings = json.load(f)

    potential_strength_r11_r11 = simulation_settings["potential_strength_r11_r11"]
    potential_strength_cargo_r11 = simulation_settings["potential_strength_cargo_r11"]
    potential_strength_cargo_r11_avidity = simulation_settings["potential_strength_cargo_r11_avidity"]
    kb_temperature_r11 = simulation_settings["kb_temperature_r11"]

    if potential_strength_cargo_r11>=1.1:
        return None

    cell_text = []
    cell_text.append(["potential_strength_r11_r11", potential_strength_r11_r11])
    cell_text.append(["potential_strength_cargo_r11", potential_strength_cargo_r11])
    cell_text.append(["potential_strength_cargo_r11_avidity", potential_strength_cargo_r11_avidity])
    cell_text.append(["kb_temperature_r11", kb_temperature_r11])

    max_iter = max(cra.get_all_iterations(output_path))
    im = plt.imread(f"{output_path}/snapshots/snapshot_{max_iter:08}.png")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    ax.table(cell_text)
    ax.set_title(f"Number {number}")
    ax.imshow(im)
    fig.tight_layout()
    plt.savefig(f"param_space/combined_{number}.png")
    plt.close(fig)

def postprocessing(output_path):
    # Save scatter snapshots
    # for iteration in cra.get_all_iterations(output_path):
    #     cra.save_scatter_snapshot(output_path, iteration)

    # Also create a movie with ffmpeg
    # bashcmd = f"ffmpeg -hide_banner -loglevel panic -y -r 30 -f image2 -pattern_type glob -i '{output_path}/scatterplots/*.png' -c:v h264 -pix_fmt yuv420p -strict -2 {output_path}/scatter_movie.mp4"
    # os.system(bashcmd)

    # Save all snapshots
    # for iteration in cra.get_all_iterations(output_path):
    #     cra.save_snapshot(output_path, iteration)
    max_iter = max(cra.get_all_iterations(output_path))
    cra.save_snapshot(output_path, max_iter)

    combine_plots(output_path)
    
    # Also create a movie with ffmpeg
    # bashcmd = f"ffmpeg -hide_banner -loglevel panic -y -r 30 -f image2 -pattern_type glob -i '{output_path}/snapshots/*.png' -c:v h264 -pix_fmt yuv420p -strict -2 {output_path}/snapshot_movie.mp4"
    # os.system(bashcmd)

    return True


def run_pipeline(args):
    output_path = run_single_simulation(*args)
    return postprocessing(output_path)

def sample_parameter_space():
    potential_strength_r11_r11 = np.arange(0.001, 0.01, 0.002)#0.015
    potential_strength_cargo_r11 = np.arange(0.05, 0.1, 0.02)#0.005
    potential_strength_cargo_r11_avidity = np.arange(0.05, 0.3, 0.05)#1.5
    kb_temperature_r11 = np.arange(0.006, 0.011, 0.002)#0.015
    # kb_temperatures = np.arange(0.001, 0.002, 0.0002)
    # kb_temperatures = [0.001]
    # clustering_strengths = np.arange(0.15, 0.25, 0.01)
    # clustering_strengths = [0.15]
    # avidities = np.arange(0.0, 2.0, 0.2)

    entries = [(i, *args) for (i, args) in enumerate(itertools.product(
        potential_strength_r11_r11,
        potential_strength_cargo_r11,
        potential_strength_cargo_r11_avidity,
        kb_temperature_r11,
    ))]
    return entries


if __name__ == "__main__":
    parameter_space = sample_parameter_space()

    with mp.Pool(14) as p:
        paths = list(tqdm.tqdm(p.imap(run_pipeline, parameter_space), total=len(parameter_space)))
