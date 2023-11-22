import os
import json
import pandas as pd
from pathlib import Path
from cr_autophagy_pyo3 import *
import multiprocessing as mp
import numpy as np
from types import SimpleNamespace

def get_last_output_path(name = "autophagy"):
    return Path("out") / name / sorted(os.listdir(Path("out") / name))[-1]

def get_simulation_settings(output_path):
    f = open(output_path / "simulation_settings.json")
    return json.load(f, object_hook=lambda d: SimpleNamespace(**d))

def _combine_batches(run_directory):
    # Opens all batches in a given directory and stores
    # them in one unified big list
    combined_batch = []
    for batch_file in os.listdir(run_directory):
        f = open(run_directory / batch_file)
        b = json.load(f)["data"]
        combined_batch.extend(b)
    return combined_batch

def get_particles_at_iter(output_path, iteration):
    dir = Path(output_path) / "cell_storage/json"
    run_directory = None
    for x in os.listdir(dir):
        if int(x) == iteration:
            run_directory = dir / x
            break
    if run_directory != None:
        df = pd.json_normalize(_combine_batches(run_directory))
        df["identifier"] = df["identifier"].apply(lambda x: tuple(x))
        df["element.cell.mechanics.mechanics.pos"] = df["element.cell.mechanics.mechanics.pos"].apply(lambda x: np.array(x, dtype=float))
        df["element.cell.mechanics.mechanics.vel"] = df["element.cell.mechanics.mechanics.vel"].apply(lambda x: np.array(x, dtype=float))
        df["element.cell.mechanics.mechanics.random_vector"] = df["element.cell.mechanics.mechanics.random_vector"].apply(lambda x: np.array(x))
        return df
    else:
        raise ValueError(f"Could not find iteration {iteration} in saved results")

def get_all_iterations(output_path):
    return sorted([int(x) for x in os.listdir(Path(output_path) / "cell_storage/json")])

def __iter_to_cells(iteration_dir):
    iteration, dir = iteration_dir
    return (int(iteration), _combine_batches(dir / iteration))

def get_particles_at_all_iterations(output_path, threads=1):
    dir = Path(output_path) / "cell_storage/json/"
    runs = [(x, dir) for x in os.listdir(dir)]
    pool = mp.Pool(threads)
    result = list(pool.map(__iter_to_cells, runs[:10]))
    return result
