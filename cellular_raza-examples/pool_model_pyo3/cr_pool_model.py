import os
import json
import pandas as pd
from pathlib import Path
from cr_pool_model_pyo3 import *
import multiprocessing as mp
import numpy as np
from types import SimpleNamespace
import pyvista as pv
import matplotlib.pyplot as plt
import tqdm
import copy


def get_last_output_path(name = "pool_model"):
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


def _convert_entries(df, element_path):
    if element_path == "cell_storage":
        df["identifier"] = df["identifier"].apply(lambda x: tuple(x))
        df["element.id"] = df["element.id"].apply(lambda x: np.array(x))
        df["element.cell.mechanics.pos"] = df["element.cell.mechanics.pos"].apply(lambda x: np.array(x))
        df["element.cell.mechanics.vel"] = df["element.cell.mechanics.vel"].apply(lambda x: np.array(x))
        df["element.cell.mechanics.random_vector"] = df["element.cell.mechanics.random_vector"].apply(lambda x: np.array(x))
        df["element.cell.cellular_reactions.intracellular_concentrations"] = df["element.cell.cellular_reactions.intracellular_concentrations"].apply(lambda x: np.array(x))
        df["element.cell.cellular_reactions.turnover_rate"] = df["element.cell.cellular_reactions.turnover_rate"].apply(lambda x: np.array(x))
        df["element.cell.cellular_reactions.production_term"] = df["element.cell.cellular_reactions.production_term"].apply(lambda x: np.array(x))
        df["element.cell.cellular_reactions.degradation_rate"] = df["element.cell.cellular_reactions.degradation_rate"].apply(lambda x: np.array(x))
        df["element.cell.cellular_reactions.secretion_rate"] = df["element.cell.cellular_reactions.secretion_rate"].apply(lambda x: np.array(x))
        df["element.cell.cellular_reactions.uptake_rate"] = df["element.cell.cellular_reactions.uptake_rate"].apply(lambda x: np.array(x))
        df["element.cell.interactionextracellulargradient"] = df["element.cell.interactionextracellulargradient"].apply(lambda x: np.array(x))

    if element_path == "voxel_storage":
        df["element.index"] = df["element.index"].apply(lambda x: np.array(x))
        df["element.voxel.min"] = df["element.voxel.min"].apply(lambda x: np.array(x))
        df["element.voxel.max"] = df["element.voxel.max"].apply(lambda x: np.array(x))
        df["element.voxel.middle"] = df["element.voxel.middle"].apply(lambda x: np.array(x))
        df["element.voxel.dx"] = df["element.voxel.dx"].apply(lambda x: np.array(x))
        df["element.voxel.index"] = df["element.voxel.index"].apply(lambda x: np.array(x))
        df["element.voxel.extracellular_concentrations"] = df["element.voxel.extracellular_concentrations"].apply(lambda x: np.array(x))
        df["element.voxel.extracellular_gradient"] = df["element.voxel.extracellular_gradient"].apply(lambda x: np.array(x))
        df["element.voxel.diffusion_constant"] = df["element.voxel.diffusion_constant"].apply(lambda x: np.array(x))
        df["element.voxel.production_rate"] = df["element.voxel.production_rate"].apply(lambda x: np.array(x))
        df["element.voxel.degradation_rate"] = df["element.voxel.degradation_rate"].apply(lambda x: np.array(x))
        df["element.neighbors"] = df["element.neighbors"].apply(lambda x: np.array(x))
        df["element.cells"] = df["element.cells"].apply(lambda x: np.array(x))
        df["element.new_cells"] = df["element.new_cells"].apply(lambda x: np.array(x))
        df["element.rng.seed"] = df["element.rng.seed"].apply(lambda x: np.array(x))
        df["element.extracellular_concentration_increments"] = df["element.extracellular_concentration_increments"].apply(lambda x: np.array(x))
        df["element.concentration_boundaries"] = df["element.concentration_boundaries"].apply(lambda x: np.array(x))

    return df

def get_elements_at_iter(output_path: Path, iteration, element_path="cell_storage"):
    dir = Path(output_path) / element_path / "json"
    run_directory = None
    for x in os.listdir(dir):
        if int(x) == iteration:
            run_directory = dir / x
            break
    if run_directory != None:
        df = pd.json_normalize(_combine_batches(run_directory))
        df = _convert_entries(df, element_path)
        return pd.DataFrame(df)
    else:
        raise ValueError(f"Could not find iteration {iteration} in saved results")


def get_all_iterations(output_path, element_path="cell_storage"):
    return sorted([int(x) for x in os.listdir(Path(output_path) / element_path / "json")])


def __iter_to_elements(args):
    df = get_elements_at_iter(*args)
    df.insert(loc=0, column="iteration", value=args[1])
    return df


def get_elements_at_all_iterations(output_path: Path, element_path="cell_storage", threads=1):
    dir = Path(output_path) / element_path / "json"
    # runs = [(x, dir) for x in os.listdir(dir)]
    pool = mp.Pool(threads)
    result = list(pool.map(__iter_to_elements, map(lambda iteration: (output_path, iteration, element_path), get_all_iterations(output_path, element_path))))
    return pd.concat(result)


def save_snapshot(output_path: Path, iteration):
    pass


def __save_snapshot_helper(args):
    return save_snapshot(*args)


def save_all_snapshots(output_path: Path, threads=1, show_bar=True):
    if threads<=0:
        threads = os.cpu_count()
    output_iterations = [(output_path, iteration) for iteration in get_all_iterations(output_path)]
    if show_bar:
        list(tqdm.tqdm(mp.Pool(threads).imap(__save_snapshot_helper, output_iterations), total=len(output_iterations)))
    else:
        mp.Pool(threads).imap(__save_snapshot_helper, output_iterations)
