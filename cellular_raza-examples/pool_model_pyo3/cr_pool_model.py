import os
import json
import pandas as pd
from pathlib import Path
from cr_pool_model_pyo3 import *
import multiprocessing as mp
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
import tqdm
import matplotlib
from matplotlib import pyplot as plt


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
        df["element.cell.cellular_reactions.production_rates"] = df["element.cell.cellular_reactions.production_rates"].apply(lambda x: np.array(x))
        df["element.cell.cellular_reactions.uptake_rates"] = df["element.cell.cellular_reactions.uptake_rates"].apply(lambda x: np.array(x))
        df["element.cell.cellular_reactions.inhibitions"] = df["element.cell.cellular_reactions.inhibitions"].apply(lambda x: np.array(x))
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
    if threads<=0:
        threads = os.cpu_count()
    dir = Path(output_path) / element_path / "json"
    # runs = [(x, dir) for x in os.listdir(dir)]
    pool = mp.Pool(threads)
    all_iterations = get_all_iterations(output_path, element_path)
    result = list(tqdm.tqdm(pool.imap(__iter_to_elements, map(lambda iteration: (output_path, iteration, element_path), all_iterations)), total=len(all_iterations)))
    return pd.concat(result)


def save_snapshot(output_path, iteration, overwrite=False):
    save_path = Path(output_path) / "snapshot_{:08}.png".format(iteration)
    if overwrite==False and os.path.isfile(save_path):
        return None

    # Get simulation settings and particles at the specified iteration
    simulation_settings = get_simulation_settings(output_path)
    df_cells = get_elements_at_iter(output_path, iteration, element_path="cell_storage")
    df_voxels = get_elements_at_iter(output_path, iteration, element_path="voxel_storage")

    # Get positions as large numpy array
    positions = np.array([np.array(x) for x in df_cells["element.cell.mechanics.pos"]])
    s = np.array([x for x in df_cells["element.cell.interaction.cell_radius"]])
    c = np.array([x for x in df_cells["element.cell.cellular_reactions.intracellular_concentrations"]])[:,1]
    norm = matplotlib.colors.Normalize(
        vmin=0,
        vmax=c.max(),
        clip=True,
    )
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.summer)
    c = mapper.to_rgba(c.max() - c)

    # Define limits for domain from simulation settings
    xlims = np.array([0.0, simulation_settings.domain.size])
    ylims = np.array([0.0, simulation_settings.domain.size])

    figsize_x = 16
    figsize_y = (ylims[1]-ylims[0])/(xlims[1]-xlims[0])*figsize_x

    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Plot rectangles for background
    # Create color mapper for background
    nutrients_min = 0.0
    norm2 = matplotlib.colors.Normalize(
        vmin=nutrients_min,
        vmax=simulation_settings.domain.initial_concentrations[0],
        clip=True
    )
    mapper2 = matplotlib.cm.ScalarMappable(norm=norm2, cmap=matplotlib.cm.cividis)

    def plot_rectangle(entry):
        x_min = entry["element.voxel.min"]
        x_max = entry["element.voxel.max"]
        conc = entry["element.voxel.extracellular_concentrations"][0]

        xy = [x_min[0], x_min[1]]
        dx = x_max[0] - x_min[0]
        dy = x_max[1] - x_min[1]
        color = mapper2.to_rgba(conc) if not np.isnan(conc) else "red"
        rectangle = matplotlib.patches.Rectangle(xy, width=dx, height=dy, color=color)
        ax.add_patch(rectangle)

    df_voxels.apply(plot_rectangle, axis=1)

    # Plot circles for bacteria
    for pos, si, ci in zip(positions, s, c):
        if si!=None:
            circle = plt.Circle(pos, radius=si, facecolor=ci, edgecolor='k')
            ax.add_patch(circle)
        else:
            print("Warning: Skip drawing bacteria with None radius!")

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    fig.tight_layout()

    fig.savefig(save_path)
    plt.close(fig)
    del df_cells
    del df_voxels
    del positions


def __save_snapshot_helper(all_args):
    return save_snapshot(*all_args[0], **all_args[1])


def save_all_snapshots(output_path: Path, threads=1, show_bar=True, **kwargs):
    if threads<=0:
        threads = os.cpu_count()
    all_args = [((output_path, iteration), kwargs) for iteration in get_all_iterations(output_path)]
    chunksize = max(int(len(all_args)/threads), 5)
    if show_bar:
        _ = list(tqdm.tqdm(mp.Pool(threads).imap(__save_snapshot_helper, all_args, chunksize=chunksize), total=len(all_args)))
    else:
        mp.Pool(threads).imap(__save_snapshot_helper, all_args, chunksize=chunksize)
