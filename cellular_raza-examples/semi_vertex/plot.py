import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
import json
import pandas as pd
from pathlib import Path
from glob import glob
import tqdm
import multiprocessing as mp


def get_last_output_path() -> Path:
    return Path(sorted(list(glob("out/semi_vertex/*")))[-1])

def get_all_iterations(output_path: Path | None = None) -> np.ndarray:
    if output_path is None:
        output_path = get_last_output_path()
    folders = glob(str(output_path / "cells/json/*"))
    return np.sort(np.array([int(Path(f).name) for f in folders]))


def load_cells(iteration: int, output_path: Path | None = None) -> pd.DataFrame:
    if output_path is None:
        output_path = get_last_output_path()

    # Load all json files
    results = []
    for file in glob(str(output_path / "cells/json/{:020}/*.json".format(iteration))):
        f = open(file)
        batch = json.load(f)
        results.extend([b["element"][0] for b in batch["data"]])
    df = pd.json_normalize(results)
    df["cell.mechanics.points"] = df["cell.mechanics.points"].apply(
        lambda x: np.array(x, dtype=float).reshape((2, -1)).T
    )
    df["cell.intracellular"] = df["cell.intracellular"].apply(lambda x: np.array(x, dtype=float))
    return df


def plot_cells(ax, df_cells, intra_low, intra_high):
    viridis = mpl.colormaps['viridis'].resampled(255)
    thresh = intra_low + 0.5 * (intra_high - intra_low)
    for pos, intracellular in zip(df_cells["cell.mechanics.points"], df_cells["cell.intracellular"]):
        if intracellular[2] < thresh:
            c = viridis(0)
        else:
            c = viridis((intracellular[2] - intra_low) / (intra_high - intra_low))
        polygon = mpatches.Polygon(pos, facecolor=c, edgecolor='white')
        ax.add_patch(polygon)


def plot_cells_at_iter(
        iteration: int,
        intra_low: float,
        intra_high: float,
        output_path: Path | None = None,
        save_path: Path | None = None,
        overwrite: bool = False,
        transparent: bool = False,
    ):
    if output_path is None:
        output_path = get_last_output_path()
    if save_path is None:
        save_path = output_path / "images"
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path / "snapshot-{:020}.png".format(iteration)
    if save_path.exists() and overwrite is False:
        return save_path
    cells = load_cells(iteration, output_path)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 800)
    ax.set_axis_off()
    plot_cells(ax, cells, intra_low, intra_high)
    fig.tight_layout()
    fig.savefig(save_path, transparent=transparent)
    plt.close(fig)
    return save_path


def __plotting_helper(args_kwargs):
    args, kwargs = args_kwargs
    return plot_cells_at_iter(*args, **kwargs)


def plot_cells_at_all_iterations(
        intra_min: float,
        intra_max: float,
        output_path: Path | None = None,
        save_path: Path | None = None,
        overwrite: bool = False,
        transparent: bool = True,
    ) -> list:
    if output_path is None:
        output_path = get_last_output_path()
    iterations = get_all_iterations(output_path)
    arguments = [(
        (it, intra_min, intra_max),
        {
            "output_path": output_path,
            "save_path": save_path,
            "overwrite": overwrite,
            "transparent": transparent,
        }
    ) for it in iterations]
    pool = mp.Pool()
    return list(tqdm.tqdm(pool.imap(__plotting_helper, arguments), total=len(iterations)))

if __name__ == "__main__":
    plot_cells_at_all_iterations(0.0, 4.0, overwrite=True, transparent=True)

