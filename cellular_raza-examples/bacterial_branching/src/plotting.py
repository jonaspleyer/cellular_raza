import json
from pathlib import Path
from glob import glob
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import multiprocessing as mp
import itertools

def get_last_output_path(search_dir: Path | None = Path("out")) -> Path:
    """
    Parameters
    ----------
    search_dir: Path
        Directory to search for results. Defaults to "out".

    Returns
    -------
    output_path: Path
        Most recent folder in the given search_dir.
    """
    return Path(sorted(list(glob(str(search_dir) + "/*")))[-1])

def _get_all_iteration_files(output_path: Path = get_last_output_path()) -> list[Path]:
    return [Path(p) for p in sorted(glob(str(output_path) + "/cells/json/*"))]

def _iteration_to_file(iteration: int, output_path: Path, cs: str = "cells") -> Path:
    return output_path / "{}/json/{:020}".format(cs, iteration)

def get_all_iterations(output_path: Path = get_last_output_path()) -> list[int]:
    iterations_files = _get_all_iteration_files(output_path)
    return [int(os.path.basename(it)) for it in iterations_files]

def load_cells_at_iteration(
        iteration: int,
        output_path: Path,
    ):
    iteration_file = _iteration_to_file(iteration, output_path, "cells")
    data = []
    for filename in glob(str(iteration_file) + "/*"):
        file = open(filename)
        di = json.load(file)["data"]
        data.extend([b["element"][0] for b in di])
    df = pd.json_normalize(data)
    for key in [
        "cell.mechanics.pos",
        "cell.mechanics.vel",
    ]:
        df[key] = df[key].apply(lambda x: np.array(x, dtype=float))
    return df

def load_subdomains_at_iteration(
        iteration: int,
        output_path: Path = get_last_output_path()
    ) -> pd.DataFrame:
    iteration_file = _iteration_to_file(iteration, output_path, "subdomains")
    data = []
    for filename in glob(str(iteration_file) + "/*"):
        file = open(filename)
        di = json.load(file)["element"]
        data.append(di)
    df = pd.json_normalize(data)
    for key in [
        "subdomain.domain_min",
        "subdomain.domain_max",
        "subdomain.min",
        "subdomain.max",
        "subdomain.dx",
        "subdomain.voxels",
        "reactions_min",
        "reactions_max",
        "reactions_dx",
        "extracellular.data",
    ]:
        df[key] = df[key].apply(lambda x: np.array(x, dtype=float))
    return df

def plot_iteration(
        iteration: int,
        intra_bounds: tuple[float, float],
        extra_bounds: tuple[float, float],
        output_path: Path = get_last_output_path(),
        save_figure: bool = True,
    ) -> matplotlib.figure.Figure | None:
    dfc = load_cells_at_iteration(iteration, output_path)
    dfs = load_subdomains_at_iteration(iteration, output_path)

    # Set size of the image
    domain_min = dfs["subdomain.domain_min"][0]
    domain_max = dfs["subdomain.domain_max"][0]
    fig, ax = plt.subplots(figsize=(16,16))
    ax.set_xlim([domain_min[0], domain_max[0]])
    ax.set_ylim([domain_min[1], domain_max[1]])

    # Plot background
    for n_sub, dfsi in dfs.iterrows():
        smin = dfsi["reactions_min"]
        smax = dfsi["reactions_max"]
        values = dfsi["extracellular.data"].reshape(dfsi["extracellular.dim"])[:,:,0]
        ax.imshow(
            values.T,
            vmin=extra_bounds[0],
            vmax=extra_bounds[1],
            extent=[smin[0], smax[0], smin[1], smax[1]],
            origin='lower',
        )

    # Plot cells
    points = np.array([p for p in dfc["cell.mechanics.pos"]])
    radii = np.array([r for r in dfc["cell.interaction.cell_radius"]])
    s = np.clip((
        np.array([r for r in dfc["cell.interaction.cell_radius"]]) / np.array([r for r in dfc["cell.division_radius"]]) - intra_bounds[0]
    ) /\
        (intra_bounds[1] - intra_bounds[0]), 0, 1)

    color_high = np.array([233, 170, 242]) / 255
    color_low = np.array([129, 12, 145]) / 255
    color = np.tensordot((1-s), color_low, 0) + np.tensordot(s, color_high, 0)

    # Plot cells as circles
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    collection = PatchCollection(
        [Circle(
            points[i,:],
            radius=radii[i],
        )
        for i in range(points.shape[0])],
        facecolors=color,
        edgecolors="black",
    )
    ax.add_collection(collection)
    ax.text(
        0.05,
        0.05,
        "Agents: {:9}".format(points.shape[0]),
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='center',
        bbox=dict(boxstyle='square', facecolor='#FFFFFF'),
    )

    ax.set_axis_off()
    fig.tight_layout()
    if save_figure:
        os.makedirs(output_path / "images", exist_ok=True)
        fig.savefig(output_path / "images/cells_at_iter_{:010}".format(iteration))
        plt.close(fig)
        return None
    else:
        return fig

def __plot_all_iterations_helper(args_kwargs):
    iteration, kwargs = args_kwargs
    plot_iteration(iteration, **kwargs)

def plot_all_iterations(
        intra_bounds: tuple[float, float],
        extra_bounds: tuple[float, float],
        output_path: Path = get_last_output_path(),
        n_threads: int | None = None,
        **kwargs,
    ):
    pool = mp.Pool(n_threads)
    kwargs["intra_bounds"] = intra_bounds
    kwargs["extra_bounds"] = extra_bounds
    kwargs["output_path"] = output_path
    iterations = get_all_iterations(output_path)
    args = zip(
        iterations,
        itertools.repeat(kwargs),
    )
    _ = list(tqdm.tqdm(pool.imap(__plot_all_iterations_helper, args), total=len(iterations)))

def generate_movie(opath: Path | None = None, play_movie: bool = True):
    if opath is None:
        opath = get_last_output_path(opath)
    bashcmd = f"ffmpeg\
        -v quiet\
        -stats\
        -y\
        -r 30\
        -f image2\
        -pattern_type glob\
        -i '{opath}/images/*.png'\
        -c:v h264\
        -pix_fmt yuv420p\
        -strict -2 {opath}/movie.mp4"
    os.system(bashcmd)

    if play_movie:
        print("Playing Movie")
        bashcmd2 = f"firefox ./{opath}/movie.mp4"
        os.system(bashcmd2)

if __name__ == "__main__":
    output_path = get_last_output_path()
    plot_all_iterations(
        (0, 1),
        (0, 10.0),
        output_path,
    )

    generate_movie(output_path)
