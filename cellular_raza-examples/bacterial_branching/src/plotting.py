import pyron
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
    return [Path(p) for p in sorted(glob(str(output_path) + "/cells/ron/*"))]

def _iteration_to_file(iteration: int, output_path: Path, cs: str = "cells") -> Path:
    return output_path / "{}/ron/{:020}".format(cs, iteration)

def get_all_iterations(output_path: Path = get_last_output_path()) -> list[int]:
    iterations_files = _get_all_iteration_files(output_path)
    return [int(os.path.basename(it)) for it in iterations_files]

def load_cells_at_iteration(
        iteration: int,
        output_path: Path,
    ):
    iteration_file = _iteration_to_file(iteration, output_path, "cells")
    data = []
    for file in glob(str(iteration_file) + "/*"):
        di = pyron.load(file)["data"]
        data.extend([b["element"][0] for b in di])
    df = pd.json_normalize(data)
    for key in [
        "cell.mechanics.pos",
        "cell.mechanics.vel",
        "cell.reactions.intracellular",
        "cell.reactions.turnover_rate",
        "cell.reactions.production_term",
        "cell.reactions.secretion_rate",
        "cell.reactions.uptake_rate",
    ]:
        df[key] = df[key].apply(lambda x: np.array(x, dtype=float))
    return df

def load_subdomains_at_iteration(
        iteration: int,
        output_path: Path = get_last_output_path()
    ) -> pd.DataFrame:
    iteration_file = _iteration_to_file(iteration, output_path, "subdomains")
    data = []
    for file in glob(str(iteration_file) + "/*"):
        di = pyron.load(file)["element"]
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
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim([domain_min[0], domain_max[0]])
    ax.set_ylim([domain_min[1], domain_max[1]])

    # Plot background
    for n_sub, dfsi in dfs.iterrows():
        smin = dfsi["reactions_min"]
        smax = dfsi["reactions_max"]
        values = dfsi["extracellular.data"].reshape(dfsi["extracellular.dim"])[:,:,0]
        ax.imshow(
            values,
            vmin=extra_bounds[0],
            vmax=extra_bounds[1],
            extent=[smin[0], smax[0], smin[1], smax[1]]
        )

    # Plot cells
    points = np.array([p for p in dfc["cell.mechanics.pos"]])
    radii = np.array([r for r in dfc["cell.interaction.length_repelling"]])
    color = (
        np.array([(r[0], 0, 0) for r in dfc["cell.reactions.intracellular"]]) - intra_bounds[0]
    ) /\
        (intra_bounds[1] - intra_bounds[0])

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
    )
    ax.add_collection(collection)

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
    iterations = get_all_iterations(output_path)
    plot_all_iterations(
        (0, 1.0),
        (0, 1.0),
        output_path, 
    )

    generate_movie(output_path)