import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
from glob import glob
from pathlib import Path
import os
import multiprocessing as mp
import itertools

def get_last_save_path(opath: Path | None = None) -> Path | None:
    if opath is None:
        opath = Path("out/puzzles")
    results = sorted(glob(str(opath) + "/*"))
    if len(results) == 0:
        return None
    else:
        return Path(results[-1])

def get_iterations(opath: Path | None = None) -> list[int]:
    opath = get_last_save_path(opath)
    return [int(Path(x).name) for x in glob(str(opath) + "/cells/json/*")]

def load_cells(iteration: int, opath: Path | None = None) -> list[dict] | None:
    if opath is None:
        spath = get_last_save_path(opath)
    else:
        spath = opath
    batches = glob(str(spath) + "/cells/json/{:020}/*.json".format(iteration))
    if len(batches) == 0:
        return None
    else:
        cells = []
        for b in batches:
            f = open(b)
            data = json.load(f)["data"]
            for d in data:
                cells.append(d["element"][0]["cell"])
        return cells

def extract_vertices(cells: list[dict] | None) -> list[np.ndarray] | None:
    if cells is None:
        return None
    vertices = []
    for cell in cells:
        vertices.append(np.array(cell["mechanics"]["puzzle"]["vertices"]))
    return vertices

def plot_vertices(vertices: list[np.ndarray] | None) -> matplotlib.figure.Figure:
    if vertices is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 8))
    # TODO magic numbers
    ax.set_xlim([0, 40])
    ax.set_ylim([0, 40])
    for cell_vertices in vertices:
        polygon = matplotlib.patches.Polygon(
            cell_vertices,
            closed=True,
            fill=True,
            facecolor="green",
            edgecolor="black",
        )
        ax.add_patch(polygon)
    return fig

def plot_cells_at_iteration(iteration: int, opath: Path | None = None) -> matplotlib.figure.Figure:
    cells = load_cells(iteration, opath)
    vertices = extract_vertices(cells)
    return plot_vertices(vertices)

def plot_all_iterations(opath: Path | None = None) -> list[matplotlib.figure.Figure]:
    iterations = get_iterations(opath)
    return [plot_cells_at_iteration(it, opath) for it in iterations]

def _concurrent_single_iter_plotter(args_kwargs):
    it, spath = args_kwargs
    fig = plot_cells_at_iteration(it, spath)
    figpath = spath / "snapshots/snapshot_{:020}.png".format(it)
    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath)
    plt.close(fig)

def save_plot_all_iterations(opath: Path | None = None) -> int | None:
    spath = get_last_save_path(opath)
    if spath is None:
        return None
    iterations = get_iterations(opath)

    print("Plotting Results")
    import tqdm
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(
            _concurrent_single_iter_plotter,
            zip(iterations, itertools.repeat(spath))
        ), total=len(iterations)))
        return len(results)

def generate_movie(opath: Path | None = None, play_movie: bool = True):
    if opath is None:
        opath = get_last_save_path(opath)
    bashcmd = f"ffmpeg\
        -v quiet\
        -stats\
        -y\
        -r 30\
        -f image2\
        -pattern_type glob\
        -i '{opath}/snapshots/*.png'\
        -c:v h264\
        -pix_fmt yuv420p\
        -strict -2 {opath}/movie.mp4"
    os.system(bashcmd)

    if play_movie:
        print("Playing Movie")
        bashcmd2 = f"firefox ./{opath}/movie.mp4"
        os.system(bashcmd2)


if __name__ == "__main__":
    save_plot_all_iterations()
    generate_movie()
