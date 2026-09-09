import pyvista as pv
import glob
import json
from pathlib import Path
import numpy as np
import multiprocessing as mp
import itertools
import tqdm
import matplotlib.pyplot as plt
from itertools import repeat
import argparse
import scipy as sp
import string

COLOR_HEALTHY = "#8ea604"
COLOR_INFECTED1 = "#d76a03"
COLOR_INFECTED2 = "#bf3100"


def set_mpl_rc_params():
    plt.rcParams.update(
        {
            "font.family": "Courier New",  # monospace font
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "figure.titlesize": 20,
        }
    )


def configure_ax(ax, minor=True):
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25)
    ax.minorticks_on()
    if minor:
        ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
    else:
        ax.grid(False, which="minor")
    ax.set_axisbelow(True)


def get_all_iterations(path: Path, cells_subdomains: str = "cells"):
    return [
        (int(iter_dir.split("/")[-1]), iter_dir)
        for iter_dir in sorted(
            glob.glob(str(path / "{}/json/*".format(cells_subdomains)))
        )
    ]


def __get_iteration_dir(iteration: int, path: Path, cells_subdomains: str = "cells"):
    iterations = get_all_iterations(path, cells_subdomains)
    for iteration_int, iteration_dir in iterations:
        if iteration_int == iteration:
            return Path(iteration_dir)
    return None


def get_last_save_path(path: Path):
    paths = sorted([p for p in glob.glob(str(path) + "/*")])
    return Path(paths[-1])


def get_cells_at_iteration(iteration: int, path: Path):
    iter_dir = __get_iteration_dir(iteration, path)
    cells = []
    for batch in glob.glob(str(iter_dir) + "/*"):
        with open(batch) as f:
            d = json.load(f)
            cells_new = d["data"]
            cells.extend(cells_new)
    return cells


def get_subdomains_at_iteration(iteration: int, path: Path) -> list:
    iter_dir = __get_iteration_dir(iteration, path, "subdomains")
    subdomains = []
    for single in glob.glob(str(iter_dir) + "/*"):
        with open(single) as f:
            d = json.load(f)
            subdomains.append(d["element"])
    return subdomains


def map_health_state(c) -> int:
    st = c["element"][0]["cell"]["health_state"]
    if st == "Healthy":
        return 0
    elif st == "Selfinfected":
        return 1
    elif st == "Infected":
        return 2
    else:
        return -1


def get_min_max_from_subdomains(
    iteration: int, path: Path
) -> tuple[np.ndarray, np.ndarray]:
    subdomains = get_subdomains_at_iteration(iteration, path)
    min = np.array(subdomains[0]["domain_min"])
    max = np.array(subdomains[0]["domain_max"])
    return (min, max)


def get_spheres(iteration: int, path: Path):
    cells = get_cells_at_iteration(iteration, path)

    def get_info(cell):
        return (
            cell["element"][0]["cell"]["mechanics"]["pos"],
            cell["element"][0]["cell"]["interaction"]["radius"],
            map_health_state(cell),
        )

    position_radius_health = [get_info(ci) for ci in cells]
    pset = pv.PolyData([np.array(x[0]) for x in position_radius_health])
    pset.point_data["diameter"] = 2.0 * np.array([x[1] for x in position_radius_health])
    pset.point_data["infected"] = np.array([x[2] for x in position_radius_health])

    sphere = pv.Sphere()
    spheres = pset.glyph(geom=sphere, scale="diameter", orient=False)
    return spheres


def plot_spheres(
    iteration: int,
    path: Path,
    opath=None,
    transparent_background: bool = False,
    ret=False,
):
    spheres = get_spheres(iteration, path)
    domain_min, domain_max = get_min_max_from_subdomains(iteration, path)
    dx = np.max(domain_max[:2] - domain_min[:2])
    middle = 0.5 * (domain_min + domain_max)

    plotter = pv.Plotter(off_screen=True, window_size=[1024, 1024])
    plotter.set_background([100, 100, 100])
    plotter.add_mesh(
        spheres,
        scalars="infected",
        cmap=[COLOR_HEALTHY, COLOR_INFECTED1, COLOR_INFECTED2],
        show_edges=False,
        show_scalar_bar=False,
    )
    plotter.enable_ssao(radius=12)
    plotter.enable_anti_aliasing()
    plotter.camera_position = "xy"
    plotter.camera.position = (0.5 * dx, 0.5 * dx, -2 * dx)
    # plotter.camera.up = (0, 1, 0)
    plotter.camera.focal_point = middle
    if opath is None:
        opath = path / "images/{:010}.png".format(iteration)
        opath.parent.mkdir(parents=True, exist_ok=True)
    img = plotter.screenshot(opath, transparent_background=transparent_background)
    plotter.close()
    del plotter
    del spheres
    if ret:
        return img


def __plot_spheres_helper(args_kwargs):
    args, kwargs = args_kwargs
    plot_spheres(*args, **kwargs, transparent_background=True)


def plot_all_spheres(
    path: Path,
    n_threads: int | None = None,
    **kwargs: dict,
):
    iterations = [it[0] for it in get_all_iterations(path)]

    if n_threads is None:
        n_threads = mp.cpu_count()

    pool = mp.Pool(n_threads)
    print("Generating Images")
    _ = list(
        tqdm.tqdm(
            pool.imap(
                __plot_spheres_helper,
                zip(zip(iterations, itertools.repeat(path)), itertools.repeat(kwargs)),
                chunksize=1,
            ),
            total=len(iterations),
        ),
    )
    pool.close()
    pool.join()


def __get_cell_numbers(iteration_path):
    iteration, path = iteration_path
    cells = get_cells_at_iteration(iteration, path)
    markers = np.array([map_health_state(c) for c in cells])
    return np.sum(markers == 0), np.sum(markers == 1), np.sum(markers == 2)


def run_plot_all_spheres(path, pyargs):
    plot_all_spheres(path, pyargs.n_threads)


def plot_combined(path, pyargs):
    iterations = get_all_iterations(path)
    iterations = np.array([i[0] for i in iterations])

    try:
        n_cells = np.load(path / "n_cells.npy")
    except:
        n_cells = []
        with mp.Pool(pyargs.n_threads) as pool:
            n_cells = list(
                tqdm.tqdm(
                    pool.imap(__get_cell_numbers, zip(iterations, repeat(path))),
                    total=len(iterations),
                )
            )
        n_cells = np.array(n_cells)
        np.save(path / "n_cells", n_cells)

    mp.set_start_method("spawn")
    set_mpl_rc_params()
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    for label, ax in zip(string.ascii_uppercase, axs):
        ax.text(
            0.03,
            0.97,
            label,
            fontsize=40,
            fontweight="semibold",
            fontfamily="serif",
            va="top",
            horizontalalignment="left",
            transform=ax.transAxes,
            color="k" if label == "A" else "gray",
        )

    # Plot number of cells for each iteration
    configure_ax(axs[0])
    axs[0].stackplot(
        iterations,
        n_cells[:, 0],
        n_cells[:, 1] + n_cells[:, 2],
        colors=[COLOR_HEALTHY, COLOR_INFECTED2],
        labels=["Healthy", "Infected"],
    )
    axs[0].legend(frameon=False)
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Number of Cells")

    configure_ax(axs[1])

    axs[1].plot(
        n_cells[:, 0],
        n_cells[:, 1] + n_cells[:, 2],
        color="#A0A0A0",
        linestyle=":",  # (5, (1, 3)),
        alpha=0.75,
    )

    axs[1].set_xlabel("Healthy Cells")
    axs[1].set_ylabel("Infected Cells")
    axs[1].set_title("Phase Space")

    # n_low = int(len(iterations) / 10)
    xmin = np.min(n_cells[:, 0])
    xmax = np.max(n_cells[:, 0])
    dx = xmax - xmin
    ymin = np.min(n_cells[:, 1] + n_cells[:, 2])
    ymax = np.max(n_cells[:, 1] + n_cells[:, 2])
    dy = ymax - ymin
    s = 0.1

    # Calculate KDE
    x = n_cells[:, 0]
    y = n_cells[:, 1] + n_cells[:, 2]
    kernel = sp.stats.gaussian_kde([x, y])
    X, Y = np.mgrid[
        xmin - s * dx : xmax + s * dx : 100, ymin - s * dy : ymax + s * dy : 100
    ]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)
    axs[1].imshow(
        np.rot90(Z),
        cmap="viridis",
        extent=[xmin - s * dx, xmax + s * dx, ymin - s * dy, ymax + s * dy],
        aspect="auto",
    )

    axs[1].set_xlim(xmin - s * dx, xmax + s * dx)
    axs[1].set_ylim(ymin - s * dy, ymax + s * dy)

    img = plot_spheres(
        iterations[-1],
        path,
        opath=None,
        transparent_background=True,
        ret=True,
    )
    axs[2].imshow(img, aspect="auto")
    # axs[2].set_axis_off()

    # fig.tight_layout()
    fig.savefig(path / "phase-space.png")
    fig.savefig(path / "phase-space.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="Input Directory", default=None)
    parser.add_argument("--n-threads", help="Number of threads", default=None)
    parser.add_argument("--plot-all-spheres", action="store_true", default=False)
    parser.add_argument("-p", "--path", default=None)
    pyargs = parser.parse_args()

    path = Path(pyargs.input_dir or get_last_save_path(Path("out/cell_sorting")))
    pyargs.n_threads = int(pyargs.n_threads or mp.cpu_count())

    if pyargs.plot_all_spheres:
        run_plot_all_spheres(path, pyargs)

    plot_combined(path, pyargs)
