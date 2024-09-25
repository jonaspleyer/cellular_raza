import pyvista as pv
import glob
import json
from pathlib import Path
import numpy as np
import multiprocessing as mp
import itertools
import tqdm

def get_all_iterations(path: Path, cells_subdomains: str = "cells"):
    return [
        (int(iter_dir.split("/")[-1]), iter_dir)
        for iter_dir in sorted(glob.glob(str(path / "{}/json/*".format(cells_subdomains))))
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

def get_min_max_from_subdomains(iteration: int, path: Path) -> tuple[np.ndarray, np.ndarray]:
    subdomains = get_subdomains_at_iteration(iteration, path)
    min = np.array(subdomains[0]["domain_min"])
    max = np.array(subdomains[0]["domain_max"])
    return (min, max)

def get_spheres(iteration: int, path: Path):
    cells = get_cells_at_iteration(iteration, path)
    get_info = lambda cell: (
        cell["element"][0]["cell"]["mechanics"]["pos"],
        cell["element"][0]["cell"]["interaction"]["cell_radius"],
        cell["element"][0]["cell"]["interaction"]["species"]
    )
    position_radius_species = [get_info(ci) for ci in cells]
    pset = pv.PolyData([np.array(x[0]) for x in position_radius_species])
    pset.point_data["diameter"] = 2.0 * np.array([x[1] for x in position_radius_species])
    pset.point_data["species"] = np.array([x[2] for x in position_radius_species])

    sphere = pv.Sphere()
    spheres = pset.glyph(geom=sphere, scale="diameter", orient=False)
    return spheres

def plot_spheres(iteration: int, path: Path, opath = None, transparent_background: bool = False):
    spheres = get_spheres(iteration, path)
    domain_min, domain_max = get_min_max_from_subdomains(iteration, path)
    dx = np.max(domain_max - domain_min)
    middle = 0.5 * (domain_min + domain_max)

    plotter = pv.Plotter(off_screen=True, window_size=[1024, 1024])
    plotter.set_background([100, 100, 100])
    plotter.add_mesh(
        spheres,
        scalars="species",
        show_edges=False,
        cmap=["blue", "red"],
        show_scalar_bar=False
    )
    plotter.enable_ssao(radius=12)
    plotter.enable_anti_aliasing()
    plotter.camera.position = (- 2*dx, - 2*dx, +2*dx)
    plotter.camera.focal_point = middle
    if opath is None:
        opath = path / "images/{:010}.png".format(iteration)
        opath.parent.mkdir(parents=True, exist_ok=True)
    img = plotter.screenshot(opath, transparent_background=transparent_background)
    plotter.close()
    return img

def __plot_spheres_helper(args_kwargs):
    args, kwargs = args_kwargs
    plot_spheres(*args, **kwargs)

def plot_all_spheres(path: Path, **kwargs: dict):
    iterations = [it[0] for it in get_all_iterations(path)]
    pool = mp.Pool()
    print("Generating Images")
    list(
        tqdm.tqdm(
            pool.imap_unordered(
                __plot_spheres_helper,
                zip(zip(iterations, itertools.repeat(path)), itertools.repeat(kwargs))
            ),
            total=len(iterations)
    ))

if __name__ == "__main__":
    last_save_path = get_last_save_path(Path("out/cell_sorting"))

    plot_all_spheres(last_save_path, transparent_background=True)
