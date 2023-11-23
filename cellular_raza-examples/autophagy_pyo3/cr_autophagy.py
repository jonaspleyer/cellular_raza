import os
import json
import pandas as pd
from pathlib import Path
from cr_autophagy_pyo3 import *
import multiprocessing as mp
import numpy as np
from types import SimpleNamespace
import pyvista as pv
import matplotlib.pyplot as plt
import tqdm


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


def get_particles_at_iter(output_path: Path, iteration):
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


def get_particles_at_all_iterations(output_path: Path, threads=1):
    dir = Path(output_path) / "cell_storage/json/"
    runs = [(x, dir) for x in os.listdir(dir)]
    pool = mp.Pool(threads)
    result = list(pool.map(__iter_to_cells, runs[:10]))
    return result


def generate_spheres(output_path: Path, iteration):
    # Filter for only particles at the specified iteration
    df = get_particles_at_iter(output_path, iteration)
    # df = df[df["iteration"]==iteration]

    # Create a dataset for pyvista for plotting
    pset = pv.PolyData(np.array([np.array(x) for x in df["element.cell.mechanics.mechanics.pos"]]))

    # Extend dataset by species and diameter
    pset.point_data["diameter"] = 2.0*df["element.cell.interaction.cell_radius"]
    pset.point_data["species"] = df["element.cell.interaction.species"]

    # Create spheres glyphs from dataset
    sphere = pv.Sphere()
    spheres = pset.glyph(geom=sphere, scale="diameter", orient=False)

    return spheres


def save_snapshot(output_path: Path, iteration):
    simulation_settings = get_simulation_settings(output_path)
    ofolder = Path(output_path) / "snapshots"
    ofolder.mkdir(parents=True, exist_ok=True)
    opath = ofolder / "snapshot_{:08}.png".format(iteration)
    if os.path.isfile(opath):
        return
    spheres = generate_spheres(output_path, iteration)

    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            jupyter_backend = 'none'
    except:
        jupyter_backend = None
    spheres.plot(
        off_screen=True,
        screenshot=opath,
        scalars="species",
        scalar_bar_args={
            "title":"Species",
        },
        cpos=[
            (
                -1.5*simulation_settings.domain_size,
                -1.5*simulation_settings.domain_size,
                -1.5*simulation_settings.domain_size
            ),(
                25,
                25,
                25
            ),(
                0.0,
                0.0,
                0.0
            )
        ],
        # TODO possibly enable this if needed again
        jupyter_backend=jupyter_backend,
    )


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


def save_scatter_snapshot(output_path: Path, iteration):
    df = get_particles_at_iter(output_path, iteration)

    cargo_at_end = df[df["element.cell.interaction.species"]=="Cargo"]["element.cell.mechanics.mechanics.pos"]
    cargo_at_end = np.array([np.array(elem) for elem in cargo_at_end])
    non_cargo_at_end = df[df["element.cell.interaction.species"]!="Cargo"]["element.cell.mechanics.mechanics.pos"]
    non_cargo_at_end = np.array([np.array(elem) for elem in non_cargo_at_end])
    cargo_middle = np.average(non_cargo_at_end, axis=0)

    def appendSpherical_np(xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
        ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
        return ptsnew

    non_cargo_at_end_spherical = appendSpherical_np(non_cargo_at_end - cargo_middle)
    r = non_cargo_at_end_spherical[:,3]
    r_inv = np.max(r) - r
    phi = non_cargo_at_end_spherical[:,4]
    theta = non_cargo_at_end_spherical[:,5]

    fig, ax = plt.subplots()
    ax.set_title("Radial distribution of particles around cargo center")
    ax.scatter(phi, theta, s=r_inv, alpha=0.5)

    ax.set_xlabel("$\\varphi$ [rad]")
    ax.set_ylabel("$\\theta$ [rad]")
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_xticklabels(["$0$", "$\\frac{\\pi}{4}$", "$\\frac{\\pi}{2}$", "$\\frac{3\\pi}{4}$", "$\\pi$"])
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels(["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])

    ax.set_xlim([-np.pi/12, np.pi*(1+1/12)])
    ax.set_ylim([-np.pi*(1+1/6), np.pi*(1+1/6)])

    ofolder = output_path / "scatterplots"
    ofolder.mkdir(parents=True, exist_ok=True)
    fig.savefig(ofolder / f"snapshot_{iteration:08}_scatter.png")
    plt.close(fig)


def __save_scatter_snapshot_helper(args):
    return save_scatter_snapshot(*args)


def save_all_scatter_snapshots(output_path: Path, threads=1, show_bar=True):
    if threads<=0:
        threads = os.cpu_count()
    output_iterations = [(output_path, iteration) for iteration in get_all_iterations(output_path)]
    if show_bar:
        list(tqdm.tqdm(mp.Pool(threads).imap(__save_scatter_snapshot_helper, output_iterations), total=len(output_iterations)))
    else:
        mp.Pool(threads).map(__save_scatter_snapshot_helper, output_iterations)
