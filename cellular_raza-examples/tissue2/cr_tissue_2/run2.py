import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
from tqdm import tqdm
import multiprocessing as mp
import itertools

import cr_tissue_2 as crt


def save_snapshot(iteration, domain_size, result, resolution=30):
    fig, ax = plt.subplots(figsize=(12, 12))
    for polygon, [area, target_area, perimeter, target_perimeter] in result[iteration]:
        color2 = mpl.colormaps["coolwarm"](0.5 * perimeter / target_perimeter)
        ax.add_patch(
            mpl.patches.Polygon(
                polygon.T,
                facecolor=color2,
                linestyle="-",
                edgecolor="k",
            )
        )

    dx = domain_size
    ax.set_xlim(-0.01 * dx, 1.01 * domain_size)
    ax.set_ylim(-0.01 * dx, 1.01 * domain_size)

    fig.savefig(f"out/{iteration:010}.png")
    plt.close(fig)


def __pool_save_snapshot_helper(args):
    return save_snapshot(*args)


if __name__ == "__main__":
    settings = crt.SimulationSettings()

    domain_size = 60
    domain_size_start_x = 30
    domain_size_start_y = 30
    n_agents = 10
    n_voxels = 3

    settings.dt = 0.2
    settings.t_max = 10_000.0
    settings.save_interval = 10.0
    settings.domain_size = domain_size
    settings.n_voxels = n_voxels

    radius = 5.0
    target_area = np.pi * radius**2
    target_perimeter = 2 * np.pi * radius * 1.3

    # Generate initial starting points of cells
    sampler = sp.stats.qmc.LatinHypercube(d=2, seed=10)
    domain_size = settings.domain_size
    midpoints = sampler.random(n_agents)
    dlow = [
        domain_size / 2 - domain_size_start_x / 2,
        domain_size / 2 - domain_size_start_y / 2,
    ]
    dhigh = [
        domain_size / 2 + domain_size_start_x / 2,
        domain_size / 2 + domain_size_start_y / 2,
    ]
    midpoints = sp.stats.qmc.scale(midpoints, dlow, dhigh)

    # Generate a polygon for each starting point
    n_vertices = 40
    samples = []
    for middle in midpoints:
        angle_delta = 2 * np.pi / n_vertices
        coords = np.array(
            [
                [
                    np.cos(angle_delta * i),
                    np.sin(angle_delta * i),
                ]
                for i in range(n_vertices)
            ]
        )
        x = middle + radius * coords
        dx = 0.02 * radius * np.random.rand(*x.shape)
        samples.append(x + dx)

    agents = []
    for pos in samples:
        agent = crt.Agent(
            pos.T,
            force_area=0.01,
            force_perimeter=0.1,
            force_dist=0.0001,
            force_angle=0.0001,
            min_dist=0.8 * radius,
            target_area=target_area,
            target_perimeter=target_perimeter,
            damping=0.3,
            diffusion_constant=0.0000,
        )
        agents.append(agent)

    result = crt.run_simulation(settings, agents)
    print()

    arglist = zip(
        result, itertools.repeat(settings.domain_size), itertools.repeat(result)
    )

    pool = mp.Pool()
    _ = list(tqdm(pool.imap(__pool_save_snapshot_helper, arglist), total=len(result)))
