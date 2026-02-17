import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
from tqdm import tqdm
import multiprocessing as mp
import itertools

import cr_tissue as crt


def __construct_polygon(path, resolution):
    polygon = []
    for segm in path:
        segment_ty = segm[0]
        if segment_ty == "line":
            x = np.array([segm[1], segm[2]])
            polygon.extend(x)
        elif segment_ty == "arc":
            p = segm[1]
            theta1 = segm[2]
            theta2 = segm[3]
            r = segm[4]
            theta = np.linspace(theta1, theta2, resolution)
            points = np.vstack(
                (
                    r * np.cos(theta) + p[0],
                    r * np.sin(theta) + p[1],
                )
            ).T[::-1]
            polygon.extend(points)

    return np.array(polygon)


def save_snapshot(iteration, domain_size, result, resolution=30):
    fig, ax = plt.subplots(figsize=(8, 8))
    for _, path, area_ratio in result[iteration]:
        color = mpl.colormaps["Spectral"](area_ratio)
        if len(path) > 0:
            polygon = __construct_polygon(path, resolution)
            ax.add_patch(
                mpl.patches.Polygon(
                    polygon,
                    facecolor=color,
                    linestyle="-",
                    edgecolor="k",
                )
            )

    centers = np.array([m[0] for m in result[iteration]])
    ax.scatter(centers[:, 0], centers[:, 1], marker=".", color="k")

    dx = domain_size
    ax.set_xlim(-0.01 * dx, 1.01 * domain_size)
    ax.set_ylim(-0.01 * dx, 1.01 * domain_size)

    fig.savefig(f"out/{iteration:010}.png")
    plt.close(fig)


def __pool_save_snapshot_helper(args):
    return save_snapshot(*args)


if __name__ == "__main__":
    settings = crt.SimulationSettings()

    domain_size = 500
    domain_size_start = 200
    n_agents = 700

    settings.dt = 0.5
    settings.t_max = 20_000.0
    settings.save_interval = 200.0
    settings.domain_size = domain_size
    settings.n_voxels = 50
    settings.approximation_steps = 10

    radius = 5.0
    target_area = np.pi * radius**2
    target_perimeter = 2 * np.pi * radius * 1.0

    sampler = sp.stats.qmc.LatinHypercube(d=2, seed=10)
    domain_size = settings.domain_size
    samples = sampler.random(n_agents)
    dlow = domain_size / 2 - domain_size_start / 2
    dhigh = domain_size / 2 + domain_size_start / 2
    samples = sp.stats.qmc.scale(samples, dlow, dhigh)

    # samples = np.array(
    #     [
    #         [0.49 * domain_size, 0.49 * domain_size],
    #         [0.49 * domain_size, 0.51 * domain_size],
    #         [0.51 * domain_size, 0.49 * domain_size],
    #         [0.51 * domain_size, 0.51 * domain_size],
    #         [0.55 * domain_size, 0.50 * domain_size],
    #     ]
    # )

    agents = []
    for pos in samples:
        agent = crt.Agent(
            pos,
            force_area=0.0001,
            force_perimeter=0.0001,
            force_dist=0.1,
            min_dist=0.6 * radius,
            target_area=target_area,
            target_perimeter=target_perimeter,
            damping=1.0,
            diffusion_constant=0.0001,
        )
        agents.append(agent)

    result = crt.run_simulation(settings, agents)
    print()

    arglist = zip(
        result, itertools.repeat(settings.domain_size), itertools.repeat(result)
    )

    pool = mp.Pool()
    _ = list(tqdm(pool.imap(__pool_save_snapshot_helper, arglist), total=len(result)))
