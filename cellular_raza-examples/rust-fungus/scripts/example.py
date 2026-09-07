import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
from tqdm import tqdm
import multiprocessing as mp
import itertools
import json

import cr_rust_fungus as crf


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
                alpha=0.5,
            )
        )

    dx = domain_size
    ax.set_xlim(-0.01 * dx, 1.01 * domain_size)
    ax.set_ylim(-0.01 * dx, 1.01 * domain_size)

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(f"out/{iteration:010}.png")
    plt.close(fig)


def __pool_save_snapshot_helper(args):
    return save_snapshot(*args)


def midpoints_gen(n_agents, xlim, ylim):
    midpoints = []

    x0 = xlim[0]
    dx = (xlim[1] - xlim[0]) / n_agents[0]
    y0 = ylim[0]
    dy = (ylim[1] - ylim[0]) / n_agents[1]

    # dx = domain_size_start_x / n
    # dy = domain_size_start_y / n * np.sqrt(3) / 2
    # xlow = domain_size / 2 - 0.5 * dx * n + dx / 4
    # ylow = domain_size / 2 - 0.5 * dy * n + dy / 2
    for row in range(n_agents[1]):
        for col in range(n_agents[0]):
            midpoints.append([x0 + dx * (col + 0.5), y0 + dy * (row + 0.5)])

    return midpoints


if __name__ == "__main__":
    settings = crf.SimulationSettings()
    rng = np.random.default_rng(settings.rng_seed)

    domain_size = 60
    n_voxels = 4

    settings.dt = 5.0
    settings.t_max = 2000.0
    settings.save_interval = 100.0
    settings.domain_size = domain_size
    settings.n_voxels = n_voxels

    try:
        with open("out/initial_cells.json", "r") as f:
            initial_agents = json.load(f)
    except:
        midpoints = midpoints_gen(
            n_agents=[9, 3],
            xlim=[0.0, 60.0],
            ylim=[0.0, 20.0],
        )
        midpoints = np.array(midpoints)

        radius = 3.8
        radius_variance = 1.0
        radii = radius + radius_variance * (0.5 - rng.random(midpoints.shape[0]))

        # Generate a polygon for each starting point
        n_vertices = 40
        agents = []
        for middle, radius in zip(midpoints, radii):
            # Calculate randomly placed points around centers
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
            dx = 0.1 * radius * rng.random(x.shape)
            pos = x + dx

            # Calculate Target Area and perimeter
            target_area = np.pi * radius**2
            target_perimeter = 2 * np.pi * radius * 1.025

            agent = crf.PlantCell(
                pos.T,
                force_area=0.001,
                force_perimeter=0.025,
                force_dist=0.002,
                force_angle=0.0001,
                interaction_range=radius / 5,
                min_dist=0.8 * radius,
                target_area=target_area,
                target_perimeter=target_perimeter,
                damping=0.1,
                diffusion_constant=0.0000,
            )
            agents.append(agent)

        result = crf.run_simulation(settings, agents)
        print()

        final_iter = list(sorted(result.keys()))[-1]
        final_cells = [k[0] for k in result[final_iter]]
        crf.store_cells(final_cells, "out/initial_cells.json")

    # arglist = zip(
    #     result, itertools.repeat(settings.domain_size), itertools.repeat(result)
    # )

    # pool = mp.Pool()
    # _ = list(tqdm(pool.imap(__pool_save_snapshot_helper, arglist), total=len(result)))
