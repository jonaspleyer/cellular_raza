import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib.patches import Circle, Rectangle, Polygon
import string

import cr_rust_fungus as crf


def save_snapshot(iteration, domain_size, result, opath=Path("out"), prefix=""):
    fig, ax = plt.subplots(figsize=(12, 12))

    def plot_plant(cell):
        pos = np.array(cell[0].position).T
        ax.add_patch(
            Polygon(
                pos,
                facecolor="#909090",
                linestyle="-",
                edgecolor="k",
                alpha=0.5,
            )
        )

    def plot_fungus(cell, edgecolor=None, facecolor=None):
        for pi in pos:
            circ = Circle(
                pi,
                radius=cell[0].radius,
                edgecolor=edgecolor,
                facecolor=facecolor,
            )
            ax.add_patch(circ)
        for i in range(pos.shape[0] - 1):
            p1 = pos[i]
            p2 = pos[i + 1]
            z = p2 - p1
            width = float(np.linalg.norm(z))
            angle = np.arctan2(-z[1], z[0]) % (2 * np.pi)
            dir = np.array([-z[1], z[0]]) / width
            r = cell[0].radius
            rect = Rectangle(
                p1 - dir * r,
                width,
                height=2 * r,
                angle=-angle * 360 / 2 / np.pi,
                edgecolor=edgecolor,
                facecolor=facecolor,
            )
            ax.add_patch(rect)

    for cell in result[iteration]:
        pos = np.array(cell[0].position).T
        if not cell.is_fungus():
            plot_plant(cell)
        else:
            plot_fungus(cell, edgecolor="gray", facecolor="gray")
            plot_fungus(cell, facecolor="#c0e384")
            ax.plot(pos[:, 0], pos[:, 1], color="gray", marker="+", linestyle=":")

    dx = domain_size
    ax.set_xlim(-0.01 * dx, 1.01 * domain_size)
    ax.set_ylim(-0.01 * dx, 1.01 * domain_size)

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(opath / f"{prefix}{iteration:010}.png")
    plt.close(fig)


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

    settings.dt = 8.0
    settings.t_max = 20_000.0
    settings.save_interval = 1_000.0
    settings.domain_size = domain_size
    settings.n_voxels = n_voxels

    try:
        initial_cells = crf.load_cells("out/initial_plant_cells.json")
    except:
        midpoints = midpoints_gen(
            n_agents=[9, 4],
            xlim=[0.0, 60.0],
            ylim=[0.0, 28.0],
        )
        midpoints = np.array(midpoints)

        radius = 3.8
        radius_variance = 1.0
        radii = radius + radius_variance * (0.5 - rng.random(midpoints.shape[0]))

        # Generate a polygon for each starting point
        n_vertices = 40
        plant_cells = []
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
                damping=0.3,
                diffusion_constant=0.0000,
            )
            plant_cells.append(agent)

        result = crf.run_simulation(settings, plant_cells)
        print()

        final_iter = list(sorted(result.keys()))[-1]
        initial_cells = [k[0] for k in result[final_iter]]
        crf.store_cells(initial_cells, "out/initial_plant_cells.json")

        iterations = list(sorted(result.keys()))
        save_snapshot(iterations[0], settings.domain_size, result, prefix="pre")
        save_snapshot(iterations[1], settings.domain_size, result, prefix="pre")
        save_snapshot(iterations[-1], settings.domain_size, result, prefix="pre")

    # Create Fungus Cells now
    spring_length = 3.0
    pos = spring_length * (np.arange(8) - 3.5)
    pos = np.array([pos + domain_size / 2, 0 * pos + domain_size * 0.5])
    fungal_cells = [
        crf.Fungus(
            pos,
            diffusion_constant=0.0,
            spring_tension=0.001,
            rigidity=0.00002,
            spring_length=spring_length,
            damping=0.10,
            radius=1.7,
            potential_stiffness=0.05,
            cutoff=2.0,
            strength=0.05,
        )
    ]

    # Update Settings
    settings.t_max = 3_000.0
    settings.dt = 5.0
    settings.save_interval = 200.0

    # Combine cells and run simulation
    agents = [*initial_cells, *fungal_cells]
    result = crf.run_simulation(settings, agents)

    for iteration in tqdm(result, total=len(result)):
        save_snapshot(iteration, settings.domain_size, result)
