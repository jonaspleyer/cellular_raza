import matplotlib.pyplot as plt
from matplotlib import rc_context
import numpy as np
from pathlib import Path
from matplotlib.patches import Circle, Rectangle, Polygon, Arc
import string

import cr_rust_fungus as crf


def save_snapshot(
    cells,
    domain_size,
    opath=Path("out"),
    savename=None,
    fig_ax=None,
):
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig, ax = fig_ax

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

    for cell in cells:
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

    if fig_ax is None:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if savename is not None:
        fig.savefig(opath / f"{savename}.png")
    if savename is not None or fig_ax is None:
        plt.close(fig)


def midpoints_gen(n_agents, xlim, ylim):
    midpoints = []

    x0 = xlim[0]
    dx = (xlim[1] - xlim[0]) / n_agents[0]
    y0 = ylim[0]
    dy = (ylim[1] - ylim[0]) / n_agents[1]

    for row in range(n_agents[1]):
        for col in range(n_agents[0]):
            midpoints.append([x0 + dx * (col + 0.5), y0 + dy * (row + 0.5)])

    return midpoints


def run_or_load_sim(settings):
    # Try loading previous result
    path = crf.find_results(settings)
    if path is not None:
        print("Loaded Result from", path)
        return path

    rng = np.random.default_rng(settings.rng_seed)

    midpoints1 = midpoints_gen(
        n_agents=[3, 3],
        xlim=[0.0, 27.0],
        ylim=[14.0, 35.0],
    )
    midpoints2 = midpoints_gen(
        n_agents=[3, 3],
        xlim=[33.0, 60.0],
        ylim=[14.0, 35.0],
    )
    midpoints3 = midpoints_gen(
        n_agents=[2, 1],
        xlim=[0.0, 20.0],
        ylim=[7.0, 14.0],
    )
    midpoints4 = midpoints_gen(
        n_agents=[3, 1],
        xlim=[30.0, 60.0],
        ylim=[7.0, 14.0],
    )
    midpoints5 = midpoints_gen(
        n_agents=[8, 1],
        xlim=[0.0, 60.0],
        ylim=[0.0, 7.0],
    )

    midpoints = np.array(
        [*midpoints1, *midpoints2, *midpoints3, *midpoints4, *midpoints5]
    )

    radius = 4.45
    radius_variance = 0.5
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
        target_perimeter = 2 * np.pi * radius * settings.perimeter_mod

        agent = crf.PlantCell(
            pos.T,
            force_area=0.001,
            force_perimeter=0.025,
            force_dist=0.002,
            force_angle=0.0001,
            interaction_range=radius / 10,
            min_dist=0.8 * radius,
            target_area=target_area,
            target_perimeter=target_perimeter,
            damping=0.3,
            diffusion_constant=0.0000,
        )
        plant_cells.append(agent)

    # Create Fungus Cells now
    spring_length = 3.0
    pos = spring_length * (np.arange(11) - 3.5)
    pos = np.array([0 * pos + settings.domain_size / 2, pos + settings.domain_size / 4])
    pos[:, 0] = [22, 10]
    pos[:, 1] = [25, 11]
    pos[:, 2] = [27, 12]
    pos[:, 3] = [29, 13]
    fungal_cells = [
        crf.Fungus(
            pos,
            diffusion_constant=0.0,
            spring_tension=0.001,
            rigidity=0.00002,
            spring_length=spring_length,
            damping=0.03,
            radius=1.7,
            potential_stiffness=0.05,
            cutoff=2.0,
            strength=0.05,
        )
    ]

    # Combine cells and run simulation
    agents = [*plant_cells, *fungal_cells]
    result = crf.run_simulation(settings, agents)
    print()
    return result


if __name__ == "__main__":
    settings = crf.SimulationSettings()
    # Update Settings
    settings.t_max = 20_000.0
    settings.dt = 5.0
    settings.save_interval = 1_000.0
    settings.domain_size = 60.0
    settings.n_voxels = 4

    parameters = [
        (0, 1.00),
        (1, 1.20),
    ]

    results = []
    for seed, perimeter_mod in parameters:
        settings.rng_seed = seed
        settings.perimeter_mod = perimeter_mod

        path = run_or_load_sim(settings)
        iterations = crf.get_all_iterations(path)
        final_agents = crf.load_results(iterations[-1], path)
        results.append(final_agents)

    # iterations = list(sorted(result.keys()))
    # for iteration in tqdm(iterations):
    #     save_snapshot(iteration, settings.domain_size, result)

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    # Pick one particular agent
    ax = axs[0]
    plant_cell = results[-1][int(len(results[-1]) / 2)]
    pos = plant_cell[0].position.T

    ax.plot(
        [*pos[:, 0], pos[0, 0]],
        [*pos[:, 1], pos[0, 1]],
        marker="x",
        color="k",
        linestyle="--",
    )

    # Plot zoomed in variant
    xmin = np.min(pos[:, 0])
    xmax = np.max(pos[:, 0])
    ymin = np.min(pos[:, 1])
    ymax = np.max(pos[:, 1])
    ds = max(xmax - xmin, ymax - ymin)

    ax.set_xlim(xmin - 0.12 * ds, xmin + 1.12 * ds)
    ax.set_ylim(ymin - 0.12 * ds, ymin + 1.12 * ds)

    # Select which points to display with more detail
    sxmin = xmin + 0.7 * ds
    symin = ymin + 0.3 * ds
    dss = 0.4 * ds

    xy = (sxmin, symin)
    rect = Rectangle(xy, dss, dss, facecolor="white", edgecolor="gray", zorder=100)
    ax.add_patch(rect)

    # WARNING: magic numbers ahead!
    n1 = -1
    n3 = 4
    # Clip selected points to retangle
    q = np.array([pos[n1], pos[n3]])
    q[:, 1] = np.clip(q[:, 1], symin, symin + dss)

    with rc_context({"path.sketch": (10, 10, 1)}):
        for i in range(len(q) - 1):
            ax.plot(
                q[i : i + 2, 0],
                q[i : i + 2, 1],
                color="k",
                zorder=101,
                markersize=5,
            )
    # WARNING: second time magic numbers!
    n1 = 22
    n2 = 27

    xy = (xmin - 0.1 * ds, ymin - 0.1 * ds)
    rect2 = Rectangle(xy, dss, dss, facecolor="white", edgecolor="gray", zorder=100)
    ax.add_patch(rect2)

    q = pos[n1:n2]
    dmid = (q[1] + q[2] + q[3]) / 3
    dd = max(np.max(q[:, 0]) - np.min(q[:, 0]), np.max(q[:, 1]) - np.min(q[:, 1]))

    # Clamp to intervals [-1, 1]
    r = (q - dmid) / dd
    # Expand and clamp to rectangle boundaries
    q = dmid + 0.5 * ds * r
    q = np.clip(q, xy, [xy[0] + dss, xy[1] + dss])

    ax.plot(q[:, 0], q[:, 1], color="k", linestyle="--", linewidth=2, zorder=101)

    for i in range(1, len(q) - 1):
        p1 = q[i - 1]
        p2 = q[i]
        p3 = q[i + 1]
        c1 = p1 - p2
        c2 = p3 - p2

        a1 = np.arctan2(c1[1], c1[0]) / (2 * np.pi) * 360
        a2 = np.arctan2(c2[1], c2[0]) / (2 * np.pi) * 360

        arc = Arc(p2, 1.2, 1.2, theta1=a2, theta2=a1, zorder=101)
        ax.add_patch(arc)

    for ax, label in zip(axs, string.ascii_uppercase):
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
        )

    for i, cells in enumerate(results):
        save_snapshot(
            cells,
            settings.domain_size,
            fig_ax=(
                fig,
                axs[i + 1],
            ),
        )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)
    fig.savefig("temp.png")
