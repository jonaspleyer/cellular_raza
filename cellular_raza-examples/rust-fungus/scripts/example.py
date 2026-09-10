import matplotlib.pyplot as plt
from matplotlib import rc_context
import numpy as np
from pathlib import Path
from matplotlib.patches import Circle, Rectangle, Polygon, Arc
import string

import cr_rust_fungus as crf


POS_PLANT_CELL = np.array(
    [
        [51.08850804, 31.5084175],
        [51.03684374, 32.31141754],
        [51.09340809, 33.11760559],
        [51.16790736, 33.92134747],
        [51.13762193, 34.70647444],
        [50.74718319, 35.38282616],
        [50.0839777, 35.81932956],
        [49.29361357, 36.01024525],
        [48.46825535, 36.04664108],
        [47.63730791, 36.03229772],
        [46.80526594, 36.03443942],
        [45.9734252, 36.06687566],
        [45.14192885, 36.09207278],
        [44.31588987, 36.03615876],
        [43.52649951, 35.81423527],
        [42.85084407, 35.36718549],
        [42.40250076, 34.7031364],
        [42.28255641, 33.91961963],
        [42.32090349, 33.11074379],
        [42.37188646, 32.29685487],
        [42.39152068, 31.48077207],
        [42.35774109, 30.66541126],
        [42.26051608, 29.85699799],
        [42.15461583, 29.07591411],
        [42.43973203, 28.39286091],
        [43.10537175, 28.0887864],
        [43.8693455, 28.22615695],
        [44.66259466, 28.4006257],
        [45.4685328, 28.52704353],
        [46.28366925, 28.58675261],
        [47.10225773, 28.56847437],
        [47.9167426, 28.46817236],
        [48.72044669, 28.29074739],
        [49.51134214, 28.05378175],
        [50.287681, 27.82798122],
        [51.0563427, 27.93105536],
        [51.59920242, 28.45268352],
        [51.73949212, 29.20811517],
        [51.5230743, 29.96350514],
        [51.26336855, 30.72389781],
    ]
)


def plot_schematic(ax):
    # Pick one particular agent
    pos = POS_PLANT_CELL

    # Plot the cell
    ax.plot(
        [*pos[:, 0], pos[0, 0]],
        [*pos[:, 1], pos[0, 1]],
        marker="o",
        color="k",
        linestyle=":",
        markersize=5,
        linewidth=2,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    # Plot zoomed in variant
    xmin = np.min(pos[:, 0])
    xmax = np.max(pos[:, 0])
    ymin = np.min(pos[:, 1])
    ymax = np.max(pos[:, 1])
    ds = max(xmax - xmin, ymax - ymin)

    # Define spacing used for all schema
    dss = 0.4 * ds

    ax.set_xlim(xmin - 0.15 * ds, xmin + 1.15 * ds)
    ax.set_ylim(ymin - 0.15 * ds, ymin + 1.15 * ds)

    def plot_schema_1():
        # WARNING: magic numbers!
        n1 = 10
        n2 = 15
        q = np.array(pos[n1:n2])

        xy = (xmin + 0.1 * ds, ymin + 0.7 * ds)
        rect = Rectangle(xy, dss, dss, facecolor="white", edgecolor="gray", zorder=100)
        ax.add_patch(rect)
        ax.text(
            xy[0] + 0.03 * ds,
            xy[1] + dss - 0.03 * ds,
            "i",
            fontsize=30,
            fontweight="semibold",
            fontfamily="serif",
            va="top",
            horizontalalignment="left",
            zorder=101,
        )

        # Expand to rectangle
        r = 1.3 * (q - q[2]) + np.array(xy) + np.array([0.5 * dss, 0.6 * dss])
        # Clip to rectangle
        r[:, 0] = np.clip(r[:, 0], xy[0], xy[0] + dss)
        ax.plot(
            r[1:-1, 0],
            r[1:-1, 1],
            color="k",
            linewidth=2,
            zorder=101,
            marker="o",
            linestyle=":",
        )
        ax.plot(r[:2, 0], r[:2, 1], color="k", linewidth=2, zorder=101, linestyle=":")
        ax.plot(r[-2:, 0], r[-2:, 1], color="k", linewidth=2, zorder=101, linestyle=":")

        for i in range(len(r) - 1):
            p = 0.5 * (r[i] + r[i + 1])
            z = r[i] - r[i + 1]
            dir = 0.7 * np.array([-z[1], z[0]])
            ax.annotate(
                "", xy=p + dir, xytext=p, arrowprops=dict(arrowstyle="->"), zorder=101
            )

    def plot_schema_2():
        xy = (xmin + 0.7 * ds, ymin + 0.3 * ds)
        rect = Rectangle(xy, dss, dss, facecolor="white", edgecolor="gray", zorder=100)
        ax.add_patch(rect)
        ax.text(
            xy[0] + 0.03 * ds,
            xy[1] + dss - 0.03 * ds,
            "ii",
            fontsize=30,
            fontweight="semibold",
            fontfamily="serif",
            va="top",
            horizontalalignment="left",
            zorder=101,
        )

        # WARNING: magic numbers!
        n1 = -1
        n3 = 4
        # Clip selected points to retangle
        q = np.array([pos[n1], pos[n3]])
        q[:, 1] = np.clip(q[:, 1], xy[1], xy[1] + dss)

        with rc_context({"path.sketch": (10, 10, 1)}):
            for i in range(len(q) - 1):
                ax.plot(
                    q[i : i + 2, 0],
                    q[i : i + 2, 1],
                    color="k",
                    zorder=101,
                )

    def plot_schema_3():
        # WARNING: magic numbers!
        n1 = 22
        n2 = 27

        xy = (xmin - 0.1 * ds, ymin - 0.1 * ds)
        rect2 = Rectangle(xy, dss, dss, facecolor="white", edgecolor="gray", zorder=100)
        ax.add_patch(rect2)
        ax.text(
            xy[0] + dss - 0.03 * ds,
            xy[1] + dss - 0.03 * ds,
            "iii",
            fontsize=30,
            fontweight="semibold",
            fontfamily="serif",
            va="top",
            horizontalalignment="right",
            zorder=101,
        )

        q = pos[n1:n2]
        dmid = (q[1] + q[2] + q[3]) / 3
        dd = max(np.max(q[:, 0]) - np.min(q[:, 0]), np.max(q[:, 1]) - np.min(q[:, 1]))

        # Clamp to intervals [-1, 1]
        r = (q - dmid) / dd
        # Expand and clamp to rectangle boundaries
        q = dmid + 0.5 * ds * r
        q = np.clip(q, xy, [xy[0] + dss, xy[1] + dss])

        ax.plot(
            q[1:-1, 0],
            q[1:-1, 1],
            color="k",
            linestyle=":",
            linewidth=2,
            zorder=101,
            marker="o",
            markersize=5,
        )
        ax.plot(q[:2, 0], q[:2, 1], color="k", linestyle=":", linewidth=2, zorder=101)
        ax.plot(q[-2:, 0], q[-2:, 1], color="k", linestyle=":", linewidth=2, zorder=101)

        for i in range(1, len(q) - 1):
            p1 = q[i - 1]
            p2 = q[i]
            p3 = q[i + 1]
            c1 = p1 - p2
            c2 = p3 - p2

            a1 = np.arctan2(c1[1], c1[0]) / (2 * np.pi) * 360
            a2 = np.arctan2(c2[1], c2[0]) / (2 * np.pi) * 360

            arc = Arc(p2, 1.5, 1.5, theta1=a2, theta2=a1, zorder=101)
            ax.add_patch(arc)

    plot_schema_1()
    plot_schema_2()
    plot_schema_3()


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
                facecolor="#96ab6f",  # "#909090",
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
            plot_fungus(cell, facecolor="#ffbe4f")
            ax.plot(pos[:, 0], pos[:, 1], color="gray", marker="+", linestyle=":")

    dx = domain_size
    ax.set_xlim(-0.01 * dx, 1.01 * domain_size)
    ax.set_ylim(-0.01 * dx, 1.01 * domain_size)
    ax.set_xticks([])
    ax.set_yticks([])

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


def generate_agents(settings, fungal=True):
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
        xlim=[27.0, 60.0],
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
            force_dist=0.005,
            force_angle=0.0001,
            interaction_range=radius / 10,
            min_dist=0.8 * radius,
            target_area=target_area,
            target_perimeter=target_perimeter,
            damping=0.3,
            diffusion_constant=0.0000,
        )
        plant_cells.append(agent)

    if fungal:
        # Create Fungus Cells now
        spring_length = 2.0
        pos = spring_length * np.arange(11)
        pos = np.array([0 * pos + settings.domain_size / 2, pos + 35 - pos[-1]])
        fungal_cells = [
            crf.Fungus(
                pos,
                diffusion_constant=0.0,
                spring_tension=0.002,
                rigidity=0.0025,
                spring_length=spring_length,
                damping=0.03,
                radius=1.5,
                potential_stiffness=0.05,
                cutoff=2.0,
                strength=0.05,
                growth_rate=0.0010,
                # fixed_pos=None,
                fixed_pos=(pos.shape[1] - 1, pos[:, -1]),
            )
        ]

        # Combine cells and run simulation
        agents = [*plant_cells, *fungal_cells]
    else:
        agents = plant_cells

    return agents


if __name__ == "__main__":
    settings = crf.SimulationSettings()
    # Update Settings
    settings.t_max = 5_000.0
    settings.dt = 5.0
    settings.save_interval = 50.0
    settings.domain_size = 60.0
    settings.n_voxels = 4

    settings.rng_seed = 0
    settings.perimeter_mod = 1.2

    agents = generate_agents(settings, fungal=False)
    path = crf.run_simulation(settings, agents)
    iterations = crf.get_all_iterations(path)
    final_agents = crf.load_results(iterations[-1], path)

    settings.t_max = 2_000.0
    fungus = generate_agents(settings, fungal=True)[-1]
    path = crf.run_simulation(settings, [*final_agents, fungus])

    fig, axs = plt.subplots(1, 3, figsize=(24, 8 * 1.04))

    plot_schematic(axs[0])

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

    iterations = crf.get_all_iterations(path)
    i1 = iterations[1]
    i2 = iterations[-1]
    for i, iteration in enumerate([i1, i2]):
        cells = crf.load_results(iteration, path)
        save_snapshot(
            cells,
            settings.domain_size,
            fig_ax=(
                fig,
                axs[i + 1],
            ),
        )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)
    fig.savefig("rust-fungus-mechanics.pdf")
