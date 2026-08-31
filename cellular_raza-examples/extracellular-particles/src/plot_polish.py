from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
import string
import sys
import os
import scipy as sp

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.plot import plot_iteration, load_data

DT = 0.1

COLOR1 = "#6bd2db"
COLOR2 = "#0ea7b5"
COLOR3 = "#0c457d"
COLOR4 = "#ffbe4f"
COLOR5 = "#e8702a"
COLOR6 = "#a02b08"


def set_mpl_rc_params():
    plt.rcParams.update(
        {
            "font.family": "Courier New",  # monospace font
            "font.size": 25,
            "axes.titlesize": 25,
            "axes.labelsize": 25,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
            "legend.fontsize": 25,
            "figure.titlesize": 25,
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


def bivariate_pcf(cells, particles, r, dr, area=None):
    cells = np.asarray(cells, float)
    particles = np.asarray(particles, float)

    if area is None:
        allpts = np.vstack([cells, particles])
        mins, maxs = allpts.min(0), allpts.max(0)
        area = np.prod(maxs - mins)
    lambda_particles = len(particles) / area

    edges = np.array([r - dr, r + dr])

    tree = sp.spatial.cKDTree(particles)
    counts_cum = np.array(
        [tree.query_ball_point(cells, e, return_length=True).sum() for e in edges]
    )
    counts_ring = counts_cum[1] - counts_cum[0]  # counts per annulus

    ring_area = np.pi * (edges[1] ** 2 - edges[0] ** 2)
    expected = ring_area * lambda_particles * len(cells)

    g = np.divide(
        counts_ring, expected, out=np.zeros_like(counts_ring, float), where=expected > 0
    )
    return g


if __name__ == "__main__":
    set_mpl_rc_params()
    s = 0.04
    fig = plt.figure(figsize=(24, 8 + 8 * (1 - 2 * s)))
    gs = GridSpec(
        2,
        1,
        wspace=s,
        left=0,
        right=1,
        bottom=0,
        top=1,
        height_ratios=(8, 8 * (1 - 2 * s)),
    )
    fig1 = fig.add_subfigure(gs[0])
    fig2 = fig.add_subfigure(gs[1])

    data = load_data()
    iterations = np.array(sorted(data.keys()))[::10]

    cell_area = []
    intracellular_particles = []
    extracellular_particles = []
    n_cells = []
    pcfs = []
    for it in iterations:
        cell_data = data[it]["cells"]
        subdomain_data = data[it]["subdomains"]

        i_particles = []
        areas = []
        n_cells.append(len(cell_data))
        cell_positions = []
        for c in cell_data:
            # Select the positions of every particle
            pi = np.array(c["particles"][0]).reshape((-1, 4))[:, :3]
            i_particles.append(pi)
            areas.append(c["interaction"]["radius"] ** 2 * np.pi)
            cell_positions.append(c["mechanics"]["pos"][:2])

        i_particles = np.vstack(i_particles)
        cell_positions = np.vstack(cell_positions)

        intracellular_particles.append(i_particles.shape[0])
        cell_area.append(areas)

        e_particles = np.vstack(
            [
                np.array(si["element"]["particles"][0]).reshape((-1, 4))[:, :2]
                for si in subdomain_data
            ]
        )
        extracellular_particles.append(e_particles.shape[0])

        g = bivariate_pcf(cell_positions, e_particles, r=15, dr=15)
        pcfs.append(g)

    pcfs = np.array(pcfs)
    t = np.array(iterations) * DT

    gs1 = GridSpec(
        1, 3, left=0.05, right=1 - 0.025, bottom=0.1, top=1 - 0.1, wspace=0.25
    )
    ax1 = fig1.add_subplot(gs1[0])
    ax1.plot(t, [np.sum(a) for a in cell_area], color=COLOR5, label="Cell Area")
    ax11 = ax1.twinx()
    ax11.plot(t, n_cells, color=COLOR3, linestyle="--", label="Cell Count")
    handles1, labels1 = ax11.get_legend_handles_labels()
    handles2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(
        [*handles1, *handles2],
        [*labels1, *labels2],
        frameon=False,
        loc="upper center",
        ncols=2,
        bbox_to_anchor=(0.45, 1.13),
    )
    ax1.set_xlabel("Time [min]")

    ax2 = fig1.add_subplot(gs1[1])

    y1 = [pi for pi in intracellular_particles]
    y2 = [pi for pi in extracellular_particles]
    ax2.stackplot(
        t,
        y1,
        y2,
        colors=[COLOR5, COLOR3],
        labels=["Intracellular", "Extracellular"],
        alpha=0.8,
    )
    i1 = np.argmin(t < 300)
    i2 = np.argmin(t < 800)
    ymax = y1[0] + y2[0]
    ylim = 1.1 * ymax
    ax2.set_ylim(0, ylim)
    ax2.axvline(x=t[i1], ymax=(y1[i1] + y2[i1]) / ylim, color="white", linewidth=4)
    ax2.axvline(x=t[i2], ymax=(y1[i2] + y2[i2]) / ylim, color="white", linewidth=4)
    ax2.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax2.set_xlabel("Time [min]")

    ax3 = fig1.add_subplot(gs1[2])
    configure_ax(ax3)
    ax3.plot(t, pcfs, color=COLOR3, label="Bivariate Pair CF")
    ax3.set_xlabel("Time [min]")
    ax3.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))

    for ax, label in zip([ax1, ax2], string.ascii_uppercase):
        configure_ax(ax)
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

    gs2 = GridSpec(1, 3, wspace=s, left=0, right=1, bottom=0, top=1)
    for i, j in enumerate([i1, i2, -1]):
        ax = fig2.add_subplot(gs2[i])
        ax.set_axis_off()
        plot_iteration(iterations[j], data, ax=ax)

        ax.text(
            0.03,
            0.97,
            string.ascii_uppercase[i + 3],
            fontsize=40,
            fontweight="semibold",
            fontfamily="serif",
            va="top",
            horizontalalignment="left",
            transform=ax.transAxes,
        )

    fig.savefig("out/plot-polish.pdf")
