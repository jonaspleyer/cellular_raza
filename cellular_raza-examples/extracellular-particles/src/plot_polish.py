from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
import string
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

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


from src.plot import plot_iteration, load_data

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
    for it in iterations:
        cell_data = data[it]["cells"]
        subdomain_data = data[it]["subdomains"]
        particles_cells = []
        areas = []
        n_cells.append(len(cell_data))
        for c in cell_data:
            # Select the positions of every particle
            pi = np.array(c["particles"][0]).reshape((-1, 4))[:, :3]
            particles_cells.append(pi.shape[0])
            areas.append(c["interaction"]["radius"] ** 2 * np.pi)

        intracellular_particles.append(np.sum(particles_cells))
        cell_area.append(areas)

        extracellular_particles.append(
            np.sum(
                [
                    np.array(si["element"]["particles"][0]).reshape((-1, 4)).shape[0]
                    for si in subdomain_data
                ]
            )
        )

    t = np.array(iterations) * DT

    gs1 = GridSpec(1, 3, left=0.05, right=1 - 0.05, bottom=0.1, top=1 - 0.1)
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
        bbox_to_anchor=(0.5, 1.1),
    )
    ax1.set_xlabel("Time [min]")

    prev = 10000
    for i in range(len(intracellular_particles)):
        new = intracellular_particles[i] + extracellular_particles[i]
        # print(new, prev)
        # assert new <= prev
        prev = new

    ax2 = fig1.add_subplot(gs1[1])
    ax2.stackplot(
        t,
        [pi for pi in intracellular_particles],
        [pi for pi in extracellular_particles],
        colors=[COLOR5, COLOR3],
        labels=["Intracellular", "Extracellular"],
        alpha=0.8,
    )
    ax2.legend(
        frameon=False,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
    )
    ax2.set_xlabel("Time [min]")

    # ax3 = fig1.add_subplot(gs1[2])
    fig13 = fig1.add_subfigure(gs1[2])
    n_plots = 5
    gs13 = GridSpec(n_plots, 1, hspace=0, bottom=0.1, top=1 - 0.1)
    # , left=0, right=1, bottom=0, top=1, wspace=0)
    ax_prev = None
    for i in range(n_plots):
        it = min(int(len(cell_area) / n_plots * (i + 1)), len(cell_area) - 1)
        ax = fig13.add_subplot(gs13[i], sharex=ax_prev)
        ax_prev = ax
        ax.hist(
            cell_area[it],
            edgecolor=COLOR3,
            bins=10,
            density=True,
            facecolor=COLOR1,
            alpha=0.8,
            label=f"t={t[it] / 60:2.0f}h",
        )
        ax.legend(frameon=False, loc="upper right")
        ax.set_ylim(0, 0.039)
        if i != n_plots - 1:
            ax.set_xticks([])  # ["" for _ in ax.get_xticklabels()])
        if i == 0:
            ax.text(
                0.03,
                1 - n_plots * 0.03,
                "C",
                fontsize=40,
                fontweight="semibold",
                fontfamily="serif",
                va="top",
                horizontalalignment="left",
                transform=ax.transAxes,
            )

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
    for i, j in enumerate([1, int(len(iterations) / 4), -1]):
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
