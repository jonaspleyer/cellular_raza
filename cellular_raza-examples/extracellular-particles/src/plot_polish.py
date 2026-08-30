from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
import string
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.plot import plot_iteration, load_data

if __name__ == "__main__":
    s = 0.04
    fig = plt.figure(figsize=(24, 8 * (1 - 2 * s)))
    gs = GridSpec(1, 3, wspace=s, left=0, right=1, bottom=0, top=1)

    data = load_data()
    iterations = np.array(sorted(data.keys()))

    for i, j in enumerate([1, int(len(iterations) / 2), -1]):
        ax = fig.add_subplot(gs[i])
        ax.set_axis_off()
        plot_iteration(iterations[j], data, ax=ax)

        ax.text(
            0.03,
            0.97,
            string.ascii_uppercase[i],
            fontsize=40,
            fontweight="semibold",
            fontfamily="serif",
            va="top",
            horizontalalignment="left",
            transform=ax.transAxes,
        )

    fig.savefig("out/plot-polish.pdf")
