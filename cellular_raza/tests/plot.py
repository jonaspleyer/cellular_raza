import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from plot_brownian_langevin import (
    set_mpl_rc_params,
    configure_ax,
    COLOR1,
    COLOR2,
    COLOR3,
    COLOR4,
    COLOR5,
)
import string

if __name__ == "__main__":
    files = sorted(glob("tests/*.csv"))

    set_mpl_rc_params()

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    for i, file, label in zip(range(len(files)), files, string.ascii_uppercase):
        ax = axs[i]
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
        # One line in such a file has the following entries
        # (
        #   t,
        #   gerror_bound0, gerror_bound1,
        #   lerror_bound0, lerror_bound1,
        #   res_exact0, res_exact_1,
        #   res_cr0_0, res_cr1_0,
        #   res_cr1_0, res_cr1_1,
        #   ...
        # )
        results = np.genfromtxt(file, delimiter=",")

        t = results[:, 0]
        gerror = results[:, 2]
        lerror = results[:, 4]
        res_exact = results[:, 6]

        colors = [COLOR1, COLOR2, COLOR3, COLOR4, COLOR5]
        for n in range(7, results.shape[1]):
            if n % 2 == 0:
                color = colors[int(n / 2) - 3]
                ax.plot(
                    t,
                    results[:, n],
                    label=f"Cell {int(n / 2) - 3:1.0f}",
                    linestyle="--",
                    color=color,
                )
        ax.errorbar(
            t, res_exact, gerror, label="Exact", linestyle=":", color="k", alpha=0.5
        )
        ax.legend(frameon=False)
        ax.set_xlabel("Time")
        ax.set_ylabel("Intracellular Concentration y(t)")
        ax.set_title(f"Test {i + 1}")

    fig.tight_layout()
    fig.savefig("out/contact_reactions.pdf")
