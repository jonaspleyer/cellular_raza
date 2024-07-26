import matplotlib.pyplot as plt
import numpy as np
from glob import glob

if __name__ == "__main__":
    files = glob("tests/*.csv")

    for file in files:
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

        t = results[:,0]
        gerror = results[:,2]
        lerror = results[:,4]
        res_exact = results[:,6]

        fig, ax = plt.subplots()
        # ax.fill_between(t, -lerror, lerror, alpha=0.2, color="green")
        ylim_low = np.inf
        ylim_high = - np.inf
        # ax.errorbar(t, 0*t, 0.002 * res_exact, label="0.2% Analytical Solution", color="k", linestyle="--")
        # ax.plot(t, gerror, label="Global truncation Error", color="blue")
        for n in range(7, results.shape[1]):
            if n % 2 == 0:
                ax.plot(t, results[:,n], label="Solution {:1.0f}".format(n), linestyle="--")
                # ax.plot(t, res_exact - results[:,n], label="Solution {:1.0f}".format(n))
                # ylim_low = min(np.min(res_exact - results[:,n]), ylim_low, 0)
                # ylim_high = max(np.max(res_exact - results[:,n]), ylim_high, 0)
        ax.errorbar(
            t,
            res_exact,
            gerror,
            label="Analytical Solution",
            linestyle=":",
            color="k",
            alpha=0.5
        )
        dx = ylim_high - ylim_low
        margin = 0.1
        # ax.set_ylim(ylim_low - margin * dx, ylim_high + margin * dx)
        ax.legend()
        fig.tight_layout()
        fig.savefig(file.replace(".csv", ".png"))
