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
        for n in range(7, results.shape[1]):
            if n % 2 == 0:
                ax.plot(t, results[:,n], label="Solution {:1.0f}".format(n), linestyle="--")
        ax.errorbar(
            t,
            res_exact,
            gerror,
            label="Analytical Solution",
            linestyle=":",
            color="k",
            alpha=0.5
        )
        ax.set_title("cellular_raza/" + str(file))
        ax.legend()
        fig.tight_layout()
        fig.savefig(file.replace(".csv", ".png"))
