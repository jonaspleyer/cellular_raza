import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
import scipy as sp

def plot_results(
        storage_name: str,
        diffusion_constant: float,
        dimension: int,
        dt: float,
    ):
    print(storage_name)
    last_save_dir = list(sorted(glob("out/{}/*".format(storage_name))))[-1]

    # Obtain all values for cells
    iterations_cells = []
    for iteration_dir in sorted(glob(last_save_dir + "/cells/json/*")):
        cells = []
        for batch in list(sorted(glob(iteration_dir + "/*"))):
            with open(batch) as f:
                cells.extend(json.load(f)["data"])
        iterations_cells.append(cells)

    # Obtain all values for subdomains
    iteration_subdomains = []
    for iteration_dir in sorted(glob(last_save_dir + "/subdomains/json/*")):
        subdomains = []
        for single in list(sorted(glob(iteration_dir + "/*"))):
            with open(single) as f:
                subdomains.append(json.load(f)["element"])
        iteration_subdomains.append(subdomains)

    # Calculate the trajectories
    trajectories = np.array(
        np.array([[
            values_at_iter[j]["element"][0]["cell"]["pos"]
            for values_at_iter in iterations_cells
        ] for j in range(len(iterations_cells[0]))]
    ))

    # Get Domain size
    domain_min = np.array(iteration_subdomains[0][0]["domain_min"])
    domain_max = np.array(iteration_subdomains[0][0]["domain_max"])
    domain_middle = 0.5 * (domain_min + domain_max)
    dh = np.max(np.abs(trajectories - domain_middle), axis=(0,1))

    if trajectories.shape[2] == 2:
        # Plot the obtained results for each iteration
        s = dh[0] / dh[1]
        lim_lower = domain_middle - 1.1 * dh
        lim_upper = domain_middle + 1.1 * dh
        xlim = [lim_lower[0], lim_upper[0]]
        ylim = [lim_lower[1], lim_upper[1]]

        fig, ax = plt.subplots(figsize=(8, s*8))
        ax.set_title("Trajectories")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        for traj in trajectories:
            ax.plot(traj[:,0], traj[:,1], color="gray", linestyle=":")
        fig.tight_layout()
        fig.savefig("{}/trajectories.png".format(last_save_dir))

        heatmap, _, _ = np.histogram2d(
            trajectories[:,:,0].reshape((-1,)),
            trajectories[:,:,1].reshape((-1,)),
            range=[xlim, ylim],
            bins=50
        )
        extent = [*lim_lower, *lim_upper]
        fig, ax = plt.subplots(figsize=(8, 8*s))
        ax.imshow(heatmap.T, extent=extent, origin='lower')
        ax.set_title("Heatmap of explored space")
        fig.tight_layout()
        fig.savefig("{}/heatmap.png".format(last_save_dir))

    # Plot the mean squared displacement per iteration
    msd = np.mean(np.sum((trajectories - domain_middle)**2, axis=2), axis=0)
    msd_err = np.std(np.sum((trajectories - domain_middle)**2, axis=2), axis=0)
    fig, ax = plt.subplots()
    x = np.arange(len(msd))
    ax.plot(x, msd, color="gray", linestyle="-", label="Mean Displacements")

    def prediction_brownian(t, dim, diffusion):
        return 2 * dim * diffusion * t

    y = prediction_brownian(dt * x, dimension, diffusion_constant)
    popt, pcov = sp.optimize.curve_fit(
        lambda t, D: prediction_brownian(t, D, dimension),
        dt * x,
        msd,
        sigma=msd_err,
    )

    ax.plot(
        x,
        y,
        label="Prediction $2nDt$ with $D={}$".format(diffusion_constant),
        color="k",
        linestyle=":",
    )
    ax.plot(
        x,
        prediction_brownian(dt * x, popt[0], dimension),
        label="Fit $D={:4.3} \\pm {:4.3}$".format(popt[0], pcov[0][0]**0.5),
        linestyle="--",
        color="orange",
    )

    ax.legend()

    ax.set_title("Mean Squared Displacement {}".format(storage_name))
    fig.tight_layout()
    fig.savefig("{}/mean-squared-displacement.png".format(last_save_dir))


plot_results(
    storage_name = "brownian_1d_1",
    diffusion_constant = 1.0,
    dimension=1,
    dt=1e-3,
)

plot_results(
    storage_name = "brownian_1d_2",
    diffusion_constant = 0.5,
    dimension=1,
    dt=1e-3,
)

plot_results(
    storage_name = "brownian_1d_3",
    diffusion_constant = 0.25,
    dimension=1,
    dt=1e-3,
)

plot_results(
    storage_name = "brownian_2d_1",
    diffusion_constant = 1.0,
    dimension=2,
    dt=1e-3,
)

plot_results(
    storage_name = "brownian_2d_2",
    diffusion_constant = 0.5,
    dimension=2,
    dt=1e-3,
)

plot_results(
    storage_name = "brownian_2d_3",
    diffusion_constant = 0.25,
    dimension=2,
    dt=1e-3,
)

plot_results(
    storage_name = "brownian_3d_1",
    diffusion_constant = 1.0,
    dimension=3,
    dt=1e-3,
)

plot_results(
    storage_name = "brownian_3d_2",
    diffusion_constant = 0.5,
    dimension=3,
    dt=1e-3,
)

plot_results(
    storage_name = "brownian_3d_3",
    diffusion_constant = 0.25,
    dimension=3,
    dt=1e-3,
)
