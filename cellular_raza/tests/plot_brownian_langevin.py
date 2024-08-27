import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
import scipy as sp

def get_last_save_dir(storage_name: str) -> str:
    return list(sorted(glob("out/{}/*".format(storage_name))))[-1]

def get_trajectories(storage_name: str) -> np.ndarray:
    last_save_dir = get_last_save_dir(storage_name)

    # Obtain all values for cells
    iterations_cells = []
    for iteration_dir in sorted(glob(last_save_dir + "/cells/json/*")):
        cells = []
        for batch in list(sorted(glob(iteration_dir + "/*"))):
            with open(batch) as f:
                cells.extend(json.load(f)["data"])
        iterations_cells.append(cells)

    # Calculate the trajectories
    trajectories = np.array(
        np.array([[
            values_at_iter[j]["element"][0]["cell"]["pos"]
            for values_at_iter in iterations_cells
        ] for j in range(len(iterations_cells[0]))]
    ))
    return trajectories

def get_domain_boundaries(storage_name: str) -> tuple[np.ndarray, np.ndarray]:
    last_save_dir = get_last_save_dir(storage_name)
    # Obtain all values for subdomains
    iteration_dir = glob(last_save_dir + "/subdomains/json/*")[0]
    single = glob(iteration_dir + "/*")[0]
    with open(single) as f:
        subdomain = json.load(f)["element"]
        dmin = np.array([subdomain["domain_min"][0]])
        dmax = np.array([subdomain["domain_max"][0]])
        # return np.ndarray([subdomain["domain_min"]]), np.ndarray([subdomain["domain_max"]])
        return dmin, dmax

def plot_2d_only(trajectories: np.ndarray, domain_middle: np.ndarray, last_save_dir: str):
    # Plot the obtained results for each iteration
    dh = np.max(np.abs(trajectories - domain_middle), axis=(0,1))
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

    # Plot a heatmap of the total explored space
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

def plot_msd(trajectories: np.ndarray, domain_middle: np.ndarray):
    # Plot the mean squared displacement per iteration
    msd = np.mean(np.sum((trajectories - domain_middle)**2, axis=2), axis=0)
    msd_err = np.std(np.sum((trajectories - domain_middle)**2, axis=2), axis=0)

    x = np.arange(len(msd))
    fig, ax = plt.subplots()
    ax.plot(x, msd, color="gray", linestyle="-", label="Mean Displacements")
    return fig, ax, x, msd, msd_err

def plot_brownian(
        storage_name: str,
        diffusion_constant: float,
        dimension: int,
        dt: float,
    ):

    # Get trajectories
    last_save_dir = get_last_save_dir(storage_name)
    trajectories = get_trajectories(storage_name)

    # Get Domain size
    domain_min, domain_max = get_domain_boundaries(storage_name)
    domain_middle = 0.5 * (domain_min + domain_max)

    fig, ax, x, msd, msd_err = plot_msd(trajectories, domain_middle)

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

    if trajectories.shape[2] == 2:
        plot_2d_only(trajectories, domain_middle, last_save_dir)

BROWNIAN_VALUES = [
    {"storage_name":"brownian_1d_1", "diffusion_constant": 1.0, "dimension":1, "dt":1e-3},
    {"storage_name":"brownian_1d_1", "diffusion_constant": 1.0, "dimension":1, "dt":1e-3},
    {"storage_name":"brownian_1d_2", "diffusion_constant": 0.5, "dimension":1, "dt":1e-3},
    {"storage_name":"brownian_1d_3", "diffusion_constant":0.25, "dimension":1, "dt":1e-3},
    {"storage_name":"brownian_2d_1", "diffusion_constant": 1.0, "dimension":2, "dt":1e-3},
    {"storage_name":"brownian_2d_2", "diffusion_constant": 0.5, "dimension":2, "dt":1e-3},
    {"storage_name":"brownian_2d_3", "diffusion_constant":0.25, "dimension":2, "dt":1e-3},
    {"storage_name":"brownian_3d_1", "diffusion_constant": 1.0, "dimension":3, "dt":1e-3},
    {"storage_name":"brownian_3d_2", "diffusion_constant": 0.5, "dimension":3, "dt":1e-3},
]

# for kwargs in BROWNIAN_VALUES:
#     plot_brownian(**kwargs)

def plot_langevin(
        storage_name: str,
        damping: float,
        diffusion: float,
        dim: int,
        dt: float
    ):
    kb_temperature = 1.0
    mass = 1.0

    # Get trajectories
    last_save_dir = get_last_save_dir(storage_name)
    trajectories = get_trajectories(storage_name)

    # Get Domain size
    domain_min, domain_max = get_domain_boundaries(storage_name)
    domain_middle = 0.5 * (domain_min + domain_max)

    # Plot the mean squared displacement per iteration
    fig, ax, x, msd, msd_err = plot_msd(trajectories, domain_middle)

    def prediction_langevin(t, damping, kb_temperature, mass, dim):
        return - dim * kb_temperature / mass / damping**2\
            * np.exp(1.0 - (- damping * t))\
            * np.exp(3.0 - (- damping * t))\
            + 2.0 * dim * kb_temperature / mass * t / damping

    y = prediction_langevin(dt * x, damping, kb_temperature, mass, dim)
    popt, pcov = sp.optimize.curve_fit(
        lambda x, damping, kb_temp, mass: prediction_langevin(
            dt * x, damping, kb_temp, mass, dim
        ),
        dt * x,
        msd,
        sigma=msd_err,
    )

    ax.plot(
        x,
        y,
        label="Prediciton $a$ with ..",
        color="k",
        linestyle=":",
    )
    ax.plot(
        x,
        prediction_langevin(dt * x, *popt, dim),
        label="Fit $D$",
        linestyle="--",
        color="orange",
    )

    ax.legend()

    ax.set_title("Mean Squared Displacement")
    fig.tight_layout()
    fig.savefig("{}/mean-squared-displacement.png".format(last_save_dir))

    if trajectories.shape[2] == 2:
        plot_2d_only(trajectories, domain_middle, last_save_dir)


LANGEVIN_VALUES = [
        {"storage_name":"langevin_2d_1", "damping": 0.02, "diffusion": 0.1, "dim":1, "dt":1e-3},
]

for kwargs in LANGEVIN_VALUES:
    plot_langevin(**kwargs)
