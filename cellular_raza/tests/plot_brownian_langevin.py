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
        ax.plot(traj[:,0], traj[:,1], color="k", linestyle="-")
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
    plt.close(fig)

def plot_msd(trajectories: np.ndarray, domain_middle: np.ndarray):
    # Plot the mean squared displacement per iteration
    msd = np.mean(np.sum((trajectories - domain_middle)**2, axis=2), axis=0)
    msd_err = np.std(np.sum((trajectories - domain_middle)**2, axis=2), axis=0)\
        / trajectories.shape[0]**0.5

    x = np.arange(msd.shape[0])
    fig, ax = plt.subplots()
    ax.errorbar(x, msd, msd_err, color="gray", linestyle="-", label="Mean Displacements")
    return fig, ax, x, msd, msd_err

def plot_brownian(
        storage_name: str,
        diffusion_constant: float,
        dimension: int,
        dt: float,
    ):
    print(storage_name)

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
    plt.close(fig)

    if trajectories.shape[2] == 2:
        plot_2d_only(trajectories, domain_middle, last_save_dir)

# Note that the values here do need to match the values of the
# tests as defined by the cellular_raza test suite.
# See cellular_raza/tests/brownian_diffusion_constant_approx.rs
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
    {"storage_name":"brownian_3d_3", "diffusion_constant":0.25, "dimension":3, "dt":1e-3},
]

for kwargs in BROWNIAN_VALUES:
    plot_brownian(**kwargs)

def plot_langevin(
        storage_name: str,
        damping: float,
        diffusion: float,
        dim: int,
        dt: float
    ):
    print(storage_name)
    kb_temperature_div_mass = diffusion * damping

    # Get trajectories
    last_save_dir = get_last_save_dir(storage_name)
    trajectories = get_trajectories(storage_name)

    # Get Domain size
    domain_min, domain_max = get_domain_boundaries(storage_name)
    domain_middle = 0.5 * (domain_min + domain_max)

    # Plot the mean squared displacement per iteration
    fig, ax, x, msd, msd_err = plot_msd(trajectories, domain_middle)

    def prediction_langevin(t, damping, kb_temperature_div_mass, dim):
        return - dim * kb_temperature_div_mass / damping**2\
            * (1.0 - np.exp(- damping * t))\
            * (3.0 - np.exp(- damping * t))\
            + 2.0 * dim * kb_temperature_div_mass * t / damping

    popt, pcov = sp.optimize.curve_fit(
        lambda t, damping, kb_temp_div_mass: prediction_langevin(
            t, damping, kb_temp_div_mass, dim
        ),
        dt * x[1:],
        msd[1:],
        # sigma=msd_err[1:],
        p0=(damping, kb_temperature_div_mass),
    )

    y = prediction_langevin(dt * x, damping, kb_temperature_div_mass, dim)
    ax.plot(
        x,
        y,
        label="Prediciton $\\left<r^2(t)\\right>$",
        color="k",
        linestyle=":",
    )
    ax.plot(
        x,
        prediction_langevin(dt * x, *popt, dim),
        label="Fit $\\lambda={:4.3} \\pm {:4.3}, D={:4.3}\\pm {:4.3}$".format(
            popt[0],
            pcov[0][0]**0.5,
            popt[1] / popt[0],
            ((pcov[1][1]/popt[0]**2) + (pcov[0][0]*popt[1]**2/popt[0]**4))**0.5
        ),
        linestyle="--",
        color="orange",
    )

    ax.legend()

    ax.set_title("Mean Squared Displacement")
    fig.tight_layout()
    fig.savefig("{}/mean-squared-displacement.png".format(last_save_dir))

    if trajectories.shape[2] == 2:
        plot_2d_only(trajectories, domain_middle, last_save_dir)

# Note that the values here do need to match the values of the
# tests as defined by the cellular_raza test suite.
# See cellular_raza/tests/brownian_diffusion_constant_approx.rs
LANGEVIN_VALUES = [
    {"storage_name": "langevin_3d_1", "diffusion": 80.0, "dim": 3, "damping": 10.0, "dt": 1e-3},
    {"storage_name": "langevin_3d_2", "diffusion": 40.0, "dim": 3, "damping": 10.0, "dt": 1e-3},
    {"storage_name": "langevin_3d_3", "diffusion": 20.0, "dim": 3, "damping": 10.0, "dt": 1e-3},
    {"storage_name": "langevin_3d_4", "diffusion": 40.0, "dim": 3, "damping":  1.0, "dt": 1e-3},
    {"storage_name": "langevin_3d_5", "diffusion": 40.0, "dim": 3, "damping":  0.1, "dt": 1e-3},
    {"storage_name": "langevin_2d_1", "diffusion": 80.0, "dim": 2, "damping": 10.0, "dt": 1e-3},
    {"storage_name": "langevin_2d_2", "diffusion": 40.0, "dim": 2, "damping": 10.0, "dt": 1e-3},
    {"storage_name": "langevin_2d_3", "diffusion": 20.0, "dim": 2, "damping": 10.0, "dt": 1e-3},
    {"storage_name": "langevin_2d_4", "diffusion": 20.0, "dim": 2, "damping":  1.0, "dt": 1e-3},
    {"storage_name": "langevin_2d_5", "diffusion": 20.0, "dim": 2, "damping":  0.1, "dt": 1e-3},
    {"storage_name": "langevin_1d_1", "diffusion": 80.0, "dim": 1, "damping": 10.0, "dt": 1e-3},
    {"storage_name": "langevin_1d_2", "diffusion": 40.0, "dim": 1, "damping": 10.0, "dt": 1e-3},
    {"storage_name": "langevin_1d_3", "diffusion": 20.0, "dim": 1, "damping": 10.0, "dt": 1e-3},
    {"storage_name": "langevin_1d_4", "diffusion": 20.0, "dim": 1, "damping":  1.0, "dt": 1e-3},
    {"storage_name": "langevin_1d_5", "diffusion": 20.0, "dim": 1, "damping":  0.1, "dt": 1e-3},
]

for kwargs in LANGEVIN_VALUES:
    plot_langevin(**kwargs)
