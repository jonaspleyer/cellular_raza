import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from glob import glob
import scipy as sp
import string

mpl.use("Agg")

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
        np.array(
            [
                [
                    values_at_iter[j]["element"][0]["cell"]["pos"]
                    for values_at_iter in iterations_cells
                ]
                for j in range(len(iterations_cells[0]))
            ]
        )
    )
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


def plot_2d_only(
    trajectories: np.ndarray, domain_middle: np.ndarray, last_save_dir: str
):
    # Plot the obtained results for each iteration
    dh = np.max(np.abs(trajectories - domain_middle), axis=(0, 1))
    s = dh[0] / dh[1]
    lim_lower = domain_middle - 1.1 * dh
    lim_upper = domain_middle + 1.1 * dh
    xlim = [lim_lower[0], lim_upper[0]]
    ylim = [lim_lower[1], lim_upper[1]]

    fig, ax = plt.subplots(figsize=(8, s * 8))
    ax.set_title("Trajectories")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], color="k", linestyle="-")
    fig.tight_layout()
    fig.savefig("{}/trajectories.png".format(last_save_dir))
    plt.close(fig)

    # Plot a heatmap of the total explored space
    heatmap, _, _ = np.histogram2d(
        trajectories[:, :, 0].reshape((-1,)),
        trajectories[:, :, 1].reshape((-1,)),
        range=[xlim, ylim],
        bins=50,
    )
    extent = [*lim_lower, *lim_upper]
    fig, ax = plt.subplots(figsize=(8, 8 * s))
    ax.imshow(heatmap.T, extent=extent, origin="lower")
    ax.set_title("Heatmap of explored space")
    fig.tight_layout()
    fig.savefig("{}/heatmap.png".format(last_save_dir))
    plt.close(fig)


def plot_msd(ax, trajectories: np.ndarray, domain_middle: np.ndarray):
    # Plot the mean squared displacement per iteration
    msd = np.mean(np.sum((trajectories - domain_middle) ** 2, axis=2), axis=0)
    msd_err = (
        np.std(np.sum((trajectories - domain_middle) ** 2, axis=2), axis=0)
        / trajectories.shape[0] ** 0.5
    )

    x = np.arange(msd.shape[0])
    ax.fill_between(
        x, msd - msd_err, msd + msd_err, color="gray", alpha=0.5, label="Data"
    )
    return x, msd, msd_err


def plot_brownian(
    ax,
    storage_name: str,
    diffusion_constant: float,
    dimension: int,
    dt: float,
    color,
):
    print(storage_name)

    # Get trajectories
    last_save_dir = get_last_save_dir(storage_name)
    trajectories = get_trajectories(storage_name)

    # Get Domain size
    domain_min, domain_max = get_domain_boundaries(storage_name)
    domain_middle = 0.5 * (domain_min + domain_max)

    x, msd, msd_err = plot_msd(ax, trajectories, domain_middle)

    def prediction_brownian(t, dim, diffusion):
        return 2 * dim * diffusion * t

    y = prediction_brownian(dt * x, dimension, diffusion_constant)
    popt, pcov = sp.optimize.curve_fit(
        lambda t, D: prediction_brownian(t, D, dimension),
        dt * x[1:],
        msd[1:],
        p0=(diffusion_constant,),
        sigma=msd_err[1:] * trajectories.shape[0] ** 0.5,
        absolute_sigma=True,
    )

    ax.plot(
        x,
        y,
        label=f"D={diffusion_constant:<4.3}",
        color="k",
        linestyle=":",
        linewidth=4,
    )
    ax.plot(
        x,
        prediction_brownian(dt * x, popt[0], dimension),
        label="D={:4.3}±{:4.3}".format(popt[0], pcov[0][0] ** 0.5),
        linestyle="--",
        linewidth=4,
        color=color,
    )

    ax.set_title(f"Brownian {dimension}D")

    # if trajectories.shape[2] == 2:
    #     plot_2d_only(trajectories, domain_middle, last_save_dir)


def plot_langevin(
    ax,
    storage_name: str,
    damping: float,
    diffusion: float,
    dim: int,
    dt: float,
    color,
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
    x, msd, msd_err = plot_msd(ax, trajectories, domain_middle)

    def prediction_langevin(t, damping, kb_temperature_div_mass, dim):
        return (
            -dim
            * kb_temperature_div_mass
            / damping**2
            * (1.0 - np.exp(-damping * t))
            * (3.0 - np.exp(-damping * t))
            + 2.0 * dim * kb_temperature_div_mass * t / damping
        )

    popt, pcov = sp.optimize.curve_fit(
        lambda t, damping, kb_temp_div_mass: prediction_langevin(
            t, damping, kb_temp_div_mass, dim
        ),
        dt * x[2:],
        msd[2:],
        p0=(damping, kb_temperature_div_mass),
        sigma=msd_err[2:] * trajectories.shape[0] ** 0.5,
        absolute_sigma=True,
    )

    y = prediction_langevin(dt * x, damping, kb_temperature_div_mass, dim)
    ax.plot(
        x,
        y,
        label=f"λ={damping:4.3} D={diffusion:4.3}",
        color="k",
        linestyle=":",
        linewidth=4,
    )
    ax.plot(
        x,
        prediction_langevin(dt * x, *popt, dim),
        label="λ={:4.3}±{:4.3}\nD={:4.3}±{:4.3}".format(
            popt[0],
            pcov[0][0] ** 0.5,
            popt[1] / popt[0],
            ((pcov[1][1] / popt[0] ** 2) + (pcov[0][0] * popt[1] ** 2 / popt[0] ** 4))
            ** 0.5,
        ),
        linestyle="--",
        linewidth=4,
        color=color,
    )
    ax.set_title(f"Langevin {dim}D")


if __name__ == "__main__":
    set_mpl_rc_params()

    fig, axs = plt.subplots(2, 3, figsize=(24, 16), sharey="row", sharex=True)
    for ax, label in zip(axs.flatten(), string.ascii_uppercase):
        configure_ax(ax, minor=False)
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

    for i, name, dim, d in [
        (0, "brownian_1d_1", 1, 1.0),
        (1, "brownian_2d_2", 2, 0.5),
        (2, "brownian_3d_3", 3, 0.25),
    ]:
        plot_brownian(
            axs[0, i],
            storage_name=name,
            diffusion_constant=d,
            dimension=dim,
            dt=1e-3,
            color=COLOR5,
        )
    for i, name, dim, d, damp in [
        (0, "langevin_1d_1", 1, 80.0, 10.0),
        (1, "langevin_2d_2", 2, 40.0, 10.0),
        (2, "langevin_3d_3", 3, 20.0, 10.0),
    ]:
        plot_langevin(
            axs[1, i],
            storage_name=name,
            diffusion=d,
            dim=dim,
            damping=damp,
            dt=1e-3,
            color=COLOR5,
        )

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=["Data", "Analytical Prediction", "Fit"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncols=3,
        frameon=False,
    )
    for i, ax in enumerate(axs.flatten()):
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=[handles[1], handles[2]],
            labels=[labels[1], labels[2]],
            frameon=False,
            loc="lower right" if i < 3 else "center left",
        )

    axs[0, 0].set_ylabel("Mean Squared Displacement")
    axs[1, 0].set_ylabel("Mean Squared Displacement")
    for i in range(3):
        axs[1, i].set_xlabel("Time")

    fig.tight_layout(rect=(0, 0, 1, 1 - 0.03))
    fig.subplots_adjust(wspace=0)
    fig.savefig("mean-squared-displacements-brownian-langevin.pdf")
