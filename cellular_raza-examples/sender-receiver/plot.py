import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import sys

def get_last_run_dir():
    folders = sorted(glob.glob("out/sender_receiver/*"))
    all_folders = []
    for folder in folders:
        all_folders += glob.glob(str(folder) + "/*")
    return Path(sorted(all_folders)[-1])

def create_time(n_steps):
    dt = 0.1
    t = dt / 60 * np.arange(n_steps)
    return t


def plot_pid_controller(last_run_dir = None):
    if last_run_dir == None:
        last_run_dir = get_last_run_dir()
    else:
        last_run_dir = Path(last_run_dir)
    data = np.genfromtxt(last_run_dir / "pid_controller.csv", delimiter=",")

    fig, ax = plt.subplots(
        2, 2,
        figsize=(10,8),
        sharex=True,
        sharey='row',
        gridspec_kw={"wspace":0, "hspace":0}
    )

    # WARNING: This coincides with the time-increment of the simulation
    # If we change this constant in the source code, we also have to change it here.
    t = create_time(len(data[:,0]))

    ax[0,0].plot(t, data[:,0], label="Average Concentration")
    ax[0,0].plot(t, 1 + 0*t, color="grey", linestyle="--", label="target")
    ax[0,0].legend()
    ax[0,0].set_xlabel("Time [min]")
    ax[0,0].set_ylabel("Concentration [nM]")

    ax[0,1].plot(t, data[:,1], label="Difference to set-point")
    ax[0,1].plot(t, 0*t, color="grey", linestyle="--", label="target")
    ax[0,1].set_xlabel("Time [min]")
    ax[0,1].legend()

    ax[1,0].plot(t, data[:,2], label="proportional")
    ax[1,0].plot(t, data[:,3], label="differential")
    ax[1,0].plot(t, data[:,4], label="integral")
    ax[1,0].plot(t, 0.0 * t, color="grey", linestyle="--", label="target")
    ax[1,0].legend()
    ax[1,0].set_ylabel("Controller Response [nM/min]")
    ax[1,0].set_xlabel("Time [min]")

    # WARNING: This 1 + 0*data[:,0] is a magic number!
    # It coincides with the target_concentration of the controller
    # If this number is changed in the source code, we also need to change
    # it here!
    ax[1,1].plot(t, data[:,5], label="total")
    ax[1,1].plot(t, 0*t, color="grey", linestyle="--", label="target")
    ax[1,1].set_xlabel("Time [min]")
    ax[1,1].legend()

    fig.tight_layout()
    fig.savefig(last_run_dir / "pid_controller.png")
    fig.savefig(last_run_dir / "pid_controller.pdf")
    plt.tight_layout()
    plt.close(fig)

def plot_delay_ode_controller(last_run_dir = None):
    if last_run_dir == None:
        last_run_dir = get_last_run_dir()
    else:
        last_run_dir = Path(last_run_dir)
    data = np.genfromtxt(last_run_dir / "delay_ode_mpc.csv", delimiter=",")

    # Create plotting base canvas
    fig, ax = plt.subplots(
        1, 2,
        figsize=(10,4),
        sharex=True,
        sharey="row",
        gridspec_kw={"wspace":0, "hspace":0}
    )

    # Get time data
    t = create_time(len(data[:,0]))

    # Plot all exported data
    ax[0].plot(t, data[:,0], label="Average Concentration", color="k")
    ax[0].plot(t, data[:,3], label="Predicted Concentration", color="k", linestyle="--")
    ax[0].plot(t, 1 + 0.0 * data[:,0], color="grey", linestyle="-.", label="target")
    ax[0].legend()
    ax[0].set_xlabel("Time [min]")
    ax[0].set_ylabel("Concentration [nM]")

    ax[1].plot(t, data[:,1], label="Cost", color="k")
    ax[1].plot(t, 0.0 * data[:,0], color="grey", linestyle="-.", label="target")
    ax[1].legend()
    ax[1].set_xlabel("Time [min]")

    fig.tight_layout()

    # Save to files
    fig.savefig(last_run_dir / "delay_ode_mpc.png")
    fig.savefig(last_run_dir / "delay_ode_mpc.pdf")

    # Close plots and (hopefully) free memory
    plt.close(fig)

if __name__ == "__main__":
    try:
        last_run_dir = sys.argv[1]
    except:
        last_run_dir = None
    try:
        plot_pid_controller(last_run_dir)
    except:
        pass
    try:
        plot_delay_ode_controller(last_run_dir)
    except:
        pass

