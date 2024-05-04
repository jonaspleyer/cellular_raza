import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def get_last_run_dir():
    folders = sorted(glob.glob("out/sender_receiver/*"))
    return Path(folders[-1])

def create_time(n_steps):
    dt = 0.1
    t = dt / 60 * np.arange(n_steps)
    return t


def plot_pid_controller():
    last_run_dir = get_last_run_dir()
    data = np.genfromtxt(last_run_dir / "pid_controller.csv", delimiter=",")

    fig, ax = plt.subplots(
        2, 2,
        figsize=(10,8),
        sharex=True,
        sharey='row',
        gridspec_kw={"wspace":0, "hspace":0}
    )

    descr = [
        "average concentration",
        "du",
        "proportional",
        "differential",
        "integral",
        "total",
    ]

    # WARNING: This coincides with the time-increment of the simulation
    # If we change this constant in the source code, we also have to change it here.
    t = create_time(len(data[:,0]))
    for i in range(2):
        for j in range(3):
            if data.shape[1] > j*2+i:
                k = min(j, 1)
                if j*2+i > 0:
                    ax[k, i].plot(t, 0.0 * data[:,j*2+i], color="grey", linestyle="--")
                if j*2+i == 3:
                    ax[k, 0].plot(t, data[:,j*2+i], label=descr[j*2+i])
                    ax[k, 0].legend()
                    ax[k, 0].set_ylabel("Controller Response [nM/min]")
                elif j*2+i == 0:
                    # WARNING: This 1 + 0*data[:,0] is a magic number!
                    # It coincides with the target_concentration of the controller
                    # If this number is changed in the source code, we also need to change
                    # it here!
                    ax[0,0].plot(t, 1 + 0*data[:,0], color="grey", linestyle="--")
                    ax[k, i].plot(t, data[:,j*2+i], label=descr[j*2+i])
                    ax[k, i].set_ylabel("Concentration [nM]")
                else:
                    ax[k, i].plot(t, data[:,j*2+i], label=descr[j*2+i])
                ax[k, i].legend()
                ax[k, i].set_xlabel("Time [min]")

    fig.tight_layout()
    fig.savefig(last_run_dir / "pid_controller.png")
    fig.savefig(last_run_dir / "pid_controller.pdf")
    plt.tight_layout()
    plt.close(fig)

def plot_delay_ode_controller():
    last_run_dir = get_last_run_dir()
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
    ax[0].plot(t, 1 + 0.0 * data[:,0], color="grey", linestyle="--")
    ax[0].legend()
    ax[0].set_xlabel("Time [min]")
    ax[0].set_ylabel("Concentration [nM]")

    ax[1].plot(t, data[:,1], label="Cost", color="k")
    ax[1].plot(t, 0.0 * data[:,0], color="grey", linestyle="--")
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
        plot_pid_controller()
        print("[x] pid_controller")
    except:
        print("[ ] pid_controller")
    try:
        plot_delay_ode_controller()
        print("[x] delay_ode_controller")
    except:
        print("[ ] delay_ode_controlle")

