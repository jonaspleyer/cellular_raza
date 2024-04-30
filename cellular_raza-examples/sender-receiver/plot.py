import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def get_last_run_dir():
    folders = sorted(glob.glob("out/sender_receiver/*"))
    return Path(folders[-1])

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
    dt = 0.1
    t = dt / 60 * np.arange(len(data[:,0]))
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
    plt.show()
    plt.close(fig)

def plot_exponential_controller():
    pass

def plot_delay_ode_controller():
    last_run_dir = get_last_run_dir()
    data = np.genfromtxt(last_run_dir / "delay_ode_mpc.csv", delimiter=",")

    descr = [
        "average concentration",
        "cost",
        "predicted production term",
        "predicted concentration",
    ]

    # Create plotting base canvas
    fig, ax = plt.subplots(2, 2)

    # Plot all exported data
    for i in range(2):
        for j in range(2):
            ax[j,i].plot(data[:,j*2+i], label=descr[j*2+i])
            ax[j,i].legend()

    fig.tight_layout()

    # Save to files
    fig.savefig(last_run_dir / "delay_ode_mpc.png")
    fig.savefig(last_run_dir / "delay_ode_mpc.pdf")

    # Close plots and (hopefully) free memory
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    try:
        plot_pid_controller()
        print("[x] pid_controller")
    except:
        print("[ ] pid_controller")
    try:
        plot_exponential_controller()
        print("[x] exponential_controller")
    except:
        print("[ ] exponential_controller")
    try:
        plot_delay_ode_controller()
        print("[x] delay_ode_controller")
    except:
        print("[ ] delay_ode_controlle")

