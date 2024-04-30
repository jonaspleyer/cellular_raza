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
        figsize=(8,8),
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

    for i in range(2):
        for j in range(3):
            if data.shape[1] > j*2+i:
                if j*2+i > 0:
                    ax[min(j,1), i].plot(0.0 * data[:,j*2+i], color="grey", linestyle="--")
                if j*2+i == 3:
                    ax[min(j, 1), 0].plot(data[:,j*2+i], label=descr[j*2+i])
                    ax[min(j, 1), 0].legend()
                elif j*2+i == 0:
                    ax[0,0].plot(2 + 0*data[:,0], color="grey", linestyle="--")
                    ax[min(j, 1), i].plot(data[:,j*2+i], label=descr[j*2+i])
                else:
                    ax[min(j, 1), i].plot(data[:,j*2+i], label=descr[j*2+i])
                ax[min(j, 1), i].legend()

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

