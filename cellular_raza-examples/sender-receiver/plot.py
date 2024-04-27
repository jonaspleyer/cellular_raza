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
        3, 2,
        figsize=(12,8),
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
                    ax[j, i].plot(0.0 * data[:,j*2+i], color="grey", linestyle="--")
                ax[j,i].plot(data[:,j*2+i])
                ax[j,i].set_title(descr[j*2+i])

    fig.tight_layout()
    fig.savefig(last_run_dir / "pid_controller.png")
    fig.savefig(last_run_dir / "pid_controller.pdf")
    plt.tight_layout()
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    try:
        plot_pid_controller()
        print("[x] pid_controller")
    except:
        print("[ ] pid_controller")
