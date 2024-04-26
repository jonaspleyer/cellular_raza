import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def get_last_run_dir():
    folders = sorted(glob.glob("out/sender_receiver/*"))
    return Path(folders[-1])

if __name__ == "__main__":
    last_run_dir = get_last_run_dir()
    data = np.genfromtxt(last_run_dir / "controller_logs.csv", delimiter=",")

    fig, ax = plt.subplots(2, 2)

    # descr1 = [
    #     "du",
    #     "proportional",
    #     "differential",
    #     "integral",
    # ]

    descr2 = [
        "current concentration",
        "cost",
        "predicted production term",
        "current production term",
    ]

    for i in range(2):
        for j in range(2):
            if data.shape[1] > j*2+i:
                ax[i,j].plot(data[:,j*2+i])
                ax[i,j].set_title(descr2[j*2+i])

    fig.tight_layout()
    fig.savefig(last_run_dir / "controller_logs.png")
    fig.savefig(last_run_dir / "controller_logs.pdf")
    plt.show()
    plt.close(fig)

