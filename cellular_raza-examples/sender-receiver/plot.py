import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.genfromtxt("controller_logs.csv", delimiter=",")

    fig, ax = plt.subplots(2, 2)

    descr = [
        "du",
        "proportional",
        "differential",
        "integral",
    ]
    
    for i in range(2):
        for j in range(2):
            ax[i,j].plot(data[:,j*2+i])
            ax[i,j].set_title(descr[j*2+i])

    fig.tight_layout()
    fig.savefig("controller_logs.png")
    fig.savefig("controller_logs.pdf")
    plt.close(fig)

