import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import itertools
import glob


def export_pdf(i, x_slices, y_slices, x_min, x_max, y_min, y_max, size=5**2):
    radius = 0.5
    n_vox = 40

    fig, ax = plt.subplots(figsize=(10, 10))

    for x, y in zip(x_slices, y_slices):
        ax.plot(float(x), float(y), c="k", marker=".", markersize=4 * np.pi * radius**2)
        # circle1 = plt.Circle((x, y), radius, color='k', fill=False)
        # ax.add_patch(circle1)
    
    xdata=np.linspace(x_min, x_max, n_vox+1)
    ydata=np.linspace(y_min, y_max, n_vox+1)

    ax.set_xticks(xdata)
    ax.set_yticks(ydata)
    ax.grid(axis='both')
    
    ax.set_xlim((x_min,x_max))
    ax.set_ylim((y_min,y_max))
    fig.savefig("out/output_{:010.0f}.png".format(i))
    plt.close(fig)


if __name__ == "__main__":
    p = mp.Pool()

    dataframes = p.map(pd.read_csv, glob.glob("out/*.csv"))
    
    sizes = [len(dataframes[i].t) for i in range(len(dataframes))]
    size = max(sizes)

    domain_size = 200

    x_min = -domain_size
    x_max = domain_size

    y_min = -domain_size
    y_max = domain_size

    step = 2

    x_values = [[df.x0[i] for df in dataframes if len(df.x0) > i] for i in np.arange(0, size, step)]
    y_values = [[df.x1[i] for df in dataframes if len(df.x1) > i] for i in np.arange(0, size, step)]
    
    p.starmap(export_pdf,
        zip(
            np.arange(0, size, step),
            x_values,
            y_values,
            itertools.repeat(x_min),
            itertools.repeat(x_max),
            itertools.repeat(y_min),
            itertools.repeat(y_max)
        )
    )