import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import itertools
import glob


def export_pdf(i, x_slices, y_slices, x_min, x_max, y_min, y_max, size=33**2):
    fig, ax = plt.subplots(figsize=(6, 4))
    for x, y in zip(x_slices, y_slices):
        ax.scatter(x, y, s=size, c="k")
    
    ax.set_xlim((x_min,x_max))
    ax.set_ylim((y_min,y_max))
    fig.savefig("out/output_{:010.0f}.png".format(i))
    plt.close(fig)


if __name__ == "__main__":
    dataframes = [pd.read_csv(filename) for filename in glob.glob("out/*.csv")]
    size = len(dataframes[0].x0)

    x_min = -6# min([min(df.x0) for df in dataframes])
    x_max = 6# max([max(df.x0) for df in dataframes])

    y_min = -6# min([min(df.x1) for df in dataframes])
    y_max = 6# max([max(df.x1) for df in dataframes])

    p = mp.Pool()

    step = 2

    p.starmap(export_pdf,
        zip(
            np.arange(0, size, step),
            [[df.x0[i] for df in dataframes] for i in np.arange(0, size, step)],
            [[df.x1[i] for df in dataframes] for i in np.arange(0, size, step)],
            itertools.repeat(x_min),
            itertools.repeat(x_max),
            itertools.repeat(y_min),
            itertools.repeat(y_max)
        ))