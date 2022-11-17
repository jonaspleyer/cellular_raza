import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import itertools
import glob
import time


def export_png(i, x_slices, y_slices, x_min, x_max, y_min, y_max, size=10**2):
    radius = 1
    n_vox = 20

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


def generate_save_pairs(dataframes, step, x_min, x_max, y_min, y_max):
    for i, df in enumerate(dataframes):
        if i % step == 0 and len(df) > 1:
            yield i, df.x0, df.x1, x_min, x_max, y_min ,y_max


if __name__ == "__main__":
    p = mp.Pool(40)

    # Read the input csvs
    start = time.time()
    dataframes = p.map(pd.read_csv, sorted(glob.glob("out/*.csv")))
    print("Reading csv files takes:", time.time() - start)

    # Calculate intermediate values
    start = time.time()
    domain_size = 100

    x_min = -domain_size
    x_max = domain_size

    y_min = -domain_size
    y_max = domain_size

    step = 2
    print("Assigning values takes:", time.time()-start)

    # Export images
    p.starmap(export_png, generate_save_pairs(dataframes, step, x_min, x_max, y_min, y_max))
    print("Generating all images takes:", time.time()-start)
