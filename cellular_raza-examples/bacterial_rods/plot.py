import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import pandas as pd
from pathlib import Path
from typing import Optional
import os
import tqdm
import multiprocessing as mp

def plot_iteration(output_path: Path, iteration: int):
    cells = load_cells_from_iteration(output_path, iteration)

    # Create Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    patches = []
    for _, cell in cells.iterrows():
        pos = cell["cell.pos"]
        radius = cell["cell.radius"]
        patches += [
            matplotlib.patches.Circle(pos[:2,i], radius, color="k")
            for i in range(pos.shape[1])
        ]
        patches += [
            matplotlib.patches.Circle(pos[:2,i], 0.9 * radius, color="green")
            for i in range(pos.shape[1])
        ]
    patches = matplotlib.collections.PatchCollection(patches, match_original=True)
    ax.add_collection(patches)

    # Some more settings
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()

    # Save Figure
    (output_path / "snapshots").mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / "snapshots/snapshot_{:020}.png".format(iteration))
    plt.close(fig)
    del fig

def __plot_all_iterations_helper(args_kwargs):
    args, kwargs = args_kwargs
    plot_iteration(*args, **kwargs)

def plot_all_iterations(output_path: Optional[Path] = None, threads: int = 1, **kwargs):
    print("Plotting all iterations")
    output_path, iterations = get_all_iterations(output_path)
    
    args_kwargs = [((output_path, iteration), kwargs) for iteration in iterations]

    pool = mp.Pool(threads)
    with mp.Pool(threads) as pool:
        list(tqdm.tqdm(pool.imap(
            __plot_all_iterations_helper,
            args_kwargs
        ), total=len(args_kwargs)))

def generate_movie(output_path: Optional[Path] = None):
    if output_path == None:
        output_path = get_last_output_path()
    print("Generating Snapshot Movie")
    bashcmd = f"ffmpeg\
        -v quiet\
        -stats\
        -y\
        -r 30\
        -f image2\
        -pattern_type glob\
        -i '{output_path}/snapshots/*.png'\
        -c:v h264\
        -pix_fmt yuv420p\
        -strict -2 {output_path}/snapshot_movie.mp4"
    os.system(bashcmd)
    bashcmd2 = f"vlc {output_path}/snapshot_movie.mp4"
    os.system(bashcmd2)

if __name__ == "__main__":
    plot_all_iterations(threads=30)

    generate_movie()

