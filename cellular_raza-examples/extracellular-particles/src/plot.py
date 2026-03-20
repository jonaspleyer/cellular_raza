import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
import json
from tqdm import tqdm
import pyvista as pv
import os

# Get last output dir


def load_data(output_dir=None):
    if output_dir is None:
        output_dir = Path(list(sorted(glob("out//extracellular_particles/*")))[-1])

    # Load iterations
    iterations = sorted(
        [int(Path(i).name) for i in glob(str(output_dir / "cells/json/") + "/*")]
    )

    # Load cells
    def load_data_at_iter(dir: Path):
        def __load_data(dir):
            batches = glob(str(dir / "batch*.json"))
            data_gathered = []
            for b in batches:
                with open(b) as f:
                    d = json.load(f)
                    data_gathered.extend([di["element"][0]["cell"] for di in d["data"]])
            singles = glob(str(dir / "single*.json"))
            for s in singles:
                with open(s) as f:
                    d = json.load(f)
                    data_gathered.append(d)
            return data_gathered

        cell_dir = dir / f"cells/json/{i:020}/"
        cell_data = __load_data(cell_dir)
        subdomain_dir = dir / f"subdomains/json/{i:020}/"
        subdomain_data = __load_data(subdomain_dir)

        return cell_data, subdomain_data

    data = {}
    for i in iterations:
        cell_data, subdomain_data = load_data_at_iter(output_dir)
        data[i] = {"cells": cell_data, "subdomains": subdomain_data}

    return data


def plot_iteration(iteration, data, margin=0):
    cell_data = data[iteration]["cells"]
    subdomain_data = data[iteration]["subdomains"]

    particles = []
    cells = []
    for c in cell_data:
        # Select the positions of every particle
        pi = np.array(c["particles"][0]).reshape((-1, 6))[:, :3]
        particles.append(pi)

        pos = c["mechanics"]["pos"]
        radius = c["interaction"]["radius"]

        cells.append((pos, radius))

    particles_domain = [
        np.array(si["element"]["particles"][0]).reshape((-1, 6))
        for si in subdomain_data
    ]

    particles = np.vstack([pi for pi in particles])
    particles_domain = np.vstack([pi for pi in particles_domain])

    xymin = np.min([si["element"]["base"]["min"] for si in subdomain_data], axis=0)
    xymax = np.min([si["element"]["base"]["max"] for si in subdomain_data], axis=0)

    ratio = (xymax[0] - xymin[0]) / (xymax[1] - xymin[1])
    fig, ax = plt.subplots(figsize=(ratio * 8, 8))

    ax.set_xlim(xymin[0] - margin, xymax[0] + margin)
    ax.set_ylim(xymin[1] - margin, xymax[1] + margin)

    ax.add_patch(
        mpl.patches.Rectangle(xymin[:2], *(xymax[:2]), color="gray", fill=False)
    )

    ax.scatter(
        particles[:, 0],
        particles[:, 1],
        color="orange",
        marker=".",
        alpha=0.5,
    )
    ax.scatter(
        particles_domain[:, 0],
        particles_domain[:, 1],
        color="red",
        marker=".",
        alpha=0.5,
    )

    for pos, radius in cells:
        circ = mpl.patches.Circle(xy=pos[:2], radius=radius, fill=False, color="k")
        ax.add_patch(circ)
        ax.scatter([pos[0]], [pos[1]], marker="x", color="k")

    fig.savefig(f"out/{iteration:08}.png")
    plt.close(fig)


def create_movie():
    cmd = "ffmpeg\
        -pattern_type glob -i 'out/*.png'\
        -c:v h264\
        -pix_fmt yuv420p\
        -y\
        output.mp4"
    os.system(cmd)


if __name__ == "__main__":
    data = load_data()

    for iteration in tqdm(data.keys()):
        plot_iteration(iteration, data)

    create_movie()
