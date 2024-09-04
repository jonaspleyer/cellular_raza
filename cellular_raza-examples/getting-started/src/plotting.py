from pathlib import Path
import json
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import tqdm# For nice progress bar in the terminal (optional)

def get_last_output_path() -> Path:
    return Path(sorted(glob(str("out/*")))[-1])

def get_all_iterations(opath: Path = get_last_output_path()) -> np.ndarray:
    iteration_dirs = glob(str(opath) + "/cells/json/*")
    return np.array(sorted([int(os.path.basename(dir)) for dir in iteration_dirs]))

def get_cells_at_iteration(iteration: int, opath: Path = get_last_output_path()):
    cells = []
    for batch in glob(str(opath) + "/cells/json/{:020}/*".format(iteration)):
        f = open(batch)
        cells.extend([c["element"] for c in json.load(f)["data"]])
    return cells

def _get_domain_size(iteration: int, opath: Path = get_last_output_path()):
    singles = glob(str(opath) + "/subdomains/json/{:020}/*".format(iteration))
    f = open(singles[0])
    sdm = json.load(f)
    domain_min = sdm["element"]["domain_min"]
    domain_max = sdm["element"]["domain_max"]
    return domain_min, domain_max

class Plotter:
    def __init__(self, opath: Path = get_last_output_path(), fig = None):
        iterations = get_all_iterations(opath)
        self.domain_min, self.domain_max = _get_domain_size(iterations[0], opath)
        s = (self.domain_max[1] - self.domain_min[1]) / (self.domain_max[0] - self.domain_min[0])
        self.fig, self.ax = plt.subplots(figsize=(6, s*6))
        self.ax.set_xlim((self.domain_min[0], self.domain_max[0]))
        self.ax.set_ylim((self.domain_min[1], self.domain_max[1]))
        self.fig.tight_layout()
        self.opath = opath

    def plot_iteration(
            self,
            iteration: int,
        ):
    
        self.ax.set_xlim((self.domain_min[0], self.domain_max[0]))
        self.ax.set_ylim((self.domain_min[1], self.domain_max[1]))
    
        cells = get_cells_at_iteration(iteration, self.opath)
        positions = np.array([cell[0]["cell"]["mechanics"]["pos"] for cell in cells])
        radii = np.array([cell[0]["cell"]["interaction"]["radius"] for cell in cells])
        patches = PatchCollection(
            [Circle(pos, radius) for (pos, radius) in zip(positions, radii)],
            facecolor="green",
            edgecolor="k",
        )
        self.ax.add_collection(patches)

    def save_iteration(
        self,
        iteration: int,
    ):
        self.plot_iteration(iteration)
        os.makedirs(str(self.opath) + "/snapshots/", exist_ok=True)
        self.fig.savefig(str(self.opath) + "/snapshots/{:020}.png".format(iteration))
        self.ax.cla()

    def save_all_iterations(self):
        iterations = get_all_iterations()
        for iteration in tqdm.tqdm(iterations, total=len(iterations)):
            self.save_iteration(iteration)

def generate_movie(opath: Path = get_last_output_path(), and_open: bool = False):
    cmd = "ffmpeg\
        -y\
        -pattern_type glob\
        -i \'{}/snapshots/*.png\'\
        -c:v libx264\
        -r 15\
        -pix_fmt yuv420p {}/movie.mp4\
        ".format(opath, opath)
    os.system(cmd)
    if and_open:
        cmd2 = "firefox {}/movie.mp4".format(opath)
        os.system(cmd2)

if __name__ == "__main__":
    plotter = Plotter()
    plotter.save_all_iterations()
    generate_movie(and_open=True)
