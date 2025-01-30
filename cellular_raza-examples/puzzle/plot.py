import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
from glob import glob
from pathlib import Path
import os


def get_last_save_path(opath: Path | None = None) -> Path:
    if opath is None:
        opath = Path("out/puzzles")
    results = sorted(glob(str(opath) + "/*"))
    return Path(results[-1])


def get_iterations(opath: Path = get_last_save_path()) -> list[int]:
    return [int(os.path.basename(x)) for x in glob(str(opath) + "/cells/json/*")]


def load_cells(iteration: int, opath: Path = get_last_save_path()) -> list[dict] | None:
    batches = glob(str(opath) + "/cells/json/{:020}/*.json".format(iteration))
    if len(batches) == 0:
        return None
    else:
        cells = []
        for b in batches:
            f = open(b)
            data = json.load(f)["data"]
            for d in data:
                cells.append(d["element"][0]["cell"])
        return cells


def extract_vertices(cells: list[dict]):
    for cell in cells:
        yield np.array(cell["mechanics"]["puzzle"]["vertices"]["Value"])


def extract_triangulations(cells: list[dict]):
    for cell in cells:
        vertices = np.array(cell["mechanics"]["puzzle"]["vertices"]["Value"])
        triangles = np.array(cell["mechanics"]["puzzle"]["triangulation"]["triangles"])
        for triangle in triangles:
            yield vertices[triangle]


def get_domain_size(opath: Path = get_last_save_path()):
    d = glob(glob(str(opath) + "/subdomains/json/*")[0] + "/*")[0]
    f = open(d)
    subdomain = json.load(f)["element"]["subdomain"]
    domain_min = subdomain["domain_min"]
    domain_max = subdomain["domain_max"]
    return domain_min, domain_max


class Plotter:
    def __init__(self, opath: Path | str = get_last_save_path()):
        self.opath = Path(opath)
        self.domain_min, self.domain_max = get_domain_size(self.opath)
        s = (self.domain_max[1] - self.domain_min[1]) / (
            self.domain_max[0] - self.domain_min[0]
        )
        self.fig, self.ax = plt.subplots(figsize=(12, s * 12))
        self.ax.set_xlim((self.domain_min[0], self.domain_max[0]))
        self.ax.set_ylim((self.domain_min[1], self.domain_max[1]))
        self.fig.tight_layout()
        self.ax.set_axis_off()

    def plot_cells_at_iteration(self, iteration: int, transparent: bool = True):
        cells = load_cells(iteration, self.opath)
        vertices = extract_vertices(cells)
        pc1 = matplotlib.collections.PatchCollection(
            [
                matplotlib.patches.Polygon(
                    cell_vertices,
                    fill=True,
                    closed=True,
                )
                for cell_vertices in vertices
            ],
            alpha=0.7,
            facecolor="green",
            edgecolor="black",
        )
        pc1 = self.ax.add_collection(pc1)
        triangles = extract_triangulations(cells)
        pc2 = matplotlib.collections.PatchCollection(
            [
                matplotlib.patches.Polygon(
                    triangle,
                    fill=False,
                    closed=True,
                )
                for triangle in triangles
            ],
            facecolors="none",
            edgecolors="black",
            linestyle="--",
        )
        pc2 = self.ax.add_collection(pc2)
        os.makedirs("{}/snapshots/".format(self.opath), exist_ok=True)
        self.fig.savefig(
            "{}/snapshots/{:020}.png".format(self.opath, iteration),
            transparent=transparent,
        )
        pc1.remove()
        pc2.remove()

    def plot_iterations(self, iterations, progress_bar: bool = False):
        if progress_bar:
            import tqdm

            iterations = tqdm.tqdm(iterations)
        for iteration in iterations:
            self.plot_cells_at_iteration(iteration)

    def plot_all_iterations(self, progress_bar: bool = True):
        iterations = get_iterations(self.opath)
        self.plot_iterations(iterations, progress_bar)

    def generate_movie(self, play_movie: bool = True):
        bashcmd = f"ffmpeg\
            -v quiet\
            -stats\
            -y\
            -r 30\
            -f image2\
            -pattern_type glob\
            -i '{self.opath}/snapshots/*.png'\
            -c:v h264\
            -pix_fmt yuv420p\
            -strict -2 {self.opath}/movie.mp4"
        os.system(bashcmd)

        if play_movie:
            print("Playing Movie")
            bashcmd2 = f"firefox ./{self.opath}/movie.mp4"
            os.system(bashcmd2)


if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot_all_iterations()
    plotter.generate_movie()
