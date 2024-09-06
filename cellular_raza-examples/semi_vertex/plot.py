import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, RegularPolyCollection
import numpy as np
import json
import pandas as pd
from pathlib import Path
from glob import glob
import tqdm
from concurrent.futures import ThreadPoolExecutor
import os

def get_last_output_path() -> Path:
    return Path(sorted(list(glob("out/semi_vertex/*")))[-1])

class Loader:
    def __init__(self, opath: Path = get_last_output_path()):
        self.opath = opath
        self.cells = None
        self.__iterations = None

    def __get_all_iterations(self) -> np.ndarray:
        folders = glob(str(self.opath / "cells/json/*"))
        return np.sort(np.array([int(Path(f).name) for f in folders]))


    def load_cells(self, iteration: int) -> pd.DataFrame:
        # Load all json files
        results = []
        for file in glob(str(self.opath / "cells/json/{:020}/*.json".format(iteration))):
            f = open(file)
            batch = json.load(f)
            results.extend([b["element"][0] for b in batch["data"]])
        df = pd.json_normalize(results)
        df["cell.mechanics.points"] = df["cell.mechanics.points"].apply(
            lambda x: np.array(x, dtype=float).reshape((2, -1)).T
        )
        self.cells = df
        return df

    @property
    def iterations(self) -> np.ndarray:
        if self.__iterations is None:
            self.__iterations = self.__get_all_iterations()
        return self.__iterations

def run_plotter_iterations(instance: int, opath: Path, iterations, **kwargs):
    plotter = Plotter(opath)
    iterations = tqdm.tqdm(iterations, total=len(iterations), position=instance)
    plotter.plot_cells_at_iterations(iterations, **kwargs, progress_bar=False)

class Plotter:
    loader: Loader

    def __init__(self, opath: Path = get_last_output_path()):
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.loader = Loader(opath)

    def duplicate(self):
        return Plotter(self.loader.opath)

    def plot_cells(self, df_cells):
        polys = [Polygon(pos) for pos in df_cells["cell.mechanics.points"]]
        pc = PatchCollection(
            polys,
            facecolors="green",
            edgecolors='white'
        )
        self.pc = self.ax.add_collection(pc)

    def plot_cells_at_iter(
            self,
            iteration: int,
            save_path: Path | None = None,
            overwrite: bool = False,
            transparent: bool = False,
        ):
        if save_path is None:
            save_path = self.loader.opath / "images"
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / "snapshot-{:020}.png".format(iteration)
        if save_path.exists() and overwrite is False:
            return save_path
        cells = self.loader.load_cells(iteration)

        self.ax.set_xlim(0, 800)
        self.ax.set_ylim(0, 800)
        self.ax.set_axis_off()
        self.plot_cells(cells)
        self.fig.tight_layout()
        self.fig.savefig(save_path, transparent=transparent)
        self.pc.remove()
        return save_path

    def plot_cells_at_iterations(
            self,
            iterations: list[int] | np.ndarray,
            save_path: Path | None = None,
            overwrite: bool = False,
            transparent: bool = True,
            progress_bar: bool = True,
        ):
        iterations = tqdm.tqdm(iterations) if progress_bar else iterations
        for iteration in iterations:
            self.plot_cells_at_iter(iteration, save_path, overwrite, transparent)

    def plot_cells_at_all_iterations(
        self,
        save_path: Path | None = None,
        overwrite: bool = False,
        transparent: bool = False,
        n_threads: int | None = None,
    ):
        plt.close('all')
        matplotlib.use('svg')
        p = ThreadPoolExecutor(n_threads)
        if n_threads is None:
            n_threads = os.cpu_count()
        chunksize = int(len(self.loader.iterations) / n_threads)
        left_over = len(self.loader.iterations) % n_threads
        results = []
        for i in range(0, n_threads):
            lower = i * chunksize
            upper = lower + chunksize
            if left_over > 0:
                upper += 1
                left_over -= 1
            iterations = self.loader.iterations[lower:upper]
            # run_plotter_iterations(new_plotter, iterations, save_path=save_path, overwrite=overwrite, transparent=transparent)
            res = p.submit(
                run_plotter_iterations,
                i, self.loader.opath, iterations,
                save_path=save_path,
                overwrite=overwrite,
                transparent=transparent,
            )
            results.append(res)

        for res in results:
            res.result()

if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot_cells_at_all_iterations(transparent=True)

