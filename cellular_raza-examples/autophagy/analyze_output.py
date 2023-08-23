import json
from pathlib import Path
import os
from glob import glob
import argparse

import pandas as pd
import pyvista as pv
import numpy as np
import multiprocessing as mp

def get_all_items_at_iteration(iteration_folder):
    # Now we read all the elements from all files as a json
    all_elements = []
    for batch_file_name in glob(iteration_folder + "/batch_*.json"):
        batch_file = open(batch_file_name)
        parsed_json = json.load(batch_file)
        all_elements += parsed_json["data"]

    return all_elements


def write_cells_to_vtk_file(cells, filename="points"):
    # Open csv file to write into
    file = open(filename, "a")

    for cell in cells:
        position = cell["element"]["cell"]["mechanics"]["pos"]
        radius = cell["element"]["cell"]["interaction"]["cell_radius"]
        # food = cell["element"]["cell"]["cellular_reactions"]["intracellular_concentrations"][0]
        species = cell["element"]["cell"]["interaction"]["species"]
        file.write(
            "{}, {}, {}, {}, {}\n".format(position[0], position[1], position[2], radius, species)
        )

def plot_results(ofile):
    # Load all the data 
    df = pd.read_csv(ofile, header=None)

    pset = pv.PolyData(np.array(df[[0, 1, 2]]))
    pset.point_data["radius"] = 2.0*df[3]
    pset.point_data["species"] = df[4]

    sphere = pv.Sphere()
    spheres = pset.glyph(geom=sphere, scale="radius", orient=False)

    spheres.plot(
        off_screen=True,
        screenshot=ofile + ".png",
        scalars="species",
        scalar_bar_args={
            "title":"Species",
        },
        cpos=[(-150,-150,-150),(50,50,50),(0.0, 0.0, 0.0)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='cellular_raza analysis script',
        description='Automatic Conversion of Position, Radius and Intracellular concentration',
    )
    parser.add_argument('json_folder')
    parser.add_argument('output_folder')
    args = parser.parse_args()
    path = Path(args.json_folder)
    output_path = Path(args.output_folder)
    # path = "out/bacteria_population/2023-05-25-16:19:57/cell_storage/json/"

    for iteration_folder in glob(str(path)+ "/*/"):
        # print("Writing data for iteration {}".format(iteration_folder))
        iteration = os.path.basename(os.path.dirname(iteration_folder))
        cells = get_all_items_at_iteration(iteration_folder)

        filename = str(output_path / "points_{}.csv".format(iteration))
        write_cells_to_vtk_file(cells, filename)

    pool = mp.Pool()
    pool.map(plot_results, glob(str(output_path) + "/*.csv"))
