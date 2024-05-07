import pyvista as pv
import numpy as np
import argparse
import itertools
import tqdm

from plot import *

def get_cell_meshes(iteration: int, path: Path):
    cells = load_cells_from_iteration(path, iteration)
    positions = np.array([x for x in cells["cell.pos"]], dtype=float)
    radii = np.array([x for x in cells["cell.radius"]], dtype=float)
    cell_surfaces = []
    for i, p in enumerate(positions):
        spheres = []
        for j in range(p.shape[1]):
            pos = p[:,j]
            sphere = pv.Sphere(center=pos, radius=radii[i])
            spheres.append(sphere)
        merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
        cell_surfaces.append(merged)
    # pset = pv.PolyData([np.array(x[0]) for x in position_radius_species])
    # pset.point_data["diameter"] = 2.0 * radii

    # sphere = pv.Sphere()
    # spheres = pset.glyph(geom=sphere, scale="diameter", orient=False)
    # return spheres
    return cell_surfaces

def plot_spheres(iteration: int, path: Path, opath = None):
    cell_meshes = get_cell_meshes(iteration, path)

    plotter = pv.Plotter(off_screen=True)
    plotter.set_background([100, 100, 100])
    for cell in cell_meshes:
        plotter.add_mesh(
            cell,
            show_edges=False,
        )
    plotter.enable_ssao(radius=12)
    plotter.enable_anti_aliasing()
    if opath == None:
        opath = path / "images/{:010}.png".format(iteration)
        opath.parent.mkdir(parents=True, exist_ok=True)
    img = plotter.screenshot(opath)
    plotter.close()
    return img

def __plot_spheres_helper(args):
    plot_spheres(*args)

def plot_all_spheres(path: Path):
    iterations = [it for it in get_all_iterations(path)[1]]
    pool = mp.Pool()
    list(
        tqdm.tqdm(
            pool.imap_unordered(
                __plot_spheres_helper,
                zip(iterations, itertools.repeat(path))
            ),
            total=len(iterations)
    ))

if __name__ == "_main__":
    parser = argparse.ArgumentParser(
        prog="Plotter",
        description="A plotting CLI for the springs example",
        epilog="For suggestions or bug-reports please refer to\
            https://github.com/jonaspleyer/cellular_raza/"
    )
    parser.add_argument(
        "-n",
        "--iteration",
        help="Specify which iteration to plot.\
            Multiple arguments are accepted as a list.\
            If not specified, plot all iterations."
    )
    parser.add_argument(
        "-i",
        "--input",
        help=f"Input path of files. If not given,\
            it will be determined automatically by searching in './out/'"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="Path where to store generated images."
    )

    args = parser.parse_args()

if __name__ == "__main__":
    output_path = get_last_output_path()
    plot_all_spheres(output_path)
    print("Generating Movie")
    bashcmd = f"ffmpeg\
        -v quiet\
        -stats\
        -y\
        -r 30\
        -f image2\
        -pattern_type glob\
        -i '{output_path}/images/*.png'\
        -c:v h264\
        -pix_fmt yuv420p\
        -strict -2 {output_path}/movie.mp4"
    os.system(bashcmd)
    bashcmd2 = f"vlc {output_path}/movie.mp4"
    os.system(bashcmd2)

