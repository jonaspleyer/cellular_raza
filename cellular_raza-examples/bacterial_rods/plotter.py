import pyvista as pv
import numpy as np
import argparse
import itertools
import tqdm
import concurrent.futures

from plot import *

def get_cell_meshes(iteration: int, path: Path):
    cells = load_cells_from_iteration(path, iteration)
    positions = np.array([x for x in cells["cell.pos"]], dtype=float)
    radii = np.array([x for x in cells["cell.radius"]], dtype=float)
    growth_rate = np.array([x for x in cells["cell.growth_rate"]], dtype=float)
    cell_surfaces = []
    for i, p in enumerate(positions):
        meshes = []
        # Add the sphere at the first point
        meshes.append(pv.Sphere(center=p[:,0], radius=radii[i]))
        for j in range(max(p.shape[1]-1,0)):
            # Add a sphere for each following point
            meshes.append(pv.Sphere(center=p[:,j+1], radius=radii[i]))

            # Otherwise add cylinders
            pos1 = p[:,j]
            pos2 = p[:,j+1]
            center = 0.5 * (pos1 + pos2)
            direction = pos2 - center
            radius = radii[i]
            height = np.linalg.norm(pos1 - pos2)
            # sphere = pv.Sphere(center=pos, radius=radii[i])
            # meshes.append(sphere)
            cylinder = pv.Cylinder(center, direction, radius, height)
            meshes.append(cylinder)
        merged = pv.MultiBlock(meshes).combine().extract_surface().clean()
        cell_surfaces.append((merged, growth_rate[i]))
    return cell_surfaces

def plot_spheres(iteration: int, path: Path = Path("./"), opath: Optional[Path] = None, overwrite:bool = False):
    if opath == None:
        opath = path / "images/{:010}.png".format(iteration)
        opath.parent.mkdir(parents=True, exist_ok=True)
    if os.path.isfile(opath) and overwrite==False:
        return None
    cell_meshes = get_cell_meshes(iteration, path)

    # General Settings
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background([100, 100, 100])

    # Draw box around everything
    dx = 3e-6
    box = pv.Box(bounds=(-dx, 200e-6+dx, 0-dx, 15e-6+dx, 0-dx, 45e-6+dx))
    plotter.add_mesh(box, style="wireframe")
    color_min = np.array([69, 124, 214])
    color_max = np.array([82, 191, 106])
    for (cell, growth_rate) in cell_meshes:
        q = growth_rate / (3e-6 / 60)
        color = (1-q) * color_min + q * color_max
        plotter.add_mesh(
            cell,
            show_edges=False,
            color=(int(color[0]), int(color[1]), int(color[2])),
        )

    # Define camera
    plotter.camera.position = (100e-6, -250e-6, -250e-6)
    plotter.camera.focal_point = (100e-6, 25e-6, 22.5e-6)

    plotter.enable_ssao(radius=12)
    plotter.enable_anti_aliasing()
    img = plotter.screenshot(opath)
    plotter.close()
    return img

def __plot_spheres_helper(args):
    (args, kwargs) = args
    plot_spheres(*args, **kwargs)

def plot_all_spheres(path: Path, n_threads: Optional[int] = None, overwrite:bool=False):
    iterations = [it for it in get_all_iterations(path)[1]]
    if n_threads==None:
        n_threads = mp.cpu_count()-2
    args = [(it,) for it in iterations]
    kwargs = {
        "path": path,
        "overwrite": overwrite,
    }
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
        _ = list(tqdm.tqdm(executor.map(
            __plot_spheres_helper,
            zip(args, itertools.repeat(kwargs))
        ), total=len(iterations)))

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
    print("Plotting Individual Snapshots")
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
    print("Playing Movie")
    bashcmd2 = f"firefox {output_path}/movie.mp4"
    os.system(bashcmd2)

