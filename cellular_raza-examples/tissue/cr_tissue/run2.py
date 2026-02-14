import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
from tqdm.contrib.concurrent import process_map
import itertools

import cr_tissue as crf


def get_area(middle, vertices) -> float:
    area = 0.0

    for n in range(len(vertices)):
        x = vertices[n] - middle
        y = vertices[(n + 1) % len(vertices)] - middle
        area += 0.5 * np.abs(x[1] * y[0] - x[0] * y[1])
    return area


def is_circle_in_convex_polygon(middle, radius, polygon):
    """
    Checks if a circle is entirely contained within a convex polygon.
    :param polygon: List of tuples [(x1, y1), (x2, y2), ...]
    """
    n = len(polygon)

    for i in range(n):
        # Current edge vertices
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]  # Wraps around to the first point

        # 1. Check if center is on the 'inside' of the edge using cross product
        # For CCW polygons, the point must be to the left of the edge
        edge_dx = p2[0] - p1[0]
        edge_dy = p2[1] - p1[1]
        point_dx = middle[0] - p1[0]
        point_dy = middle[1] - p1[1]

        # # Z-component of cross product
        cross_product = edge_dx * point_dy - edge_dy * point_dx

        # # If cross_product < 0, the center is outside the convex hull
        # if cross_product < 0:
        #     return False

        # 2. Check distance from center to the infinite line of the edge
        # Distance from point (x0, y0) to line Ax + By + C = 0
        # Line eq: (y1-y2)x + (x2-x1)y + (x1y2 - x2y1) = 0
        dist = abs(cross_product) / np.sqrt(edge_dx**2 + edge_dy**2)

        if dist < radius:
            return False

    return True


def min_max_exact_dist_point_to_segment(p, a, b, dist):
    """
    Calculates the minimum distance between point P(px, py)
    and line segment AB defined by points A(ax, ay) and B(bx, by).
    """
    # Vector AB
    dx = b - a

    # If A and B are the same point, distance is just P to A
    d = np.linalg.norm(dx) ** 2
    if d == 0:
        return np.linalg.norm(p - a), a, a, [a]

    # Calculate the projection of P onto the line AB
    # t is the "parameter" along the line: t=0 is A, t=1 is B
    t = np.dot((p - a), dx) / d

    # Clamp t to the range [0, 1] to stay on the segment
    t = np.clip(t, 0, 1)

    # Find the closest point on the segment
    closest = a + t * dx

    d1 = np.linalg.norm(a - p)
    d2 = np.linalg.norm(b - p)
    furthest = a if d1 > d2 else b
    min_dist = np.linalg.norm(closest - p)
    exact = []
    if d1 >= dist >= min_dist:
        h = (dist - min_dist) / (d1 - min_dist)
        exact.append(closest + h * (a - closest))
    if d2 >= dist >= min_dist:
        h = (dist - min_dist) / (d2 - min_dist)
        exact.append(closest + h * (b - closest))
    # Return the distance from P to the closest point
    return min_dist, closest, furthest, exact


def determine_new_radius(middle, vertices, radius):
    # Construct new polygon
    radius_new = radius
    new_polygon = []
    for n in range(len(vertices)):
        v1 = vertices[n]
        v2 = vertices[(n + 1) % len(vertices)]
        _, closest, furthest, exact = min_max_exact_dist_point_to_segment(
            middle, v1, v2, radius_new
        )
        new_polygon.extend(exact)

    return np.array(new_polygon)


def __construct_polygon(path, resolution):
    polygon = []
    for segm in path:
        segment_ty = segm[0]
        if segment_ty == "line":
            x = np.array([segm[1], segm[2]])
            polygon.extend(x)
        elif segment_ty == "arc":
            p = segm[1]
            theta1 = segm[2]
            theta2 = segm[3]
            r = segm[4]
            theta = np.linspace(theta1, theta2, resolution)
            points = np.vstack(
                (
                    r * np.cos(theta) + p[0],
                    r * np.sin(theta) + p[1],
                )
            ).T[::-1]
            polygon.extend(points)

    return np.array(polygon)


def save_snapshot(iteration, domain_size, result, resolution=30):
    fig, ax = plt.subplots(figsize=(8, 8))
    for middle, path, species in result[iteration]:
        color = mpl.colormaps["Set1"](species)
        if len(path) > 0:
            polygon = __construct_polygon(path, resolution)
            ax.add_patch(
                mpl.patches.Polygon(
                    polygon,
                    facecolor=color,
                    linestyle="-",
                    edgecolor="k",
                )
            )

    centers = np.array([m[0] for m in result[iteration]])
    ax.scatter(centers[:, 0], centers[:, 1], marker=".", color="k")

    dx = domain_size
    ax.set_xlim(-0.01 * dx, 1.01 * domain_size)
    ax.set_ylim(-0.01 * dx, 1.01 * domain_size)

    fig.savefig(f"out/{iteration:010}.png")
    plt.close(fig)


def __pool_save_snapshot_helper(args):
    return save_snapshot(*args)


if __name__ == "__main__":
    settings = crf.SimulationSettings()

    settings.n_plants = 40
    settings.n_threads = 1
    settings.dt = 0.5
    settings.t_max = 11_000.0
    settings.save_interval = 50.0
    settings.domain_size = 200.0

    settings.target_area = 80
    settings.force_strength = 0.05
    settings.force_strength_weak = -0.001
    settings.force_strength_species = 0.03
    settings.force_relative_cutoff = 5.0
    settings.damping_constant = 1.0
    settings.cell_diffusion_constant = 0.002

    sampler = sp.stats.qmc.LatinHypercube(d=2, seed=10)
    domain_size = settings.domain_size
    samples = sampler.random(300)
    samples = sp.stats.qmc.scale(samples, 0.05 * domain_size, 0.95 * domain_size)

    # samples = np.array(
    #     [
    #         [0.49 * domain_size, 0.49 * domain_size],
    #         [0.49 * domain_size, 0.51 * domain_size],
    #         [0.51 * domain_size, 0.49 * domain_size],
    #         [0.51 * domain_size, 0.51 * domain_size],
    #     ]
    # )

    species = np.zeros(samples.shape[0], dtype=int)
    species[::3] = 1
    species[1::3] = 2

    result = crf.run_simulation(
        settings,
        plant_points=samples,
        plant_species=species,
    )
    print()

    arglist = zip(
        result, itertools.repeat(settings.domain_size), itertools.repeat(result)
    )

    r = process_map(__pool_save_snapshot_helper, arglist, workers=-1)
    # pool = mp.Pool(30)
    # _ = list(tqdm(pool.imap(__pool_save_snapshot_helper, arglist), total=len(result)))

    # for iteration in tqdm(result, desc="Rendering Snapshots"):
    #     save_snapshot(iteration, settings.domain_size, result)
