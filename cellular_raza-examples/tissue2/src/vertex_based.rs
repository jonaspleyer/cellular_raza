use nalgebra::Matrix2;
use nalgebra::Vector2;
use pyo3::prelude::*;

fn compute_cell_vertices(
    target_pos: Vector2<f64>,
    neighbor_positions: &[Vector2<f64>],
) -> Vec<Vector2<f64>> {
    // Step 1: Compute perpendicular bisectors between target cell and each neighbor
    let mut bisectors = Vec::new();
    for &neighbor in neighbor_positions {
        let midpoint = (target_pos + neighbor) / 2.0;
        let dir = neighbor - target_pos;
        // Perpendicular direction
        let perp = Vector2::new(-dir.y, dir.x);
        bisectors.push((midpoint, perp));
    }

    // Step 2: Compute polygon vertices as intersections of consecutive bisectors
    let mut vertices = Vec::new();
    let n = bisectors.len();
    for i in 0..n {
        let (p1, d1) = bisectors[i];
        let (p2, d2) = bisectors[(i + 1) % n];

        // Solve linear system: p1 + t1 * d1 = p2 + t2 * d2
        let a = Matrix2::from_columns(&[d1, -d2]);
        if let Some(t) = a.try_inverse() {
            let t_vec = t * (p2 - p1);
            let vertex = p1 + t_vec[0] * d1;
            vertices.push(vertex);
        }
    }

    // Step 3: Order vertices counterclockwise around target_pos
    vertices.sort_by(|a, b| {
        let angle_a = (a.y - target_pos.y).atan2(a.x - target_pos.x);
        let angle_b = (b.y - target_pos.y).atan2(b.x - target_pos.x);
        angle_a.partial_cmp(&angle_b).unwrap()
    });

    vertices
}

/// Compute the area of a polygon given ordered vertices
fn polygon_area(vertices: &[Vector2<f64>]) -> f64 {
    let n = vertices.len();
    let mut area = 0.0;
    for i in 0..n {
        let v0 = vertices[i];
        let v1 = vertices[(i + 1) % n];
        area += v0.x * v1.y - v1.x * v0.y;
    }
    0.5 * area.abs()
}

/// Compute the perimeter of a polygon
fn polygon_perimeter(vertices: &[Vector2<f64>]) -> f64 {
    let n = vertices.len();
    let mut perimeter = 0.0;
    for i in 0..n {
        let v0 = vertices[i];
        let v1 = vertices[(i + 1) % n];
        perimeter += (v1 - v0).norm();
    }
    perimeter
}

/// Compute forces on each vertex based on target area and perimeter
fn compute_vertex_forces(
    vertices: &[Vector2<f64>],
    target_area: f64,
    area_modulus: f64,
    perimeter_modulus: f64,
) -> Vec<Vector2<f64>> {
    let n = vertices.len();
    let mut forces = vec![Vector2::zeros(); n];

    // Compute current area and perimeter
    let area = polygon_area(vertices);
    let perimeter = polygon_perimeter(vertices);

    for i in 0..n {
        let v_prev = vertices[(i + n - 1) % n];
        let v = vertices[i];
        let v_next = vertices[(i + 1) % n];

        // Gradient of area w.r.t. vertex position
        // ∂A/∂r_i = 0.5 * ( (y_{i+1} - y_{i-1}), (x_{i-1} - x_{i+1}) )
        let dA = Vector2::new(0.5 * (v_next.y - v_prev.y), 0.5 * (v_prev.x - v_next.x));

        // Gradient of perimeter w.r.t. vertex position
        // ∂P/∂r_i = (r_i - r_prev)/|r_i - r_prev| + (r_i - r_next)/|r_i - r_next|
        let mut dP = Vector2::zeros();
        let edge_prev = v - v_prev;
        let edge_next = v - v_next;
        if edge_prev.norm() > 1e-12 {
            dP += edge_prev.normalize();
        }
        if edge_next.norm() > 1e-12 {
            dP += edge_next.normalize();
        }

        // Force: F = -2 K (A - A0) ∂A/∂r - 2 Γ P ∂P/∂r
        let f = -2.0 * area_modulus * (area - target_area) * dA
            - 2.0 * perimeter_modulus * perimeter * dP;
        forces[i] = f;
    }

    forces
}

#[pyfunction]
pub fn construct_polygons<'py>(
    python: Python<'py>,
    radius: f64,
    middle: numpy::PyReadonlyArray1<f64>,
    others: numpy::PyReadonlyArray2<f64>,
) -> Bound<'py, numpy::PyArray2<f64>> {
    let middle = middle.as_array();
    let others = others.as_array();

    let target_pos = Vector2::from([middle[0], middle[1]]);
    let neighbor_positions: Vec<_> = (0..others.shape()[0])
        .map(|i| Vector2::from([others[(i, 0)], others[(i, 1)]]))
        .filter(|x| (x - target_pos).norm() < 2.0 * radius)
        .collect();

    let vertices = compute_cell_vertices(target_pos, &neighbor_positions);
    let mut out = numpy::ndarray::Array2::zeros((vertices.len(), 2));
    for (k, v) in vertices.into_iter().enumerate() {
        out[(k, 0)] = v[0];
        out[(k, 1)] = v[1];
    }

    numpy::PyArray2::from_array(python, &out)
}

#[test]
fn test() {
    // Example: vertices of a target cell (could come from previous function)
    let vertices = vec![
        Vector2::new(0.5, 0.0),
        Vector2::new(0.25, 0.5),
        Vector2::new(-0.25, 0.5),
        Vector2::new(-0.5, 0.0),
        Vector2::new(-0.25, -0.5),
        Vector2::new(0.25, -0.5),
    ];

    let target_area = 1.0;
    let area_modulus = 1.0;
    let perimeter_modulus = 0.5;

    let forces = compute_vertex_forces(&vertices, target_area, area_modulus, perimeter_modulus);
    println!("Vertex forces:");
    for f in forces {
        println!("{:?}", f);
    }
}
