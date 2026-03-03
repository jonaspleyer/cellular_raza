use itertools::Itertools;
use nalgebra::{Matrix2xX, Vector2, VectorView2};

pub const EPSILON: f64 = 1e-6;

pub fn get_polygon_area(vertices: &nalgebra::Matrix2xX<f64>) -> f64 {
    let mut area = 0.0;
    let n_col = vertices.ncols();
    for i in 0..n_col {
        let v1 = vertices.column(i);
        let v2 = vertices.column((i + 1) % n_col);
        area += v1.perp(&v2);
    }
    0.5 * area.abs()
}

pub fn get_polygon_perimeter(vertices: &nalgebra::Matrix2xX<f64>) -> f64 {
    let mut perimeter = 0.0;
    let n = vertices.ncols();
    for i in 0..n {
        perimeter += (vertices.column(i) - vertices.column((i + 1) % n)).norm();
    }
    perimeter
}

pub fn minimum_dist_to_segment(
    point: &Vector2<f64>,
    v1: VectorView2<f64>,
    v2: VectorView2<f64>,
) -> f64 {
    let l2 = (v1 - v2).norm_squared();
    if approx::abs_diff_eq!(l2, 0.0) {
        return (point - v1).norm();
    }
    let t = ((point - v1).dot(&(v2 - v1)) / l2).clamp(0.0, 1.0);
    let closest_point = v1 + t * (v2 - v1);
    (point - closest_point).norm()
}

pub fn area_centroid(vertices: &Matrix2xX<f64>) -> Vector2<f64> {
    let n = vertices.ncols();
    let mut signed_area = 0.0;
    let mut centroid = Vector2::zeros();
    for i in 0..n {
        let p1 = vertices.column(i);
        let p2 = vertices.column((i + 1) % n);
        let cross = p1.perp(&p2);
        signed_area += cross;
        centroid += (p1 + p2) * cross;
    }
    signed_area *= 0.5;
    centroid / (6.0 * signed_area)
}

pub fn intersect(
    p1: Vector2<f64>,
    p2: Vector2<f64>,
    line_pt: Vector2<f64>,
    normal: Vector2<f64>,
) -> Vector2<f64> {
    let d1 = (p1 - line_pt).dot(&normal);
    let d2 = (p2 - line_pt).dot(&normal);
    let t = d1 / (d1 - d2);
    p1 + t * (p2 - p1)
}

fn is_on_edge_approx(poly: &Matrix2xX<f64>, q: &VectorView2<f64>) -> bool {
    poly.column_iter()
        .circular_tuple_windows::<(_, _)>()
        .any(|(p1, p2)| {
            let dir = p2 - p1;
            let p = Vector2::from([q.x, q.y]);
            if approx::abs_diff_eq!(dir.norm(), 0.0, epsilon = EPSILON) {
                return approx::abs_diff_eq!((p2 - p).norm(), 0.0, epsilon = EPSILON);
            }
            let t = dir.dot(&(p - p1)) / dir.norm_squared();
            let pnew = p1 + t * dir;
            approx::abs_diff_eq!((pnew - p).norm(), 0.0, epsilon = EPSILON)
        })
}

pub fn intersect_polygons(poly1: &Matrix2xX<f64>, poly2: &Matrix2xX<f64>) -> Vec<Matrix2xX<f64>> {
    use i_overlay::core::fill_rule::FillRule;
    use i_overlay::core::overlay_rule::OverlayRule;
    use i_overlay::float::overlay::FloatOverlay;
    use i_overlay::float::single::SingleFloatOverlay;

    let subj = poly1
        .column_iter()
        .map(|col| [col[0], col[1]])
        .collect::<Vec<_>>();
    let clip = poly2
        .column_iter()
        .map(|col| [col[0], col[1]])
        .collect::<Vec<_>>();

    let options = i_overlay::float::overlay::OverlayOptions {
        preserve_input_collinear: true,
        preserve_output_collinear: true,
        clean_result: false,
        ..Default::default()
    };
    let solver = Default::default();
    let mut float_overlay = FloatOverlay::with_subj_and_clip_custom(&subj, &clip, options, solver);
    // Returns a list of shapes
    // A shape is a list of contours
    // A contour is a list of points
    //
    // These are the types used in i_overlay
    // pub type Contour<P> = Vec<P>;
    // pub type Shape<P> = Vec<Contour<P>>;
    // pub type Shapes<P> = Vec<Shape<P>>;

    let result = float_overlay.overlay(OverlayRule::Intersect, FillRule::EvenOdd);

    result
        .iter()
        .map(|r| Matrix2xX::from_fn(r[0].len(), |i, j| r[0][j][i]))
        .collect()
}

pub fn apply_restrictions(
    cell_pos: &Matrix2xX<f64>,
    pos_helper: &mut Matrix2xX<f64>,
    cell_pos_other: &Matrix2xX<f64>,
) {
    fn get_linestring_centroid(
        ls: &Vec<(Option<usize>, usize)>,
        poly: &Matrix2xX<f64>,
    ) -> Vector2<f64> {
        if ls.len() == 1 {
            let x = poly.column(ls[0].1);
            return Vector2::from([x.x, x.y]);
        }
        let mut total_length = 0.0;
        let mut ls_centroid = Vector2::zeros();
        for ((_, i), (_, j)) in ls.iter().tuple_windows() {
            let x1 = poly.column(*i);
            let x2 = poly.column(*j);
            let d = (x1 - x2).norm();
            total_length += d;
            ls_centroid += 0.5 * (x1 + x2) * d;
        }
        ls_centroid / total_length
    }

    // let poly1 = position_to_poly(pos1);
    // let poly2 = position_to_poly(pos2);

    // Determine all intersections
    // let intersection = poly1.intersection(&poly2);
    let intersection = intersect_polygons(&cell_pos, &cell_pos_other);

    for poly in intersection {
        // Determine area centroid
        let centroid = area_centroid(&poly);
        // Determine which points of the intersection belong to which polygon
        let mut l1 = Vec::with_capacity(poly.ncols());
        let mut l2 = Vec::with_capacity(poly.ncols());

        // Gather indices of points which are in the intersection
        let build_ls = |ls: &mut Vec<(Option<usize>, usize)>, pos: &Matrix2xX<f64>| {
            for (i1, pi) in pos.column_iter().enumerate() {
                // Check if it matches given coordinate
                if let Some(i_its) = poly
                    .column_iter()
                    .position(|x| approx::abs_diff_eq!((x - pi).norm(), 0.0, epsilon = EPSILON))
                {
                    ls.push((Some(i1), i_its));
                }
            }
        };
        build_ls(&mut l1, cell_pos);
        build_ls(&mut l2, cell_pos_other);

        // Add endpoints
        let mut n_endpoints = 0;
        for (i, q) in poly.column_iter().enumerate() {
            if is_on_edge_approx(&cell_pos, &q) && is_on_edge_approx(&cell_pos_other, &q) {
                l1.push((None, i));
                l2.push((None, i));
                n_endpoints += 1;
            }
        }

        fn sort_entries(
            x: &(Option<usize>, usize),
            y: &(Option<usize>, usize),
        ) -> std::cmp::Ordering {
            match (x.0, y.0) {
                // Endpoints involved
                (None, None) => y.1.cmp(&x.1), // Note that this is inverted!
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, Some(_)) => std::cmp::Ordering::Less,
                // No endpoint involved
                (Some(i), Some(j)) => i.cmp(&j),
            }
        }
        l1.sort_by(sort_entries);
        l2.sort_by(sort_entries);

        // If the intersections are empty; we do nothing
        // If the intersections do not have 2 endpoints, we also do nothing
        if l1.is_empty() || l2.is_empty() || n_endpoints != 2 {
            continue;
        }
        let e1 = l1.remove(0);
        l1.push(e1);
        let e1 = l2.remove(0);
        l2.push(e1);

        // Determine linestring centroids for overlaps
        // - corresponding to polygon1
        // - corresponding to polygon2
        let l1c = get_linestring_centroid(&l1, &poly);
        let l2c = get_linestring_centroid(&l2, &poly);
        // Get orthogonal between centroids
        let dir = Vector2::from([-l2c[1] + l1c[1], l2c[0] - l1c[0]]).normalize();
        // Project all points along this axis
        for (i, _) in l1.iter() {
            if let Some(i) = i {
                let q = centroid;
                let t = dir.dot(&(cell_pos.column(*i) - q));
                pos_helper.set_column(*i, &(q + t * dir));
            }
        }
    }
}

pub fn calculate_bbox(position: &nalgebra::Matrix2xX<f64>) -> (Vector2<f64>, Vector2<f64>) {
    let mut blow = Vector2::from([f64::NEG_INFINITY; 2]);
    let mut bhigh = Vector2::from([f64::INFINITY; 2]);
    for n in 0..position.ncols() {
        blow[0] = blow[0].min(position[(0, n)]);
        blow[1] = blow[1].min(position[(1, n)]);
        bhigh[0] = bhigh[0].max(position[(0, n)]);
        bhigh[1] = bhigh[1].max(position[(1, n)]);
    }
    (blow, bhigh)
}

pub fn bounding_boxes_intersect(
    bbox1: &(Vector2<f64>, Vector2<f64>),
    bbox2: &(Vector2<f64>, Vector2<f64>),
) -> bool {
    let (min1, max1) = bbox1;
    let (min2, max2) = bbox2;
    let x_overlap = min1.x <= max2.x && max1.x >= min2.x;
    let y_overlap = min1.y <= max2.y && max1.y >= min2.y;
    x_overlap && y_overlap
}

fn segments_intersect(
    p1: &VectorView2<f64>,
    p2: &VectorView2<f64>,
    q1: &VectorView2<f64>,
    q2: &VectorView2<f64>,
) -> bool {
    // Helper to find the orientation of ordered triplet (p, q, r).
    //  0 -> p, q and r are collinear
    //  1 -> Clockwise
    // -1 -> Counterclockwise
    fn orientation(p: &VectorView2<f64>, q: &VectorView2<f64>, r: &VectorView2<f64>) -> i32 {
        let val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
        if val == 0.0 {
            return 0;
        }
        if val > 0.0 {
            1
        } else {
            -1
        }
    }

    // Helper to check if point q lies on segment pr
    fn on_segment(p: &VectorView2<f64>, q: &VectorView2<f64>, r: &VectorView2<f64>) -> bool {
        q.x <= p.x.max(r.x) && q.x >= p.x.min(r.x) && q.y <= p.y.max(r.y) && q.y >= p.y.min(r.y)
    }

    let o1 = orientation(p1, p2, q1);
    let o2 = orientation(p1, p2, q2);
    let o3 = orientation(q1, q2, p1);
    let o4 = orientation(q1, q2, p2);

    false
        // General case: segments straddle each other
        || o1 != o2 && o3 != o4
        // Special Cases: collinearity and overlap
        || o1 == 0 && on_segment(p1, q1, p2)
        || o2 == 0 && on_segment(p1, q2, p2)
        || o3 == 0 && on_segment(q1, p1, q2)
        || o4 == 0 && on_segment(q1, p2, q2)
}

pub fn clean_self_intersections(pos: Matrix2xX<f64>) -> Matrix2xX<f64> {
    // Calculate the sum of all angles
    let ncols = pos.ncols();

    // Compare with every kth neighbor.
    // We can start at 2 and omit self-checking and direct neighbors
    let mut overlaps = Vec::new();
    for k in 2..ncols.div_ceil(2) {
        // Iterate over all vertices
        for i in 0..ncols {
            let i1 = i;
            let i2 = (i + 1) % ncols;
            // Pick kth neighbor
            let j1 = (i + k) % ncols;
            let j2 = (j1 + 1) % ncols;
            let p1 = pos.column(i2);
            let p2 = pos.column(i1);
            let q1 = pos.column(j1);
            let q2 = pos.column(j2);
            // Test if the line segments are crossing
            // if line_segments_are_crossing(&p1, &p2, &q1, &q2) {
            if segments_intersect(&p1, &p2, &q1, &q2) {
                overlaps.push((i1, j1, k));
            }
        }
    }

    // Sort overlaps by size: start with largest
    overlaps.sort_by_key(|x| x.2);

    let mut new_pos = pos;
    // ni = index start (initial)
    // nf = index end (final)
    // k = distance between ni and nf (accounting for roundtrips exceeding ncols)
    for (ni, nf, k) in overlaps {
        if k == 2 {
            // Change the two positions
            new_pos.swap_columns((ni + 1) % ncols, nf);
        } else {
            // Reorder every entry between i and j
            let pi: Vector2<f64> = new_pos.column(ni).into();
            let pf: Vector2<f64> = new_pos.column((nf + 1) % ncols).into();
            for m in 0..k {
                let n = (ni + m) % ncols;
                let s = (m + 1) as f64 / (k + 1) as f64;
                new_pos.set_column(n, &(pi + s * (pf - pi)));
            }
        }
    }
    new_pos
}

#[test]
fn test_apply_restrictions_1() {
    //        .
    //        .
    //        |
    //        |
    // ..---------
    //        |   |
    //         ---|----..
    //            |
    //            .
    //            .
    let poly1 = nalgebra::Matrix2xX::from_columns(&[
        nalgebra::vector![0.0, 0.0],
        nalgebra::vector![0.0, 1.0],
        nalgebra::vector![1.0, 1.0],
        nalgebra::vector![1.0, 0.0],
    ]);
    let poly2 = nalgebra::Matrix2xX::from_columns(&[
        nalgebra::vector![0.5, 0.5],
        nalgebra::vector![0.5, 1.5],
        nalgebra::vector![1.5, 1.5],
        nalgebra::vector![1.5, 0.5],
    ]);
    let mut poly_helper1 = poly1.clone();
    let mut poly_helper2 = poly2.clone();

    apply_restrictions(&poly1, &mut poly_helper1, &poly2);
    apply_restrictions(&poly2, &mut poly_helper2, &poly1);

    approx::assert_abs_diff_eq!(poly_helper1[(0, 2)], 0.75);
    approx::assert_abs_diff_eq!(poly_helper1[(1, 2)], 0.75);
    approx::assert_abs_diff_eq!(poly_helper2[(0, 0)], 0.75);
    approx::assert_abs_diff_eq!(poly_helper2[(1, 0)], 0.75);
}

#[test]
fn test_apply_restrictions_2() {
    // \                     /
    //  \ --------.-------- /
    //  /\________.________/\
    // /                     \
    let poly1 = nalgebra::Matrix2xX::from_columns(&[
        nalgebra::vector![0.0, 0.0],
        nalgebra::vector![0.0, 1.0],
        nalgebra::vector![0.2, 1.2],
        nalgebra::vector![0.5, 1.2],
        nalgebra::vector![0.8, 1.2],
        nalgebra::vector![1.0, 1.0],
        nalgebra::vector![1.0, 0.0],
    ]);
    let poly2 = nalgebra::Matrix2xX::from_columns(&[
        nalgebra::vector![0.0, 2.3],
        nalgebra::vector![1.0, 2.3],
        nalgebra::vector![1.0, 1.3],
        nalgebra::vector![0.8, 1.1],
        nalgebra::vector![0.5, 1.1],
        nalgebra::vector![0.2, 1.1],
        nalgebra::vector![0.0, 1.3],
    ]);

    let mut poly_helper1 = poly1.clone();
    let mut poly_helper2 = poly2.clone();

    apply_restrictions(&poly1, &mut poly_helper1, &poly2);
    apply_restrictions(&poly2, &mut poly_helper2, &poly1);

    let expected_1 = nalgebra::Matrix2xX::from_columns(&[
        nalgebra::vector![0.0, 0.0],
        nalgebra::vector![0.0, 1.0],
        nalgebra::vector![0.2, 1.15],
        nalgebra::vector![0.5, 1.15],
        nalgebra::vector![0.8, 1.15],
        nalgebra::vector![1.0, 1.0],
        nalgebra::vector![1.0, 0.0],
    ]);
    let expected_2 = nalgebra::Matrix2xX::from_columns(&[
        nalgebra::vector![0.0, 2.3],
        nalgebra::vector![1.0, 2.3],
        nalgebra::vector![1.0, 1.3],
        nalgebra::vector![0.8, 1.15],
        nalgebra::vector![0.5, 1.15],
        nalgebra::vector![0.2, 1.15],
        nalgebra::vector![0.0, 1.3],
    ]);

    for i in 0..2 {
        for j in 0..poly_helper1.ncols() {
            approx::assert_abs_diff_eq!(poly_helper1[(i, j)], expected_1[(i, j)], epsilon = 1e-4);
            approx::assert_abs_diff_eq!(poly_helper2[(i, j)], expected_2[(i, j)], epsilon = 1e-4);
        }
    }
}

#[test]
fn test_segment_intersect_1() {
    let p1 = nalgebra::vector![0.0, 0.0];
    let p2 = nalgebra::vector![1.0, 0.0];
    let q1 = nalgebra::vector![0.0, 1.0];
    let q2 = nalgebra::vector![1.0, 1.0];
    let r = segments_intersect(&p1.as_view(), &p2.as_view(), &q1.as_view(), &q2.as_view());
    assert!(!r);
}

#[test]
fn test_segment_intersect_2() {
    let p1 = nalgebra::vector![0.0, 0.0];
    let p2 = nalgebra::vector![1.0, 1.0];
    let q1 = nalgebra::vector![0.0, 1.0];
    let q2 = nalgebra::vector![1.0, 0.0];
    let r = segments_intersect(&p1.as_view(), &p2.as_view(), &q1.as_view(), &q2.as_view());
    assert!(r);
}

#[test]
fn test_segment_intersect_3() {
    let p1 = nalgebra::vector![0.0, 0.0];
    let p2 = nalgebra::vector![1.0, 0.0];
    let q1 = nalgebra::vector![0.0, 1.0];
    let q2 = nalgebra::vector![1.0, 0.0];
    let r = segments_intersect(&p1.as_view(), &p2.as_view(), &q1.as_view(), &q2.as_view());
    assert!(r);
}

#[test]
fn test_clean_self_intersections_swap() {
    let pos = nalgebra::Matrix2xX::from_columns(&[
        nalgebra::vector![0.0, 0.0],
        nalgebra::vector![0.0, 1.0],
        nalgebra::vector![1.0, 1.0],
        nalgebra::vector![0.8, 0.4],
        nalgebra::vector![0.8, 0.6],
        nalgebra::vector![1.0, 0.0],
    ]);

    let new_pos = clean_self_intersections(pos);

    let expected = nalgebra::Matrix2xX::from_columns(&[
        nalgebra::vector![0.0, 0.0],
        nalgebra::vector![0.0, 1.0],
        nalgebra::vector![1.0, 1.0],
        nalgebra::vector![0.8, 0.6],
        nalgebra::vector![0.8, 0.4],
        nalgebra::vector![1.0, 0.0],
    ]);

    for j in 0..new_pos.ncols() {
        for i in 0..2 {
            assert!(new_pos[(i, j)] == expected[(i, j)]);
        }
    }
}
