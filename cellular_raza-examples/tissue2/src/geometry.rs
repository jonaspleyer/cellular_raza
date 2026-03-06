use i_overlay::i_float::{adapter::FloatPointAdapter, int::point::IntPoint};
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

pub fn closest_point_on_segment(
    point: &VectorView2<f64>,
    v1: &VectorView2<f64>,
    v2: &VectorView2<f64>,
) -> (f64, Vector2<f64>) {
    let l2 = (v1 - v2).norm_squared();
    if approx::abs_diff_eq!(l2, 0.0) {
        return (0.0, (*v1).into());
    }
    let t = ((point - v1).dot(&(v2 - v1)) / l2).clamp(0.0, 1.0);
    (t, v1 + t * (v2 - v1))
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

pub fn area_centroid_with_adapter(
    slice: &[IntPoint],
    adapter: &FloatPointAdapter<[f64; 2], f64>,
) -> Vector2<f64> {
    let n = slice.len();
    let mut signed_area = 0.0;
    let mut centroid = Vector2::zeros();
    for i in 0..n {
        let p1 = Vector2::from(adapter.int_to_float(&slice[i]));
        let p2 = Vector2::from(adapter.int_to_float(&slice[(i + 1) % n]));
        let cross = p1.perp(&p2);
        signed_area += cross;
        centroid += (p1 + p2) * cross;
    }
    signed_area *= 0.5;
    if approx::abs_diff_eq!(signed_area, 0.0, epsilon = EPSILON.powi(2)) {
        centroid *= 0.0;
        for p in slice {
            centroid += Vector2::from(adapter.int_to_float(&p));
        }
        centroid / (n as f64)
    } else {
        centroid / (6.0 * signed_area)
    }
}

pub fn apply_restrictions(
    poly1: &Matrix2xX<f64>,
    poly2: &Matrix2xX<f64>,
    pos_helper: &mut Matrix2xX<f64>,
) {
    use i_overlay::core::fill_rule::FillRule;
    use i_overlay::core::overlay::Overlay;
    use i_overlay::core::overlay_rule::OverlayRule;
    use i_overlay::i_float::float::rect::FloatRect;

    let options = i_overlay::core::overlay::IntOverlayOptions {
        preserve_output_collinear: true,
        preserve_input_collinear: true,
        ..Default::default()
    };

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    poly1
        .column_iter()
        .chain(poly2.column_iter())
        .for_each(|col| {
            min_x = col[0].min(min_x);
            min_y = col[1].min(min_y);
            max_x = col[0].max(max_x);
            max_y = col[1].max(max_y);
        });

    let adapter = FloatPointAdapter::<[f64; 2], f64>::new(FloatRect {
        min_x,
        max_x,
        min_y,
        max_y,
    });

    let poly_to_points = |poly: &Matrix2xX<f64>| {
        poly.column_iter()
            .map(|col| adapter.float_to_int(&[col[0], col[1]]))
            .collect::<Vec<_>>()
    };
    let points1 = poly_to_points(poly1);
    let points2 = poly_to_points(poly2);

    let solver = Default::default();
    let mut overlay = Overlay::with_contour_custom(&points1, &points2, options, solver);
    let shapes = overlay.overlay(OverlayRule::Intersect, FillRule::EvenOdd);

    for shape in shapes {
        for contour in shape {
            // Determine the centroid of the intersection
            let centroid = area_centroid_with_adapter(&contour, &adapter);

            // Filter out all points which belong to Poly1
            // This vec will contain elements in the form
            // (i, (j, x))
            // * i: Index wrt. the contour
            // * j: Index wrt. the poly1 Polygon
            // * x IntPoint of the contour

            // Case 1: (Example sketch)
            // The contour vec starts with something that is not in polygon1
            // [P2, P2, P2, Its, P1, P1, ...]
            //
            // Case 2:
            // The contour vec starts with a point in polygon1
            // [P1, P1, P1, .. , P1?]
            // and it also might end with a point in polygon1
            // We can now infer the final point

            let mut new_contour = Vec::with_capacity(contour.len());
            let n = contour.len();
            let mut partial_start = false;
            for count in 0..2 * n {
                let i = count % n;
                // SAFETY: we have used modulo operator to ensure that bound is correct
                let x = unsafe { contour.get_unchecked(i) };
                let res = points1.iter().position(|y| y == x);

                if count == 0 && res.is_some() {
                    partial_start = true;
                }

                // If we did not have a partial start
                // but find something, we can start
                if let (false, Some(j)) = (partial_start, res) {
                    new_contour.push((i, (j, *x)));
                }
                // If we did not have a partial start and stop finding
                // after we found something, we can stop the loop entirely
                if !partial_start && res.is_none() && !new_contour.is_empty() {
                    break;
                }
                // If we had a partial start and stop finding
                // we reset the partial start
                if let (true, None) = (partial_start, res) {
                    partial_start = false;
                }
                // If we had a partial start and find stuff
                // we do nothing.
            }

            // We continue if there are no elements
            if new_contour.is_empty() {
                continue;
            }

            // SAFETY: We checked that the contour is not empty
            let i1 = (unsafe { new_contour.first().unwrap_unchecked().0 } + n - 1) % n;
            let i2 = (unsafe { new_contour.last().unwrap_unchecked().0 } + 1) % n;

            // SAFETY: We wrap with module to ensure that we stay within bounds
            let q0 = unsafe { contour.get_unchecked(i1) };
            let e1 = adapter.int_to_float(&q0);
            // Gets the index of the last element
            let e2 = unsafe { adapter.int_to_float(&contour.get(i2).unwrap_unchecked()) };
            let e1 = Vector2::from(e1);
            let e2 = Vector2::from(e2);

            // Helper method to calculate projection onto line segment
            let calculate_new_point = |x: &VectorView2<f64>, e: Vector2<f64>| {
                let d1 = centroid - e;
                if d1.norm() > 0.0 {
                    let dir = d1.normalize();
                    let t = dir.dot(&(x - centroid));
                    centroid + t * dir
                } else {
                    e1
                }
            };

            // SAFETY: We have checked that the vec is not empty and that i1
            // is an actual element by the methods just above.
            for (_, (j2, _)) in new_contour {
                let x2 = poly1.column(j2);

                // Approximate the slope
                /* let ncols1 = poly1.ncols();
                let j0 = (j2 + ncols1 - 2) % ncols1;
                let j1 = (j2 + ncols1 - 1) % ncols1;
                let j3 = (j2 + 1) % ncols1;
                let j4 = (j2 + 2) % ncols1;

                let x0 = poly1.column(j0);
                let x1 = poly1.column(j1);
                let x3 = poly1.column(j3);
                let x4 = poly1.column(j4);

                let dir = (x4 - x3) + 2.0 * (x3 - x2) + 2.0 * (x2 - x1) + (x1 - x0);
                let dir = if dir.norm() > 0.0 {
                    dir.normalize()
                } else {
                    (x3 - x1).normalize()
                };

                let t = dir.dot(&(x2 - centroid));
                pos_helper.set_column(j2, &(centroid + t * dir));*/

                // Project to line1 e1-->centroid and line2 e0-->centroid
                let v1 = calculate_new_point(&x2, e1);
                let v2 = calculate_new_point(&x2, e2);

                // Determine which point is closer to original point
                // and update helper
                let cond = (v1 - x2).norm() < (v2 - x2).norm();
                if cond {
                    pos_helper.set_column(j2, &v1);
                } else {
                    pos_helper.set_column(j2, &v2);
                };
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

    apply_restrictions(&poly1, &poly2, &mut poly_helper1);
    apply_restrictions(&poly2, &poly1, &mut poly_helper2);

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

    apply_restrictions(&poly1, &poly2, &mut poly_helper1);
    apply_restrictions(&poly2, &poly1, &mut poly_helper2);

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

#[test]
fn test_intersect_polygons() {
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
    let mut pos_helper = poly1.clone();
    apply_restrictions(&poly1, &poly2, &mut pos_helper);

    let expected = nalgebra::Matrix2xX::from_columns(&[
        nalgebra::vector![0.0, 0.0],
        nalgebra::vector![0.0, 1.0],
        nalgebra::vector![0.75, 0.75],
        nalgebra::vector![1.0, 0.0],
    ]);

    for i in 0..2 {
        for j in 0..poly1.ncols() {
            approx::assert_abs_diff_eq!(pos_helper[(i, j)], expected[(i, j)]);
        }
    }
}
