use itertools::Itertools;
use nalgebra::{Matrix2xX, Vector, Vector2, VectorView2};

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
