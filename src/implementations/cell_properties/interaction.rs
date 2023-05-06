use crate::concepts::errors::CalcError;
use crate::concepts::interaction::*;

use nalgebra::SVector;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoInteraction {}

impl<Pos, For> Interaction<Pos, For> for NoInteraction {
    fn calculate_force_on(
        &self,
        _: &Pos,
        _: &Pos,
        _ext_information: &Option<()>,
    ) -> Option<Result<For, CalcError>> {
        return None;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LennardJones {
    pub epsilon: f64,
    pub sigma: f64,
}

macro_rules! implement_lennard_jones_nd(
    ($dim:literal) =>  {
        impl Interaction<SVector<f64, $dim>,SVector<f64, $dim>> for LennardJones {
            fn calculate_force_on(&self, own_pos: &SVector<f64, $dim>, ext_pos: &SVector<f64, $dim>, _ext_information: &Option<()>) -> Option<Result<SVector<f64, $dim>, CalcError>> {
                let z = own_pos - ext_pos;
                let r = z.norm();
                let dir = z/r;
                let val = 4.0 * self.epsilon / r * (12.0 * (self.sigma/r).powf(12.0) - 1.0 * (self.sigma/r).powf(1.0));
                let max = 10.0 * self.epsilon / r;
                return Some(Ok(dir * max.min(val)));
            }
        }
    }
);

implement_lennard_jones_nd!(1);
implement_lennard_jones_nd!(2);
implement_lennard_jones_nd!(3);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VertexDerivedInteraction<A, R, I1 = (), I2 = ()> {
    pub outside_interaction: A,
    pub inside_interaction: R,
    phantom_inf_1: core::marker::PhantomData<I1>,
    phantom_inf_2: core::marker::PhantomData<I2>,
}

impl<A, R, I1, I2> VertexDerivedInteraction<A, R, I1, I2> {
    pub fn from_two_forces(attracting_force: A, repelling_force: R) -> Self
    where
        A: Interaction<Vector2<f64>, Vector2<f64>, I1>,
        R: Interaction<Vector2<f64>, Vector2<f64>, I2>,
    {
        VertexDerivedInteraction {
            outside_interaction: attracting_force,
            inside_interaction: repelling_force,
            phantom_inf_1: core::marker::PhantomData::<I1>,
            phantom_inf_2: core::marker::PhantomData::<I2>,
        }
    }
}

use itertools::Itertools;
use nalgebra::Vector2;

fn nearest_point_from_point_to_line(
    point: &Vector2<f64>,
    line: (Vector2<f64>, Vector2<f64>),
) -> (f64, Vector2<f64>) {
    let ab = line.1 - line.0;
    let ap = point - line.0;
    let t = (ab.dot(&ap) / ab.norm_squared()).clamp(0.0, 1.0);
    let nearest_point = (1.0 - t) * line.0 + t * line.1;
    ((point - nearest_point).norm(), nearest_point)
}

fn nearest_point_from_point_to_multiple_lines(
    point: &Vector2<f64>,
    polygon_lines: &[(Vector2<f64>, Vector2<f64>)],
) -> Option<(f64, Vector2<f64>)> {
    polygon_lines
        .iter()
        .map(|&line| nearest_point_from_point_to_line(point, line))
        .min_by(|(distance1, _), (distance2, _)| distance1.total_cmp(&distance2))
}

fn ray_intersects_line_segment(
    ray: &(Vector2<f64>, Vector2<f64>),
    line_segment: &(Vector2<f64>, Vector2<f64>),
) -> bool {
    // Calculate the intersection point as if the ray and line were infinite
    let (r1, r2) = ray;
    let (l1, l2) = line_segment;

    // We solve the formulas
    // p = r1 + t*(r2-r1)
    // p = l1 + u*(l2-l1)
    // This can be done by using the orthogonal complements for (r2-r1) and (l2-l1) respectively.
    // Let r_o and l_o be the orthogonals to (r2-r1) and (l2-l1).
    // Then the above formulas can be set equal and multiplied by either one of the orthogonals.
    // Note that x.dot(y_o) == x.perp(y) holds for the nalgebra crate!
    // Also the perpendicular product of a vector with itself is zero.
    // This yields the two formulas (without borrowing for better readability)
    //  => r1.perp(r2-r1) + t*(r2-r1).perp(r2-r1) = l1.perp(r2-r1) + u*(l2-l1).perp(r2-r1)
    // <=> r1.perp(r2-r1)                         = l1.perp(r2-r1) + u*(l2-l1).perp(r2-r1)
    // <=> (r1-l1).perp(r2-r1)                    =                  u*(l2-l1).perp(r2-r1)
    //  => l1.perp(l2-l1) + u*(l2-l1).perp(l2-l1) = r1.perp(l2-l1) + t*(r2-r1).perp(l2-l1)
    // <=> r1.perp(l2-l1) + t*(r2-r1).perp(l2-l1) = l1.perp(l2-l1)
    // <=>                  t*(r2-r1).perp(l2-l1) = (l1-r1).perp(l2-l1)

    // Split the result in enominator and denominator
    let t_enum = (l1 - r1).perp(&(l2 - l1));
    let u_enum = (r1 - l1).perp(&(r2 - r1));
    let t_denom = (r2 - r1).perp(&(l2 - l1));
    let u_denom = -t_denom;

    // If the denominators are zero, the following possibilities arise
    // 1) Either r1 and r2 are identical or l1 and l2
    // 2) The lines are parallel and cannot intersect
    if t_denom == 0.0 || u_denom == 0.0 {
        // We can directly test if some of the points are on the same line
        // Test this by calculating the dot product of r1-l1 with l2-l1.
        // This value must be between the norm of l2-l1 squared
        let d = (r1 - l1).dot(&(l2 - l1));
        let e = (l2 - l1).norm_squared();
        println!("Exit early");
        return 0.0 <= d && d <= e;
    }

    // Handles the case where the points for the ray were layed on top of each other.
    // Then the ray will only hit the point if r1 and r2 are on the line segment itself.
    let t = t_enum / t_denom;
    let u = u_enum / u_denom;

    // In order to be on the line-segment, we require that u is between 0.0 and 1.0
    // Additionally for p to be on the ray, we require t >= 0.0
    return 0.0 <= u && u < 1.0 && 0.0 <= t;
}

impl<A, R, I1, I2, const D: usize>
    Interaction<
        super::mechanics::VertexVector2<D>,
        super::mechanics::VertexVector2<D>,
        (Option<I1>, Option<I2>),
    > for VertexDerivedInteraction<A, R, I1, I2>
where
    A: Interaction<Vector2<f64>, Vector2<f64>, I1>,
    R: Interaction<Vector2<f64>, Vector2<f64>, I2>,
{
    fn get_interaction_information(&self) -> Option<(Option<I1>, Option<I2>)> {
        let i1 = self.outside_interaction.get_interaction_information();
        let i2 = self.inside_interaction.get_interaction_information();
        Some((i1, i2))
    }

    fn calculate_force_on(
        &self,
        own_pos: &super::mechanics::VertexVector2<D>,
        ext_pos: &super::mechanics::VertexVector2<D>,
        ext_information: &Option<(Option<I1>, Option<I2>)>,
    ) -> Option<Result<super::mechanics::VertexVector2<D>, CalcError>> {
        // TODO Reformulate this code:
        // Do not calculate interactions between vertices but rather between
        // edges of polygons.

        // IMPORTANT!!
        // This code assumes that we are dealing with a regular! polygon!

        // Calculate the middle point which we will need later
        let middle_own: Vector2<f64> = own_pos
            .row_iter()
            .map(|v| v.transpose())
            .sum::<Vector2<f64>>()
            / own_pos.shape().0 as f64;

        let middle_ext: Vector2<f64> = ext_pos
            .row_iter()
            .map(|v| v.transpose())
            .sum::<Vector2<f64>>()
            / ext_pos.shape().0 as f64;

        // Also calculate our own polygon defined by lines between points
        let own_polygon_lines = own_pos
            .row_iter()
            .map(|vec| vec.transpose())
            .circular_tuple_windows()
            .collect::<Vec<(_, _)>>();

        let vec_on_edge =
            0.5 * (own_pos.view_range(0..1, 0..2) + own_pos.view_range(1..2, 0..2)).transpose();
        let point_outside_polygon = 2.0 * vec_on_edge - middle_own;

        // Store the total calculated force here
        let mut total_force = ext_pos.clone() * 0.0;

        // Match the obtained interaction informations
        let (inf1, inf2) = match ext_information {
            Some(x) => x,
            None => &(None, None),
        };

        // Pick one point from the external positions
        // and calculate which would be the nearest point on the own positions
        for (point, mut force) in ext_pos
            .row_iter()
            .map(|vec| vec.transpose())
            .zip(total_force.row_iter_mut())
        {
            // Check if the point is inside the polygon.
            // If this is the case, do not calculate any attracting force.

            // We first calculate a bounding box and test quickly with this
            let bounding_box: [[f64; 2]; 2] = own_pos.row_iter().map(|v| v.transpose()).fold(
                [[std::f64::INFINITY, -std::f64::INFINITY]; 2],
                |mut accumulator, polygon_edge| {
                    accumulator[0][0] = accumulator[0][0].min(polygon_edge.x);
                    accumulator[0][1] = accumulator[0][1].max(polygon_edge.x);
                    accumulator[1][0] = accumulator[1][0].min(polygon_edge.y);
                    accumulator[1][1] = accumulator[1][1].max(polygon_edge.y);
                    accumulator
                },
            );

            let point_is_out_of_bounding_box = point.x < bounding_box[0][0]
                || point.x > bounding_box[0][1]
                || point.y < bounding_box[1][0]
                || point.y > bounding_box[1][1];

            let external_point_is_in_polygon = match point_is_out_of_bounding_box {
                true => false,
                false => {
                    // If the bounding box was not successful, we use the ray-casting algorithm to check.
                    let n_intersections: usize = own_polygon_lines
                        .iter()
                        .map(|line| {
                            ray_intersects_line_segment(&(point, point_outside_polygon), line)
                                as usize
                        })
                        .sum();

                    // An even number means that the point was outside while odd numbers mean that the point was inside.
                    n_intersections % 2 == 1
                }
            };

            let calc;
            if external_point_is_in_polygon {
                // Calculate the force inside the cell
                calc = self
                    .inside_interaction
                    .calculate_force_on(&middle_own, &middle_ext, &inf2);
            } else {
                // Calculate the force outside
                let (_, nearest_point) =
                    match nearest_point_from_point_to_multiple_lines(&point, &own_polygon_lines) {
                        Some(point) => point,
                        None => return None,
                    };
                calc = self
                    .outside_interaction
                    .calculate_force_on(&nearest_point, &point, &inf1);
            }
            match calc {
                Some(Ok(calculated_force)) => force += calculated_force.transpose(),
                Some(Err(error)) => return Some(Err(error)),
                None => (),
            }
        }
        Some(Ok(total_force))
    }
}

mod test {
    #[test]
    fn test_closest_points() {
        // Define the line we will be using
        let p1 = nalgebra::Vector2::from([0.0, 0.0]);
        let p2 = nalgebra::Vector2::from([2.0, 0.0]);

        // Create a vector of tuples which have (input_point, expected_nearest_point, expected_distance)
        let mut test_points = Vec::new();

        // Define the points we will be testing
        // Normal point which lies perpendicular to line
        test_points.push((
            nalgebra::Vector2::from([0.5, 1.0]),
            nalgebra::Vector2::from([0.5, 0.0]),
            1.0,
        ));

        // Point to check left edge of line
        test_points.push((nalgebra::Vector2::from([-1.0, 2.0]), p1, 5.0_f64.sqrt()));

        // Point to check right edge of line
        test_points.push((nalgebra::Vector2::from([3.0, -2.0]), p2, 5.0_f64.sqrt()));

        // Check if the distance and point are matching
        for (q, r, d) in test_points.iter() {
            let (dist, nearest_point) = super::nearest_point_from_point_to_line(&q, (p1, p2));
            assert_eq!(dist, *d);
            assert_eq!(nearest_point, *r);
        }
    }

    #[test]
    fn test_point_is_in_regular_polygon() {
        use itertools::Itertools;
        // Define the polygon for which we are testing
        let polygon = [
            nalgebra::Vector2::from([-1.0, 0.0]),
            nalgebra::Vector2::from([0.0, 1.0]),
            nalgebra::Vector2::from([1.0, 0.0]),
            nalgebra::Vector2::from([0.0, -1.0]),
        ];
        // This is the polygon
        //       / \
        //     /     \
        //   /         \
        // /             \
        // \             /
        //   \         /
        //     \     /
        //       \ /

        // For testing, we need to pick a point outside of the polygon
        let point_outside_polygon = nalgebra::Vector::from([-3.0, 0.0]);

        // Define points which should be inside the polygon
        let points_inside = [
            nalgebra::Vector2::from([0.0, 0.0]),
            nalgebra::Vector2::from([0.0, 0.1]),
            nalgebra::Vector2::from([0.0, 0.99999]),
            nalgebra::Vector2::from([0.99999, 0.0]),
            nalgebra::Vector2::from([0.0, 1.0]),
            // This point here was always a problem!
            // TODO write function above such that point [1.0, 0.0] is also inside!
            // For now this is enough since this is only a very narrow edge case
            nalgebra::Vector2::from([0.99999, 0.0]),
            nalgebra::Vector2::from([-1.0, 0.0]),
            nalgebra::Vector2::from([0.0, -1.0]),
        ];

        // Check the points inside the polygon
        for p in points_inside.iter() {
            let n_intersections: usize = polygon
                .clone()
                .into_iter()
                .circular_tuple_windows::<(_, _)>()
                .map(|line| {
                    super::ray_intersects_line_segment(&(*p, point_outside_polygon), &line) as usize
                })
                .sum();
            assert_eq!(n_intersections % 2 == 1, true);
        }

        // Define points which should be outside the polygon
        let points_outside = [
            nalgebra::Vector2::from([2.0, 0.0]),
            nalgebra::Vector2::from([-1.5, 0.0]),
            nalgebra::Vector2::from([0.5, 1.2]),
            nalgebra::Vector2::from([1.3, -1.0001]),
            nalgebra::Vector2::from([1.0000000000001, 0.0]),
            nalgebra::Vector2::from([0.0, -1.000000000001]),
        ];

        // Check them
        for q in points_outside.iter() {
            let n_intersections: usize = polygon
                .clone()
                .into_iter()
                .circular_tuple_windows()
                .map(|line| {
                    super::ray_intersects_line_segment(&(*q, point_outside_polygon), &line) as usize
                })
                .sum();

            assert_eq!(n_intersections % 2 == 0, true);
        }

        // These are sample values taken from a random simulation
        let new_polygon = [
            nalgebra::Vector2::from([89.8169131069576, 105.21635977300497]),
            nalgebra::Vector2::from([88.08135232199689, 107.60515425930363]),
            nalgebra::Vector2::from([85.27315598238903, 106.69271595767589]),
            nalgebra::Vector2::from([85.27315598238903, 103.74000358833405]),
            nalgebra::Vector2::from([88.08135232199689, 102.8275652867063]),
        ];
        let new_point_outside_polygon = nalgebra::Vector2::from([80.0, 90.0]);

        let points_inside_2 = [nalgebra::Vector2::from([
            88.08135232199689,
            102.8275652867063,
        ])];

        for q in points_inside_2.iter() {
            let n_intersections: usize = new_polygon
                .clone()
                .into_iter()
                .circular_tuple_windows()
                .map(|line| {
                    super::ray_intersects_line_segment(&(*q, new_point_outside_polygon), &line)
                        as usize
                })
                .sum();

            assert_eq!(n_intersections % 2 == 0, false);
        }
    }
}
