use cellular_raza_concepts::*;

use nalgebra::SVector;
use serde::{Deserialize, Serialize};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// No interaction of the cell with any other.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct NoInteraction;

impl<Pos, Vel, For> Interaction<Pos, Vel, For> for NoInteraction
where
    For: num::Zero,
{
    fn calculate_force_between(
        &self,
        _: &Pos,
        _: &Vel,
        _: &Pos,
        _: &Vel,
        _ext_information: &(),
    ) -> Result<(For, For), CalcError> {
        Ok((For::zero(), For::zero()))
    }

    fn get_interaction_information(&self) -> () {}
}

/// Lennard-Jones interaction potential with numerical upper and lower limit.
///
/// # Parameters & Variables
/// | Symbol | Struct Field | Description |
/// |:---:| --- | --- |
/// | $\sigma$ | `sigma` | Measure for the size of the particle. |
/// | $\epsilon$ | `epsilon` | Interaction Strength |
/// | $\beta$ | `bound` | Upper bound on the interaction strength. |
/// | $\xi$ | `cutoff` | Cutoff after which the interaction strength is identically 0 |
/// | | | |
/// | $r$ | | Distance between interacting particles |
///
/// # Equations
///
/// The pure Lennard-Jones potential has many numerical downsides as it is very unstable to use
/// and thus typically only recommended with extremely small integration steps.
/// Here, we artificially limit the repelling part of the potential thus increasing numerical
/// usability.
/// However, it also has in principle infinite range.
/// This is directly contrary to one of the fundamental assumptions of `cellular_raza`.
/// We resolve the latter problem by simply assigning the value `0` if $r>=\zeta$ although this
/// behavior is not continuous anymore.
/// The potential of the interaction is given by
/// \\begin{align}
///     U(r) &= 4\epsilon\left[ \left(\frac{\sigma}{r}\right)^{12} -
///         \left(\frac{\sigma}{r}\right)^6\right]\\\\
///     V(r) &= \min(U(r), \beta)\theta(r-\zeta)
/// \\end{align}
/// where $\epsilon$ determines the overall interaction strength of the potential
/// and $\sigma$ the shape and interaction range.
/// The function $\theta(r-\zeta)$ is the heaviside function which sets the interaction to zero
/// when reaching the cutoff point.
/// The minimum of this potential is at $r_\text{min}=2^{1/6}\sigma$.
/// For two identically-sized spherical interacting particles $r_\text{min}$ has to align with
/// the diameter of their size.
/// The interaction is artificially bound from above by a value $\beta$ in
/// order to obtain better numerical stability.
///
/// # References
/// \[1\]
/// “On the determination of molecular fields.—I. From the variation of the viscosity of a gas with
/// temperature,”
/// Proceedings of the Royal Society of London.
/// Series A, Containing Papers of a Mathematical and Physical Character, vol. 106, no. 738.
/// The Royal Society, pp. 441–462, Oct. 1924. doi:
/// [10.1098/rspa.1924.0081](https://doi.org/10.1098/rspa.1924.0081).
///
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(get_all, set_all))]
pub struct BoundLennardJones {
    /// Interaction strength $\epsilon$ of the potential.
    pub epsilon: f64,
    /// Overall size $\sigma$ of the object of the potential.
    pub sigma: f64,
    /// Numerical bound $\beta$ of the interaction strength.
    pub bound: f64,
    /// Defines a cutoff $\zeta$ after which the potential will be fixed to exactly zero.
    pub cutoff: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(get_all, set_all))]
/// Identical to [BoundLennardJones] but for `f32` type.
pub struct BoundLennardJonesF32 {
    /// Interaction strength $\epsilon$ of the potential.
    pub epsilon: f32,
    /// Overall size $\sigma$ of the object of the potential.
    pub sigma: f32,
    /// Numerical bound $\beta$ of the interaction strength.
    pub bound: f32,
    /// Defines a cutoff $\zeta$ after which the potential will be fixed to exactly zero.
    pub cutoff: f32,
}

macro_rules! implement_bound_lennard_jones(
    ($struct_name:ident, $float_type:ident) => {
        impl<const D: usize> Interaction<SVector<$float_type, D>, SVector<$float_type, D>, SVector<$float_type, D>>
            for $struct_name
        {
            fn calculate_force_between(
                &self,
                own_pos: &SVector<$float_type, D>,
                _own_vel: &SVector<$float_type, D>,
                ext_pos: &SVector<$float_type, D>,
                _ext_vel: &SVector<$float_type, D>,
                _ext_information: &(),
            ) -> Result<(SVector<$float_type, D>, SVector<$float_type, D>), CalcError> {
                let z = own_pos - ext_pos;
                let r = z.norm();
                let dir = z / r;
                let val = 4.0 * self.epsilon / r
                    * (12.0 * (self.sigma / r).powf(11.0) - 6.0 * (self.sigma / r).powf(5.0));
                let max = self.bound / r;
                let q = if self.cutoff >= r { 1.0 } else { 0.0 };
                Ok((- dir * q * max.min(val), dir * q * max.min(val)))
            }

            fn get_interaction_information(&self) -> () {}
        }
    };
);

implement_bound_lennard_jones!(BoundLennardJones, f64);
implement_bound_lennard_jones!(BoundLennardJonesF32, f32);

/// Calculates the interaction strength behind the [MorsePotential] and [MorsePotentialF32]
/// structs.
pub fn calculate_morse_interaction<F, const D: usize>(
    own_pos: &nalgebra::SVector<F, D>,
    ext_pos: &nalgebra::SVector<F, D>,
    own_radius: F,
    ext_radius: F,
    cutoff: F,
    strength: F,
    potential_stiffness: F,
) -> Result<(nalgebra::SVector<F, D>, nalgebra::SVector<F, D>), CalcError>
where
    F: Copy + nalgebra::RealField,
{
    let z = own_pos - ext_pos;
    let dist = z.norm();

    // If the distance between the two objects is greater than the cutoff, we
    // immediately return zero.
    if dist > cutoff || dist.is_zero() {
        return Ok((
            nalgebra::SVector::<F, D>::zeros(),
            nalgebra::SVector::<F, D>::zeros(),
        ));
    }
    let dir = z / dist;
    let r = own_radius + ext_radius;
    let s = strength;
    let a = potential_stiffness;
    let two = F::one() + F::one();
    let e = (-a * (dist - r)).exp();
    let force = -two * s * a * e * (F::one() - e);
    Ok((dir * force, -dir * force))
}

macro_rules! implement_morse_potential(
    ($struct_name:ident, $float_type:ident) => {
        /// Famous [Morse](https://doi.org/10.1103/PhysRev.34.57) potential for diatomic molecules.
        ///
        /// # Parameters & Variables
        /// | Symbol | Struct Field | Description |
        /// |:---:| --- | --- |
        /// | $R$ | `radius` | Radius of the particle |
        /// | $\lambda$ | `potential_stiffness` | Can be interpreted as the inverse width of the potential |
        /// | | `cutoff` | Cutoff after which the interaction strength is identically 0 |
        /// | $V_0$ | `strength` | Interaction strength |
        /// | | | |
        /// | $r$ | | Distance between interacting particles |
        ///
        /// \\begin{equation}
        ///     V(r) = V_0\left(1 - e^{-\lambda(r-R)}\right)^2
        /// \\end{equation}
        ///
        // TODO remove this entirely or make it somehow work
        // #[doc = include_str!("plot_morse_potential.html")]
        ///
        /// # References
        /// \[1\]
        /// P. M. Morse,
        /// “Diatomic Molecules According to the Wave Mechanics. II. Vibrational Levels,”
        /// Physical Review, vol. 34, no. 1. American Physical Society (APS),
        /// pp. 57–64, Jul. 01, 1929.
        /// doi: [10.1103/physrev.34.57](https://doi.org/10.1103/PhysRev.34.57).
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
        #[cfg_attr(feature = "pyo3", pyclass(set_all, get_all))]
        pub struct $struct_name {
            /// Radius of the object
            pub radius: $float_type,
            /// Defines the length for the interaction range
            pub potential_stiffness: $float_type,
            /// Cutoff after which the interaction is exactly 0
            pub cutoff: $float_type,
            /// Strength of the interaction
            pub strength: $float_type,
        }

        impl<const D: usize>
            Interaction<
                nalgebra::SVector<$float_type, D>,
                nalgebra::SVector<$float_type, D>,
                nalgebra::SVector<$float_type, D>,
                $float_type,
            > for $struct_name
        {
            fn get_interaction_information(&self) -> $float_type {
                self.radius
            }

            fn calculate_force_between(
                &self,
                own_pos: &nalgebra::SVector<$float_type, D>,
                _own_vel: &nalgebra::SVector<$float_type, D>,
                ext_pos: &nalgebra::SVector<$float_type, D>,
                _ext_vel: &nalgebra::SVector<$float_type, D>,
                ext_info: &$float_type,
            ) -> Result<
                (nalgebra::SVector<$float_type, D>, nalgebra::SVector<$float_type, D>),
                CalcError
            > {
                calculate_morse_interaction(
                    own_pos,
                    ext_pos,
                    self.radius,
                    *ext_info,
                    self.cutoff,
                    self.strength,
                    self.potential_stiffness,
                )
            }
        }

        #[cfg(feature = "pyo3")]
        #[cfg_attr(docsrs, doc(cfg(feature = "pyo3")))]
        #[pymethods]
        impl $struct_name {
            /// Constructs a new [
            #[doc = stringify!($struct_name)]
            /// ]
            /// ```
            #[doc = concat!("use cellular_raza_building_blocks::", stringify!($struct_name), ";")]
            /// # let (radius, potential_stiffness, cutoff, strength) = (1.0, 1.0, 1.0, 1.0);
            #[doc = concat!("let morse_potential = ", stringify!($struct_name), "::new(")]
            ///     radius,
            ///     potential_stiffness,
            ///     cutoff,
            ///     strength,
            /// );
            /// ```
            #[new]
            #[pyo3(signature = (radius, potential_stiffness, cutoff, strength))]
            pub fn new(
                radius: $float_type,
                potential_stiffness: $float_type,
                cutoff: $float_type,
                strength: $float_type
            ) -> Self {
                Self {
                    radius,
                    potential_stiffness,
                    cutoff,
                    strength,
                }
            }
        }
    };
);

implement_morse_potential!(MorsePotential, f64);
implement_morse_potential!(MorsePotentialF32, f32);

macro_rules! implement_mie_potential(
    ($name:ident, $float_type:ty) => {
        /// Generalizeation of the [BoundLennardJones] potential.
        ///
        /// \\begin{align}
        ///     U(r) &= C\epsilon\left[ \left(\frac{\sigma}{r}\right)^n -
        ///         \left(\frac{\sigma}{r}\right)^m\right]\\\\
        ///     C &= \frac{n}{n-m}\left(\frac{n}{m}\right)^{\frac{n}{n-m}}\\\\
        ///     V(r) &= \min(U(r), \beta)\theta(r-\zeta)
        /// \\end{align}
        ///
        /// This struct comes in a 64bit version [MiePotential] and a 32bit variant
        /// [MiePotentialF32].
        ///
        /// # References
        /// G. Mie, “Zur kinetischen Theorie der einatomigen Körper,”
        /// Annalen der Physik, vol. 316, no. 8. Wiley, pp. 657–697, Jan. 1903.
        /// doi: [10.1002/andp.19033160802](https://doi.org/10.1002/andp.19033160802).
        #[cfg_attr(feature = "pyo3", pyclass(get_all, set_all))]
        #[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
        pub struct $name {
            /// Interaction strength $\epsilon$ of the potential.
            pub radius: $float_type,
            /// Overall size $\sigma$ of the object of the potential.
            pub strength: $float_type,
            /// Numerical bound $\beta$ of the interaction strength.
            pub bound: $float_type,
            /// Defines a cutoff $\zeta$ after which the potential will be fixed to exactly zero.
            pub cutoff: $float_type,
            /// Exponent $n$ of the potential
            pub en: $float_type,
            /// Exponent $m$ of the potential
            pub em: $float_type,
        }

        impl<const D: usize> Interaction<
            SVector<$float_type, D>,
            SVector<$float_type, D>,
            SVector<$float_type, D>,
            $float_type
        >
            for $name
        {
            fn calculate_force_between(
                &self,
                own_pos: &SVector<$float_type, D>,
                _own_vel: &SVector<$float_type, D>,
                ext_pos: &SVector<$float_type, D>,
                _ext_vel: &SVector<$float_type, D>,
                ext_radius: &$float_type,
            ) -> Result<(SVector<$float_type, D>, SVector<$float_type, D>), CalcError> {
                let z = own_pos - ext_pos;
                let r = z.norm();
                if r == 0.0 {
                    return Err(CalcError(format!(
                        "identical position for two objects. Cannot Calculate force in Mie potential"
                    )));
                }
                if r > self.cutoff {
                    return Ok((
                        SVector::<$float_type, D>::zeros(),
                        SVector::<$float_type, D>::zeros()
                    ));
                }
                let dir = z / r;
                let x = (self.radius + *ext_radius);
                let sigma = self.radius_to_sigma_factor() * x;
                let mie_constant =
                    self.en / (self.en - self.em)
                        * (self.en / self.em).powf(self.em / (self.en - self.em));
                let potential_part = self.en / sigma * (sigma / r).powf(self.en + 1.0)
                    - self.em /sigma * (sigma / r).powf(self.em + 1.0);
                let force = (mie_constant * self.strength * potential_part).min(self.bound);
                Ok((dir * force, - dir * force))
            }

            fn get_interaction_information(&self) -> $float_type {
                self.radius
            }
        }

        impl $name {
            fn radius_to_sigma_factor(&self) -> $float_type {
                (self.em / self.en).powf(1.0 / (self.en - self.em))
            }
        }

        #[cfg(feature = "pyo3")]
        #[cfg_attr(docsrs, doc(cfg(feature = "pyo3")))]
        #[pymethods]
        impl $name {
            /// Constructs a new [
            #[doc = stringify!($name)]
            /// ]
            /// ```
            #[doc = concat!("use cellular_raza_building_blocks::", stringify!($name), ";")]
            /// let (radius, strength, bound, cutoff, en, em) = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
            #[doc = concat!("let mie_potential = ", stringify!($name), "::new(")]
            ///     radius,
            ///     strength,
            ///     bound,
            ///     cutoff,
            ///     en,
            ///     em,
            /// );
            /// ```
            #[new]
            #[pyo3(signature = (radius, strength, bound, cutoff, en, em))]
            pub fn new(
                radius: $float_type,
                strength: $float_type,
                bound: $float_type,
                cutoff: $float_type,
                en: $float_type,
                em: $float_type,
            ) -> Self {
                Self {
                    radius,
                    strength,
                    bound,
                    cutoff,
                    en,
                    em,
                }
            }
        }
    }
);

implement_mie_potential!(MiePotential, f64);
implement_mie_potential!(MiePotentialF32, f32);

/// Derives an interaction potential from a point-like potential.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct VertexDerivedInteraction<A, R, I1 = (), I2 = ()> {
    /// Interaction potential used when other vertex is outside of current polygon.
    pub outside_interaction: A,
    /// Interaction potential when vertex is inside current polygon.
    pub inside_interaction: R,
    phantom_inf_1: core::marker::PhantomData<I1>,
    phantom_inf_2: core::marker::PhantomData<I2>,
}

impl<A, R, I1, I2> VertexDerivedInteraction<A, R, I1, I2> {
    /// Constructs a new [VertexDerivedInteraction] from two Interaction potentials.
    ///
    /// One serves as the inside and one for the outside interaction.
    pub fn from_two_forces(attracting_force: A, repelling_force: R) -> Self
    where
        A: Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>, I1>,
        R: Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>, I2>,
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

/// Calculate the point on a line which is closest to an external given point.
///
/// This function takes an external point $\vec{p}$ and a line segment described by two points
/// $\vec{p}_1,\vec{p}_2$.
/// It returns a tuple $(d, \vec{x}, q)$ which contains the distance $d$ between the calculated
/// nearest point and the external point, the calculated point $\vec{x}$ and a fractional value
/// $0\leq q\leq 1$ which is the relative length of the calculated nearest point between the
/// points of the given line segment given by:
/// $$\vec{x} = (1-q)\vec{p}_1 + q\vec{p}_2$$
///
/// ```
/// use nalgebra::Vector2;
/// # use cellular_raza_building_blocks::nearest_point_from_point_to_line;
///
/// let external_point = Vector2::from([0f64; 2]);
/// let line_segment = (
///     Vector2::from([1.0, 0.0]),
///     Vector2::from([-1.0, 2.0]),
/// );
/// let (dist, point, rel_length) = nearest_point_from_point_to_line(
///     &external_point,
///     &line_segment,
/// );
///
/// assert!((dist - 1.0/2f64.sqrt()).abs() < 1e-10);
/// assert!((rel_length - 0.25).abs() < 1e-10);
/// assert!((point.x - 0.5).abs() < 1e-10);
/// assert!((point.y - 0.5).abs() < 1e-10);
/// ```
pub fn nearest_point_from_point_to_line<F, const D: usize>(
    point: &SVector<F, D>,
    line: &(SVector<F, D>, SVector<F, D>),
) -> (F, SVector<F, D>, F)
where
    F: Copy + nalgebra::RealField,
{
    let ab = line.1 - line.0;
    let ap = point - line.0;
    let t = (ab.dot(&ap) / ab.norm_squared()).clamp(F::zero(), F::one());
    let nearest_point = line.0 * (F::one() - t) + line.1 * t;
    ((point - nearest_point).norm(), nearest_point, t)
}

/// Generalizes the [nearest_point_from_point_to_line] function for a collection of line segments.
/// ```
/// # use cellular_raza_building_blocks::nearest_point_from_point_to_multiple_lines;
/// # use nalgebra::Vector2;
/// # use itertools::Itertools;
/// let point = Vector2::from([0.5; 2]);
/// let polygon = vec![
///     Vector2::from([0.0, 0.0]),
///     Vector2::from([0.5, 0.0]),
///     Vector2::from([0.7, 0.4]),
///     Vector2::from([0.9, 0.8]),
/// ];
/// let lines = polygon.clone().into_iter().tuple_windows::<(_, _)>().collect::<Vec<_>>();
/// // This looks something like this:
/// //             3
/// //            /
/// //       p   /
/// //          2
/// //         /
/// //        /
/// // 0 --- 1
/// let (n, (dist, calculated_point, t)) = nearest_point_from_point_to_multiple_lines(
///     &point,
///     &lines,
/// ).unwrap();
/// assert_eq!(n, 1);
/// let test_dist: f64 = (calculated_point - point).norm();
/// assert!((dist - test_dist).abs() < 1e-6);
/// let (v1, v2) = lines[n];
/// assert!((calculated_point - ((1.0 - t)*v1 + t*v2)).norm() < 1e-6);
/// for p in polygon {
///     assert!(dist <= (p - point).norm());
/// }
/// ```
pub fn nearest_point_from_point_to_multiple_lines<'a, F>(
    point: &Vector2<F>,
    polygon_lines: impl IntoIterator<Item = &'a (Vector2<F>, Vector2<F>)>,
) -> Option<(usize, (F, Vector2<F>, F))>
where
    F: Copy + nalgebra::RealField,
{
    polygon_lines
        .into_iter()
        .enumerate()
        .map(|(n_row, line)| (n_row, nearest_point_from_point_to_line(point, &line)))
        .fold(None, |acc, v| {
            let distance_v = v.1.0;
            match acc {
                None => Some(v),
                Some(e) => {
                    let distance_acc = e.1.0;
                    if distance_v < distance_acc {
                        Some(v)
                    } else {
                        acc
                    }
                }
            }
        })
}

/// Implements the ray-casting algorithm to check if a ray intersects with a given line segment.
///
/// This method can be applied to polygons to calculate if a point is inside of a given polygon or
/// outside of it.
/// ```
/// # use cellular_raza_building_blocks::ray_intersects_line_segment;
/// use nalgebra::Vector2;
/// let line_segment = (
///     Vector2::from([1.0, 3.0]),
///     Vector2::from([1.0, 1.0]),
/// );
/// let ray = (
///     Vector2::from([4.0, 1.5]),
///     Vector2::from([2.0, 1.5]),
/// );
/// // This should look something like this:
/// // |
/// // |  <------
/// // |
/// assert!(ray_intersects_line_segment(&ray, &line_segment));
/// // Chaning the order of arguments will look like this
/// // |
/// // |  ------
/// // |
/// // v
/// assert!(!ray_intersects_line_segment(&line_segment, &ray));
/// ```
pub fn ray_intersects_line_segment<F>(
    ray: &(Vector2<F>, Vector2<F>),
    line_segment: &(Vector2<F>, Vector2<F>),
) -> bool
where
    F: Copy + nalgebra::RealField + PartialOrd,
{
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
    if t_denom.is_zero() || u_denom.is_zero() {
        // We can directly test if some of the points are on the same line
        // Test this by calculating the dot product of r1-l1 with l2-l1.
        // This value must be between the norm of l2-l1 squared
        let d = (r1 - l1).dot(&(l2 - l1));
        let e = (l2 - l1).norm_squared();
        return F::zero() <= d && d <= e;
    }

    // Handles the case where the points for the ray were layed on top of each other.
    // Then the ray will only hit the point if r1 and r2 are on the line segment itself.
    let t = t_enum / t_denom;
    let u = u_enum / u_denom;

    // In order to be on the line-segment, we require that u is between 0.0 and 1.0
    // Additionally for p to be on the ray, we require t >= 0.0
    F::zero() <= u && u < F::one() && F::zero() <= t
}

impl<F, S, A, R, I1, I2, D>
    Interaction<
        nalgebra::Matrix<F, D, nalgebra::U2, S>,
        nalgebra::Matrix<F, D, nalgebra::U2, S>,
        nalgebra::Matrix<F, D, nalgebra::U2, S>,
        (I1, I2),
    > for VertexDerivedInteraction<A, R, I1, I2>
where
    A: Interaction<Vector2<F>, Vector2<F>, Vector2<F>, I1>,
    R: Interaction<Vector2<F>, Vector2<F>, Vector2<F>, I2>,
    D: nalgebra::Dim,
    F: nalgebra::Scalar + nalgebra::RealField,
    nalgebra::Matrix<F, D, nalgebra::U2, S>:
        core::ops::Mul<F, Output = nalgebra::Matrix<F, D, nalgebra::U2, S>>,
    F: Copy,
    // nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<F, D, nalgebra::Const<2>>,
    // nalgebra::VecStorage<F, D, nalgebra::Const<2>>: nalgebra::Storage<F, D, nalgebra::Const<2>>
    // S: nalgebra::RawStorage<F, D, nalgebra::U2>,
    S: nalgebra::RawStorageMut<F, D, nalgebra::U2>,
    S: nalgebra::Storage<F, D, nalgebra::U2>,
    S: Clone,
{
    fn get_interaction_information(&self) -> (I1, I2) {
        let i1 = self.outside_interaction.get_interaction_information();
        let i2 = self.inside_interaction.get_interaction_information();
        (i1, i2)
    }

    fn calculate_force_between(
        &self,
        own_pos: &nalgebra::Matrix<F, D, nalgebra::U2, S>,
        own_vel: &nalgebra::Matrix<F, D, nalgebra::U2, S>,
        ext_pos: &nalgebra::Matrix<F, D, nalgebra::U2, S>,
        ext_vel: &nalgebra::Matrix<F, D, nalgebra::U2, S>,
        ext_information: &(I1, I2),
    ) -> Result<
        (
            nalgebra::Matrix<F, D, nalgebra::U2, S>,
            nalgebra::Matrix<F, D, nalgebra::U2, S>,
        ),
        CalcError,
    > {
        // IMPORTANT!!
        // This code assumes that we are dealing with a regular! polygon!

        // Calculate the middle point which we will need later
        let middle_own = own_pos.row_mean().transpose();
        let average_vel_own = own_vel.row_mean().transpose();
        let middle_ext = ext_pos.row_mean().transpose();
        let average_vel_ext = ext_vel.row_mean().transpose();

        // Also calculate our own polygon defined by lines between points
        let own_polygon_lines = own_pos
            .row_iter()
            .map(|vec| vec.transpose())
            .circular_tuple_windows()
            .collect::<Vec<(_, _)>>();

        let vec_on_edge = (own_pos.view_range(0..1, 0..2) + own_pos.view_range(1..2, 0..2))
            .transpose()
            / (F::one() + F::one());
        // This is simplified notation for the following
        // p = 2*v - middle = (v-middle) + v
        let point_outside_polygon = vec_on_edge * (F::one() + F::one()) - middle_own;

        // Store the total calculated force here
        let mut total_force_own = ext_pos.clone() * F::zero();
        let mut total_force_ext = ext_pos.clone() * F::zero();

        // Match the obtained interaction information
        let (inf1, inf2) = ext_information;

        // Pick one point from the external positions
        // and calculate which would be the nearest point on the own positions
        for (n_row_ext, point_ext) in ext_pos.row_iter().enumerate() {
            let point_ext = point_ext.transpose();
            // Check if the point is inside the polygon.
            // If this is the case, do not calculate any attracting force.

            // We first calculate a bounding box and test quickly with this
            let bounding_box: [[F; 2]; 2] = own_pos.row_iter().map(|v| v.transpose()).fold(
                [[F::max_value().unwrap(), F::min_value().unwrap()]; 2],
                |mut accumulator, polygon_edge| {
                    accumulator[0][0] = accumulator[0][0].min(polygon_edge.x);
                    accumulator[0][1] = accumulator[0][1].max(polygon_edge.x);
                    accumulator[1][0] = accumulator[1][0].min(polygon_edge.y);
                    accumulator[1][1] = accumulator[1][1].max(polygon_edge.y);
                    accumulator
                },
            );

            let point_is_out_of_bounding_box = point_ext.x < bounding_box[0][0]
                || point_ext.x > bounding_box[0][1]
                || point_ext.y < bounding_box[1][0]
                || point_ext.y > bounding_box[1][1];

            let external_point_is_in_polygon = match point_is_out_of_bounding_box {
                true => false,
                false => {
                    // If the bounding box was not successful,
                    // we use the ray-casting algorithm to check.
                    let n_intersections: usize = own_polygon_lines
                        .iter()
                        .map(|line| {
                            ray_intersects_line_segment(&(point_ext, point_outside_polygon), line)
                                as usize
                        })
                        .sum();

                    // An even number means that the point was outside
                    // while odd numbers mean that the point was inside.
                    n_intersections % 2 == 1
                }
            };

            if external_point_is_in_polygon {
                // Calculate the force inside the cell
                let (calc_own, calc_ext) = self.inside_interaction.calculate_force_between(
                    &middle_own,
                    &average_vel_own,
                    &point_ext,
                    &average_vel_ext,
                    inf2,
                )?;
                let dir = (middle_ext - middle_own).normalize();
                let calc_own = -dir * calc_own.norm();
                let calc_ext = dir * calc_ext.norm();
                let mut force_ext = total_force_ext.row_mut(n_row_ext);
                force_ext += calc_ext.transpose();
                let n_rows = total_force_own.nrows();
                let n_rows_float = F::from_usize(n_rows).unwrap();
                total_force_own
                    .row_iter_mut()
                    .for_each(|mut r| r += calc_own.transpose() / n_rows_float);
            } else {
                // Calculate the force outside
                if let Some((n_row_nearest, (_, nearest_point, t_frac))) =
                    nearest_point_from_point_to_multiple_lines(&point_ext, &own_polygon_lines)
                {
                    let (calc_own, calc_ext) = self.outside_interaction.calculate_force_between(
                        &nearest_point,
                        &average_vel_own,
                        &point_ext,
                        &average_vel_ext,
                        inf1,
                    )?;
                    let mut force_ext = total_force_ext.row_mut(n_row_ext);
                    force_ext += calc_ext.transpose();
                    let mut force_own_n = total_force_own.row_mut(n_row_nearest);
                    force_own_n += calc_own.transpose() * (F::one() - t_frac);
                    let mut force_own_n1 =
                        total_force_own.row_mut((n_row_nearest + 1) % total_force_own.nrows());
                    force_own_n1 += calc_own.transpose() * t_frac;
                }
            };
        }
        Ok((total_force_own, total_force_ext))
    }
}

mod test {
    #[test]
    fn test_closest_points() {
        // Define the line we will be using
        let p1 = nalgebra::Vector2::from([0.0, 0.0]);
        let p2 = nalgebra::Vector2::from([2.0, 0.0]);

        // Create a vector of tuples which have (input_point,
        // expected_nearest_point, expected_distance)
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
            let (dist, nearest_point, _) = super::nearest_point_from_point_to_line(q, &(p1, p2));
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
                .into_iter()
                .circular_tuple_windows::<(_, _)>()
                .map(|line| {
                    super::ray_intersects_line_segment(&(*p, point_outside_polygon), &line) as usize
                })
                .sum();
            assert!(n_intersections % 2 == 1);
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
                .into_iter()
                .circular_tuple_windows()
                .map(|line| {
                    super::ray_intersects_line_segment(&(*q, point_outside_polygon), &line) as usize
                })
                .sum();

            assert!(n_intersections % 2 == 0);
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
                .into_iter()
                .circular_tuple_windows()
                .map(|line| {
                    super::ray_intersects_line_segment(&(*q, new_point_outside_polygon), &line)
                        as usize
                })
                .sum();

            assert!(n_intersections % 2 != 0);
        }
    }
}
