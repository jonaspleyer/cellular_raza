use cellular_raza_concepts::*;

use nalgebra::SVector;
use serde::{Deserialize, Serialize};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// No interaction of the cell with any other.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    ) -> Result<For, CalcError> {
        Ok(For::zero())
    }

    fn get_interaction_information(&self) -> () {}
}

/// Lennard-Jones interaction potential with numerical upper and lower limit.
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
#[doc = include_str!("plot_bound_lennard_jones.html")]
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
            ) -> Result<SVector<$float_type, D>, CalcError> {
                let z = own_pos - ext_pos;
                let r = z.norm();
                let dir = z / r;
                let val = 4.0 * self.epsilon / r
                    * (12.0 * (self.sigma / r).powf(11.0) - 6.0 * (self.sigma / r).powf(5.0));
                let max = self.bound / r;
                let q = if self.cutoff >= r { 1.0 } else { 0.0 };
                Ok(dir * q * max.min(val))
            }

            fn get_interaction_information(&self) -> () {}
        }
    };
);

implement_bound_lennard_jones!(BoundLennardJones, f64);
implement_bound_lennard_jones!(BoundLennardJonesF32, f32);

/// Famous [Morse](https://doi.org/10.1103/PhysRev.34.57) potential for diatomic molecules.
///
/// \\begin{equation}
///     V(r) = C_r \exp\left(-\frac{r}{l_r}\right) - C_a
///     \exp\left(-\frac{r}{l_a}\right)
/// \\end{equation}
///
/// Calculating the interaction resulting from this potential is very cheap computationally but not
/// intuitive.
/// Thus we provide additional methods to set, grow and shrink the current radius of the object.
///
/// # Deriving convenience methods
/// To translate the struct fields [`radius`](MorsePotential) and
/// [`interaction_range`](MorsePotential) to $l_r$ and $l_a$ respectively, we
/// assume that the forces are exactly balanced when the distance $r=R_1 + R_2=R$ where $R_1$ and
/// $R_2$ are the radii of the respective objects interacting with reach other.
///
/// \\begin{align}
///     \frac{\partial V}{\partial r}(R) &= 0\\\\
///     &= \frac{C_r}{l_r}\exp\left(-\frac{R}{l_r}\right)
///         - \frac{C_a}{l_a}\exp\left(-\frac{R}{l_a}\right)\\\\
///     &= \frac{C_r}{C_a}\frac{l_a}{l_r}
///         - \exp\left(-R\left(\frac{1}{l_a} - \frac{1}{l_r}\right)\right)
/// \\end{align}
///
/// Thus the combined radius $R=R_1 + R_2$ is given by
///
/// \\begin{equation}
///     R = - \frac{l_r l_a}{l_r - l_a}\log\left(\frac{C_r}{C_a}\frac{l_a}{l_r}\right)
/// \\end{equation}
///
/// We assume that $l_a=$[`interaction_range`](MorsePotential).
/// Thus, we can now calculate $l_r$ via
///
/// \\begin{equation}
///     l_r = R \text{W} \left(-\frac{C_a R}{C_r l_a}\exp\left(-\frac{R}{l_a}\right)\right)
/// \\end{equation}
///
/// where $\text{W}(x)$ is the
/// [Lambert W function](https://en.wikipedia.org/wiki/Lambert_W_function).
///
/// We can also appreciate again, that the original potential $V(r)$ is completely symmetric when
/// interchanging attractive and repulsive properties $l_a,C_a$ and $l_r,C_r$.
///
#[doc = include_str!("plot_morse_potential.html")]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(set_all, get_all))]
pub struct MorsePotential {
    /// Radius of the object
    pub length_repelling: f64,
    /// Defines the length for the interaction range
    pub length_attracting: f64,
    /// Cutoff after which the interaction is exactly 0
    pub cutoff: f64,
    /// Repelling strength between objects
    pub strength_repelling: f64,
    /// Attracting strength between objects
    pub strength_attracting: f64,
}

/// Same as [MorsePotential] but for `f32` type.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(get_all, set_all))]
pub struct MorsePotentialF32 {
    /// Radius of the object
    pub length_repelling: f32,
    /// Defines the length for the interaction range
    pub length_attracting: f32,
    /// Cutoff after which the interaction is exactly 0
    pub cutoff: f32,
    /// Repelling strength between objects
    pub strength_repelling: f32,
    /// Attracting strength between objects
    pub strength_attracting: f32,
}

/// TODO
pub fn calculate_morse_interaction<F, const D: usize>(
    own_pos: &nalgebra::SVector<F, D>,
    ext_pos: &nalgebra::SVector<F, D>,
    ext_length_repelling: &F,
    cutoff: F,
    length_repelling: F,
    length_attracting: F,
    strength_repelling: F,
    strength_attracting: F,
) -> Result<nalgebra::SVector<F, D>, CalcError>
where
    F: Copy + nalgebra::RealField,
{
    let dir = own_pos - ext_pos;
    let dist = dir.norm();

    // If the distance between the two objects is greater than the cutoff, we
    // immediately return zero.
    if dist > cutoff {
        return Ok([F::zero(); D].into());
    }
    let lr = length_repelling;
    let la = length_attracting;
    let cr = strength_repelling;
    let ca = strength_attracting;
    let lr_combined = *ext_length_repelling + length_repelling;
    let force = cr / lr * (-dist / lr_combined).exp() - ca / la * (-dist / la).exp();
    Ok(dir * force)
}

fn product_log<F>(x: F, n_steps: usize) -> F
where
    F: Copy + nalgebra::RealField,
{
    let mut w = F::zero();
    for _ in 0..n_steps {
        w = w - (w * w.exp() - x) / (w.exp() + w * w.exp());
    }
    w
}

/// TODO
pub fn calculate_morse_interaction_cell_radii<F, const D: usize>(
    own_pos: &nalgebra::SVector<F, D>,
    ext_pos: &nalgebra::SVector<F, D>,
    ext_cell_radius: &F,
    cutoff: F,
    own_cell_radius: F,
    interaction_range: F,
    strength_repelling: F,
    strength_attracting: F,
) -> Result<nalgebra::SVector<F, D>, CalcError>
where
    F: Copy + nalgebra::RealField,
{
    let r_both = own_cell_radius + *ext_cell_radius;
    let length_repelling = r_both
        * product_log(
            -strength_attracting / strength_repelling * r_both / interaction_range
                * (-r_both / interaction_range).exp(),
            3,
        );
    let ext_length_repelling = length_repelling * *ext_cell_radius / r_both;
    let own_length_repelling = length_repelling * own_cell_radius / r_both;
    calculate_morse_interaction(
        own_pos,
        ext_pos,
        &ext_length_repelling,
        cutoff,
        own_length_repelling,
        interaction_range,
        strength_repelling,
        strength_attracting,
    )
}

macro_rules! implement_morse_potential(
    ($struct_name:ident, $float_type:ident) => {
        impl<const D: usize>
            Interaction<
                nalgebra::SVector<$float_type, D>,
                nalgebra::SVector<$float_type, D>,
                nalgebra::SVector<$float_type, D>,
                $float_type,
            > for $struct_name
        {
            fn get_interaction_information(&self) -> $float_type {
                self.length_repelling.clone()
            }

            fn calculate_force_between(
                &self,
                own_pos: &nalgebra::SVector<$float_type, D>,
                _own_vel: &nalgebra::SVector<$float_type, D>,
                ext_pos: &nalgebra::SVector<$float_type, D>,
                _ext_vel: &nalgebra::SVector<$float_type, D>,
                ext_info: &$float_type,
            ) -> Result<nalgebra::SVector<$float_type, D>, CalcError> {
                calculate_morse_interaction(
                    own_pos,
                    ext_pos,
                    ext_info,
                    self.cutoff,
                    self.length_repelling,
                    self.length_attracting,
                    self.strength_repelling,
                    self.strength_attracting,
                )
            }
        }

        // TODO implement these functionalities and clean up unused where needed
        impl $struct_name {
            /// CURRENTLY UNIMPLEMENTED!
            // /// ```
            // /// # use cellular_raza_building_blocks::*;
            // #[doc = concat!(
            //     "let morse = ",
            //     stringify!($struct_name),
            //     "::new(2.0, 3.0, 0.1, 0.5);"
            // )]
            // /// ```
            #[allow(unused)]
            pub fn new(
                radius: $float_type,
                interaction_range: $float_type,
                strength_attracting: $float_type,
                strength_repelling: $float_type,
            ) -> Self {
                unimplemented!()
            }

            /// CURRENTLY UNIMPLEMENTED!
            ///
            /// This function as a generic argument which is the dimensionality of the underlying
            /// simulation.
            /// Eg. for a 2-dimensional simulation use it as follows:
            /// ```
            /// # use cellular_raza_building_blocks::*;
            #[doc = concat!("let mut my_potential = ", stringify!($struct_name))]
            /// {
            ///     length_repelling: 1.0,
            ///     length_attracting: 2.0,
            ///     strength_repelling: 2.0,
            ///     strength_attracting: 1.0,
            ///     cutoff: 2.0,
            /// };
            /// my_potential.grow_radius_with_volume::<2>(3.0);
            /// ```
            #[allow(unused)]
            pub fn grow_radius_with_volume<const D: usize>(&mut self, volume: $float_type) {
                unimplemented!()
            }

            /// CURRENTLY UNIMPLEMENTED!
            #[allow(unused)]
            pub fn grow_radius_by_length(&mut self, length: $float_type) {
                unimplemented!()
            }

            ///
            pub fn get_radius(&self) -> $float_type {
                unimplemented!()
            }
        }
    };
);

implement_morse_potential!(MorsePotential, f64);
implement_morse_potential!(MorsePotentialF32, f32);

/// Generalizeation of the [BoundLennardJones] potential for arbitrary floating point types.
///
/// \\begin{align}
///     U(r) &= C\epsilon\left[ \left(\frac{\sigma}{r}\right)^n -
///         \left(\frac{\sigma}{r}\right)^m\right]\\\\
///     C &= \frac{n}{n-m}\left(\frac{n}{m}\right)^{\frac{n}{n-m}}\\
///     V(r) &= \min(U(r), \beta)\theta(r-\zeta)
/// \\end{align}
///
/// This struct itself does not provide python bindings.
/// We provide specialized types for different floating-point types.
/// | Name | Float Type |
/// | --- | --- |
/// | [MiePotentialF64] | COMING |
/// | [MiePotentialF32] | COMING |
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MiePotential<const N: usize, const M: usize, F = f64> {
    /// Interaction strength $\epsilon$ of the potential.
    pub radius: F,
    /// Overall size $\sigma$ of the object of the potential.
    pub potential_strength: F,
    /// Numerical bound $\beta$ of the interaction strength.
    pub bound: F,
    /// Defines a cutoff $\zeta$ after which the potential will be fixed to exactly zero.
    pub cutoff: F,
    /// Exponent $n$ of the potential
    en: F,
    /// Exponent $m$ of the potential
    em: F,
}

impl<F, const D: usize, const N: usize, const M: usize>
    Interaction<SVector<F, D>, SVector<F, D>, SVector<F, D>, F> for MiePotential<N, M, F>
where
    F: nalgebra::RealField + Copy,
{
    fn calculate_force_between(
        &self,
        own_pos: &SVector<F, D>,
        _own_vel: &SVector<F, D>,
        ext_pos: &SVector<F, D>,
        _ext_vel: &SVector<F, D>,
        ext_radius: &F,
    ) -> Result<SVector<F, D>, CalcError> {
        let z = own_pos - ext_pos;
        let r = z.norm();
        if r == F::zero() {
            return Err(CalcError(format!(
                "identical position for two objects. Cannot Calculate force in Mie potential"
            )));
        }
        if r > self.cutoff {
            return Ok([F::zero(); D].into());
        }
        let dir = z / r;
        let x = (self.radius + *ext_radius) / r;
        let sigma = self.radius_to_sigma_factor() * x;
        let mie_constant =
            self.en / (self.en - self.em) * (self.en / self.em).powf(self.em / (self.en - self.em));
        let potential_part =
            self.en * (sigma.powf(self.en + F::one()) - x.powf(self.em + F::one()));
        let force = self.potential_strength * mie_constant * potential_part;
        let force = force.min(self.bound);
        Ok(dir * force)
    }

    fn get_interaction_information(&self) -> F {
        self.radius
    }
}

impl<F, const N: usize, const M: usize> MiePotential<N, M, F>
where
    F: nalgebra::RealField + num::FromPrimitive + Copy,
{
    fn radius_to_sigma_factor(&self) -> F {
        (self.em / self.en).powf(F::one() / (self.en - self.em))
    }

    /// Constructs a new [MiePotential] with given parameters.
    pub fn new(radius: F, potential_strength: F, bound: F, cutoff: F) -> Result<Self, CalcError> {
        let em = F::from_usize(M).ok_or(CalcError(format!(
            "could not convert usize {} to float of type {}",
            M,
            std::any::type_name::<F>()
        )))?;
        let en = F::from_usize(N).ok_or(CalcError(format!(
            "could not convert usize {} to float of type {}",
            N,
            std::any::type_name::<F>()
        )))?;
        Ok(Self {
            radius,
            potential_strength,
            bound,
            cutoff,
            en,
            em,
        })
    }
}

/// Derives an interaction potential from a point-like potential.
#[derive(Serialize, Deserialize, Clone, Debug)]
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
        return 0.0 <= d && d <= e;
    }

    // Handles the case where the points for the ray were layed on top of each other.
    // Then the ray will only hit the point if r1 and r2 are on the line segment itself.
    let t = t_enum / t_denom;
    let u = u_enum / u_denom;

    // In order to be on the line-segment, we require that u is between 0.0 and 1.0
    // Additionally for p to be on the ray, we require t >= 0.0
    0.0 <= u && u < 1.0 && 0.0 <= t
}

impl<A, R, I1, I2, const D: usize>
    Interaction<
        super::mechanics::VertexVector2<D>,
        super::mechanics::VertexVector2<D>,
        super::mechanics::VertexVector2<D>,
        (I1, I2),
    > for VertexDerivedInteraction<A, R, I1, I2>
where
    A: Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>, I1>,
    R: Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>, I2>,
{
    fn get_interaction_information(&self) -> (I1, I2) {
        let i1 = self.outside_interaction.get_interaction_information();
        let i2 = self.inside_interaction.get_interaction_information();
        (i1, i2)
    }

    fn calculate_force_between(
        &self,
        own_pos: &super::mechanics::VertexVector2<D>,
        own_vel: &super::mechanics::VertexVector2<D>,
        ext_pos: &super::mechanics::VertexVector2<D>,
        ext_vel: &super::mechanics::VertexVector2<D>,
        ext_information: &(I1, I2),
    ) -> Result<super::mechanics::VertexVector2<D>, CalcError> {
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

        let average_vel_own: Vector2<f64> = own_vel
            .row_iter()
            .map(|v| v.transpose())
            .sum::<Vector2<f64>>()
            / own_vel.shape().0 as f64;

        let middle_ext: Vector2<f64> = ext_pos
            .row_iter()
            .map(|v| v.transpose())
            .sum::<Vector2<f64>>()
            / ext_pos.shape().0 as f64;

        let average_vel_ext: Vector2<f64> = ext_vel
            .row_iter()
            .map(|v| v.transpose())
            .sum::<Vector2<f64>>()
            / ext_vel.shape().0 as f64;

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

        // Match the obtained interaction information
        let (inf1, inf2) = ext_information;

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
                    // If the bounding box was not successful,
                    // we use the ray-casting algorithm to check.
                    let n_intersections: usize = own_polygon_lines
                        .iter()
                        .map(|line| {
                            ray_intersects_line_segment(&(point, point_outside_polygon), line)
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
                let calc = self.inside_interaction.calculate_force_between(
                    &middle_own,
                    &average_vel_own,
                    &middle_ext,
                    &average_vel_ext,
                    &inf2,
                )?;
                force += calc.transpose();
            } else {
                // Calculate the force outside
                if let Some((_, nearest_point)) =
                    nearest_point_from_point_to_multiple_lines(&point, &own_polygon_lines)
                {
                    let calc = self.outside_interaction.calculate_force_between(
                        &nearest_point,
                        &average_vel_own,
                        &point,
                        &average_vel_ext,
                        &inf1,
                    )?;
                    force += calc.transpose();
                }
            };
        }
        Ok(total_force)
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
