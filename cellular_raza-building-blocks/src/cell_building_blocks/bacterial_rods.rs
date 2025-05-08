use crate::{CartesianCuboid, CartesianSubDomain};
use cellular_raza_concepts::*;

#[cfg(feature = "approx")]
use approx::AbsDiffEq;
use num::FromPrimitive;
use serde::{Deserialize, Serialize};

use nalgebra::{Const, Dyn, Matrix, SVector, VecStorage};

/// A mechanical model for Bacterial Rods
///
/// See the [Bacterial Rods](https://cellular-raza.com/showcase/bacterial-rods) example for a
/// detailed example.
///
/// # Parameters & Variables
///
/// | Symbol | Struct Field | Description |
/// | --- | --- | --- |
/// | $\gamma$ | `spring_tension` | Tension of the springs connecting the vertices. |
/// | $D$ | `diffusion_constant` | Diffusion constant corresponding to brownian motion. |
/// | $\lambda$ | `damping` | Damping constant. |
/// | $l$ | `spring_length` | Length of an individual segment between two vertices. |
/// | $\eta$ | `rigidity` | Rigidity with respect to bending the rod. |
///
/// # Equations
///
/// The vertices which are being modeled are stored in the `pos` struct field and their
/// corresponding velocities in the `vel` field.
///
/// \\begin{equation}
///     \vec{v}_i= \text{\texttt{rod\\_mechanics\.pos\.row(i)}}
/// \\end{equation}
///
/// We define the edge $\vec{c}\_i:=\vec{v}\_i-\vec{v}\_{i-1}$.
/// The first force acts between the vertices $v\_i$ of the model and aims to maintain an equal
/// distance between all vertices via
///
/// \\begin{equation}
///     \vec{F}\_{i,\text{springs}} = -\gamma\left(1-\frac{l}{||\vec{c}\_i||}\right)\vec{c}\_i
///         +\gamma\left(1-\frac{l}{||\vec{c}\_{i+1}||}\right)\vec{c}\_{i+1}.
/// \\end{equation}
///
/// We assume the properties of a simple elastic rod.
/// With the angle $\alpha_i$ between adjacent edges $\vec{c}\_{i-1},\vec{c}\_i$ we can formulate
/// the bending force which is proportional to the curvature $\kappa\_i$ at vertex $i$
///
/// \\begin{equation}
///     \kappa\_i = 2\tan\left(\frac{\alpha\_i}{2}\right).
/// \\end{equation}
///
/// The resulting force acts along the angle bisector which can be calculated from the edge
/// vectors.
/// The forces acting on vertices $\vec{v}\_i,\vec{v}\_{i-1},\vec{v}\_{i+1}$ are given by
///
/// \\begin{align}
///     \vec{F}\_{i,\text{curvature}} &= \eta\kappa_i
///         \frac{\vec{c}\_i - \vec{c}\_{i+1}}{|\vec{c}\_i-\vec{c}\_{i+1}|}\\\\
///     \vec{F}\_{i-1,\text{curvature}} &= -\frac{1}{2}\vec{F}\_{i,\text{curvature}}\\\\
///     \vec{F}\_{i+1,\text{curvature}} &= -\frac{1}{2}\vec{F}\_{i,\text{curvature}}
/// \\end{align}
///
/// where $\eta\_i$ is the angle curvature at vertex $\vec{v}\_i$.
/// The total force $\vec{F}_{i,\text{total}}$ at vertex $i$ consists of multiple contributions.
///
/// \\begin{equation}
///     \vec{F}\_{i,\text{total}} = \vec{F}\_{i,\text{springs}} + \vec{F}\_{i,\text{curvature}}
///         + \vec{F}\_{i,\text{external}}
/// \\end{equation}
///
/// # References
///
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "
F: 'static
    + PartialEq
    + Clone
    + core::fmt::Debug
    + Serialize
    + for<'a> Deserialize<'a>,
")]
pub struct RodMechanics<F, const D: usize> {
    /// The current position
    pub pos: Matrix<
        F,
        nalgebra::Dyn,
        nalgebra::Const<D>,
        nalgebra::VecStorage<F, nalgebra::Dyn, nalgebra::Const<D>>,
    >,
    /// The current velocity
    pub vel: Matrix<
        F,
        nalgebra::Dyn,
        nalgebra::Const<D>,
        nalgebra::VecStorage<F, nalgebra::Dyn, nalgebra::Const<D>>,
    >,
    /// Controls magnitude of stochastic motion
    pub diffusion_constant: F,
    /// Spring tension between individual vertices
    pub spring_tension: F,
    /// Stiffness at each joint connecting two edges
    pub rigidity: F,
    /// Target spring length
    pub spring_length: F,
    /// Daming constant
    pub damping: F,
}

#[cfg(feature = "approx")]
impl<F, const D: usize> AbsDiffEq for RodMechanics<F, D>
where
    F: AbsDiffEq + nalgebra::Scalar,
    F::Epsilon: Clone,
{
    type Epsilon = F::Epsilon;

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let RodMechanics {
            pos,
            vel,
            diffusion_constant,
            spring_tension,
            rigidity,
            spring_length,
            damping,
        } = &self;
        pos.iter()
            .zip(other.pos.iter())
            .all(|(x, y)| x.abs_diff_eq(y, epsilon.clone()))
            && vel
                .iter()
                .zip(other.vel.iter())
                .all(|(x, y)| x.abs_diff_eq(y, epsilon.clone()))
            && diffusion_constant.abs_diff_eq(&other.diffusion_constant, epsilon.clone())
            && spring_tension.abs_diff_eq(&other.spring_tension, epsilon.clone())
            && rigidity.abs_diff_eq(&other.rigidity, epsilon.clone())
            && spring_length.abs_diff_eq(&other.spring_length, epsilon.clone())
            && damping.abs_diff_eq(&other.damping, epsilon)
    }

    fn default_epsilon() -> Self::Epsilon {
        F::default_epsilon()
    }
}

impl<F, const D: usize>
    Mechanics<
        Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        F,
    > for RodMechanics<F, D>
where
    F: nalgebra::RealField + Clone + num::Float,
    rand_distr::StandardNormal: rand_distr::Distribution<F>,
{
    fn calculate_increment(
        &self,
        force: Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
    ) -> Result<
        (
            Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
            Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        ),
        CalcError,
    > {
        use core::ops::AddAssign;
        let one_half = F::one() / (F::one() + F::one());
        let two = F::one() + F::one();

        // Calculate internal force between the two points of the Agent
        let mut total_force = force;

        // Calculate force exerted by spring action between individual vertices
        let n_rows = self.pos.nrows();
        let dist_internal = self.pos.rows(0, n_rows - 1) - self.pos.rows(1, n_rows - 1);
        dist_internal.row_iter().enumerate().for_each(|(i, dist)| {
            if !dist.norm().is_zero() {
                let dir = dist.normalize();
                let force_internal =
                    -dir * self.spring_tension * (dist.norm() - self.spring_length);
                total_force.row_mut(i).add_assign(force_internal * one_half);
                total_force
                    .row_mut(i + 1)
                    .add_assign(-force_internal * one_half);
            }
        });

        // Calculate force exerted by curvature-contributions
        use itertools::Itertools;
        dist_internal
            .row_iter()
            .tuple_windows::<(_, _)>()
            .enumerate()
            .for_each(|(i, (conn1, conn2))| {
                let angle = conn1.angle(&conn2);
                let force_d = conn1 - conn2;
                let force_dir = if !force_d.norm().is_zero() {
                    force_d.normalize()
                } else {
                    force_d
                };
                let force =
                    force_dir * two * self.rigidity * <F as num::Float>::tan(one_half * angle);
                total_force.row_mut(i).add_assign(-force * one_half);
                total_force.row_mut(i + 1).add_assign(force);
                total_force.row_mut(i + 2).add_assign(-force * one_half);
            });

        // Calculate damping force
        total_force -= &self.vel * self.damping;
        Ok((self.vel.clone(), total_force))
    }

    fn get_random_contribution(
        &self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: F,
    ) -> Result<
        (
            Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
            Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        ),
        RngError,
    > {
        let distr = match rand_distr::Normal::new(F::zero(), <F as num::Float>::sqrt(dt)) {
            Ok(e) => Ok(e),
            Err(e) => Err(cellular_raza_concepts::RngError(format!("{e}"))),
        }?;
        let dpos = nalgebra::Matrix::<F, Dyn, Const<D>, _>::from_distribution(
            self.pos.nrows(),
            &distr,
            rng,
        ) * <F as num::Float>::powi(F::one() + F::one(), -2)
            * self.diffusion_constant
            / dt;
        let dvel = nalgebra::Matrix::<F, Dyn, Const<D>, _>::zeros(self.pos.nrows());

        Ok((dpos, dvel))
    }
}

impl<F: Clone, const D: usize> Position<Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>>
    for RodMechanics<F, D>
{
    fn pos(&self) -> Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>> {
        self.pos.clone()
    }

    fn set_pos(&mut self, position: &Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>) {
        self.pos = position.clone();
    }
}

impl<F: Clone, const D: usize> Velocity<Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>>
    for RodMechanics<F, D>
{
    fn velocity(&self) -> Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>> {
        self.vel.clone()
    }

    fn set_velocity(&mut self, velocity: &Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>) {
        self.vel = velocity.clone();
    }
}

/// Automatically derives a [Interaction] suitable for rods from a point-wise interaction.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "approx", derive(AbsDiffEq))]
pub struct RodInteraction<I>(pub I);

impl<I, F, Inf, const D: usize>
    Interaction<
        Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        Inf,
    > for RodInteraction<I>
where
    I: Interaction<nalgebra::SVector<F, D>, nalgebra::SVector<F, D>, nalgebra::SVector<F, D>, Inf>,
    F: 'static + nalgebra::RealField + Copy + core::fmt::Debug + num::Zero,
{
    fn get_interaction_information(&self) -> Inf {
        self.0.get_interaction_information()
    }

    fn calculate_force_between(
        &self,
        own_pos: &Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        own_vel: &Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        ext_pos: &Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        ext_vel: &Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        ext_inf: &Inf,
    ) -> Result<
        (
            Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
            Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        ),
        CalcError,
    > {
        use core::ops::AddAssign;
        use itertools::Itertools;
        let mut force_own = nalgebra::Matrix::<F, Dyn, Const<D>, _>::zeros(own_vel.nrows());
        let mut force_ext = nalgebra::Matrix::<F, Dyn, Const<D>, _>::zeros(own_vel.nrows());
        for (i, p1) in own_pos.row_iter().enumerate() {
            for (j, (p2_n0, p2_n1)) in ext_pos.row_iter().tuple_windows::<(_, _)>().enumerate() {
                // Calculate the closest point of the external position
                let (_, nearest_point, rel_length) = crate::nearest_point_from_point_to_line(
                    &p1.transpose(),
                    &(p2_n0.transpose(), p2_n1.transpose()),
                );

                let (f_own, f_ext) = self.0.calculate_force_between(
                    &p1.transpose(),
                    &own_vel.row(i).transpose(),
                    &nearest_point,
                    &ext_vel.row(j).transpose(),
                    ext_inf,
                )?;

                force_own.row_mut(i).add_assign(f_own.transpose());
                force_ext
                    .row_mut(j)
                    .add_assign(f_ext.transpose() * (F::one() - rel_length));
                force_ext
                    .row_mut((j + 1) % own_pos.nrows())
                    .add_assign(f_ext.transpose() * rel_length);
            }
        }
        Ok((force_own, force_ext))
    }

    fn is_neighbor(
        &self,
        own_pos: &Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        ext_pos: &Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        ext_inf: &Inf,
    ) -> Result<bool, CalcError> {
        for p in own_pos.row_iter() {
            for q in ext_pos.row_iter() {
                if self
                    .0
                    .is_neighbor(&p.transpose(), &q.transpose(), ext_inf)?
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn react_to_neighbors(&mut self, neighbors: usize) -> Result<(), CalcError> {
        self.0.react_to_neighbors(neighbors)
    }
}

/// Cells are represented by rods
#[derive(Domain)]
pub struct CartesianCuboidRods<F, const D: usize> {
    /// The base-cuboid which is being repurposed
    #[DomainRngSeed]
    pub domain: CartesianCuboid<F, D>,
    /// Gel pressure which is only relevant for 3D simulations and always acts with constant
    /// force downwards (negative z-direction).
    pub gel_pressure: F,
    /// Computes friction at all surfaces of the box
    pub surface_friction: F,
    /// The distance for which the friction will be applied
    pub surface_friction_distance: F,
}

impl<C, F, const D: usize> SortCells<C> for CartesianCuboidRods<F, D>
where
    C: Position<Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>>,
    F: 'static
        + nalgebra::Field
        + Clone
        + core::fmt::Debug
        + num::FromPrimitive
        + num::ToPrimitive
        + num::Float
        + Copy,
{
    type VoxelIndex = [usize; D];

    fn get_voxel_index_of(&self, cell: &C) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.pos().row_sum().transpose() / F::from_usize(cell.pos().nrows()).unwrap();
        let index = self.domain.get_voxel_index_of_raw(&pos)?;
        Ok(index)
    }
}

impl<F, const D: usize> DomainCreateSubDomains<CartesianSubDomainRods<F, D>>
    for CartesianCuboidRods<F, D>
where
    F: 'static + num::Float + core::fmt::Debug + num::FromPrimitive,
{
    type SubDomainIndex = usize;
    type VoxelIndex = [usize; D];

    fn create_subdomains(
        &self,
        n_subdomains: std::num::NonZeroUsize,
    ) -> Result<
        impl IntoIterator<
            Item = (
                Self::SubDomainIndex,
                CartesianSubDomainRods<F, D>,
                Vec<Self::VoxelIndex>,
            ),
        >,
        DecomposeError,
    > {
        let subdomains = self.domain.create_subdomains(n_subdomains)?;
        Ok(subdomains
            .into_iter()
            .map(move |(subdomain_index, subdomain, voxels)| {
                (
                    subdomain_index,
                    CartesianSubDomainRods::<F, D> {
                        subdomain,
                        gel_pressure: self.gel_pressure,
                        surface_friction: self.surface_friction,
                        surface_friction_distance: self.surface_friction,
                    },
                    voxels,
                )
            }))
    }
}

/// The corresponding SubDomain of the [CartesianCuboidRods] domain.
#[derive(Clone, SubDomain, Serialize, Deserialize)]
#[serde(bound = "
F: 'static
    + PartialEq
    + Clone
    + core::fmt::Debug
    + Serialize
    + for<'a> Deserialize<'a>,
[usize; D]: Serialize + for<'a> Deserialize<'a>,
")]
pub struct CartesianSubDomainRods<F, const D: usize> {
    /// Base subdomain as created by the [CartesianCuboid] domain.
    #[Base]
    pub subdomain: CartesianSubDomain<F, D>,
    /// See [CartesianCuboidRods]
    pub gel_pressure: F,
    /// Computes friction at all surfaces of the box
    pub surface_friction: F,
    /// The distance for which the friction will be applied
    pub surface_friction_distance: F,
}

impl<F>
    SubDomainForce<
        Matrix<F, Dyn, Const<3>, VecStorage<F, Dyn, Const<3>>>,
        Matrix<F, Dyn, Const<3>, VecStorage<F, Dyn, Const<3>>>,
        Matrix<F, Dyn, Const<3>, VecStorage<F, Dyn, Const<3>>>,
    > for CartesianSubDomainRods<F, 3>
where
    F: nalgebra::RealField + num::Float,
{
    fn calculate_custom_force(
        &self,
        pos: &Matrix<F, Dyn, Const<3>, VecStorage<F, Dyn, Const<3>>>,
        vel: &Matrix<F, Dyn, Const<3>, VecStorage<F, Dyn, Const<3>>>,
    ) -> Result<
        Matrix<F, Dyn, Const<3>, VecStorage<F, Dyn, Const<3>>>,
        cellular_raza_concepts::CalcError,
    > {
        use core::ops::AddAssign;
        let mut force = nalgebra::MatrixXx3::from_fn(pos.nrows(), |_, m| {
            if m == 2 {
                -self.gel_pressure
            } else {
                F::zero()
            }
        });
        for (i, (p, v)) in pos.row_iter().zip(vel.row_iter()).enumerate() {
            let d1 = (p.transpose() - self.subdomain.domain_min)
                .map(|x| <F as num::Float>::abs(x) <= self.surface_friction_distance);
            let d2 = (p.transpose() - self.subdomain.domain_max)
                .map(|x| <F as num::Float>::abs(x) <= self.surface_friction_distance);
            let q = v.norm();
            if q != F::zero() && d1.zip_map(&d2, |x, y| x || y).into_iter().any(|x| *x) {
                let dir = v / q;
                force
                    .row_mut(i)
                    .add_assign(-dir * self.gel_pressure * self.surface_friction);
            }
        }
        Ok(force)
    }
}

impl<F, const D: usize>
    SubDomainMechanics<
        Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
    > for CartesianSubDomainRods<F, D>
where
    F: nalgebra::RealField + num::Float,
{
    fn apply_boundary(
        &self,
        pos: &mut Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
        vel: &mut Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>,
    ) -> Result<(), BoundaryError> {
        // TODO refactor this with matrix multiplication!!!
        // This will probably be much more efficient and less error-prone!

        // For each position in the springs Agent<D1, D2>
        let two = F::one() + F::one();
        pos.row_iter_mut()
            .zip(vel.row_iter_mut())
            .for_each(|(mut p, mut v)| {
                // For each dimension in the space
                for i in 0..p.ncols() {
                    // Check if the particle is below lower edge
                    if p[i] < self.subdomain.get_domain_min()[i] {
                        p[i] = self.subdomain.get_domain_min()[i] * two - p[i];
                        v[i] = <F as num::Float>::abs(v[i]);
                    }

                    // Check if the particle is over the edge
                    if p[i] > self.subdomain.get_domain_max()[i] {
                        p[i] = self.subdomain.get_domain_max()[i] * two - p[i];
                        v[i] = -<F as num::Float>::abs(v[i]);
                    }
                }
            });

        // If new pos is still out of boundary return error
        for j in 0..pos.nrows() {
            let p = pos.row(j);
            for i in 0..pos.ncols() {
                if p[i] < self.subdomain.get_domain_min()[i]
                    || p[i] > self.subdomain.get_domain_max()[i]
                {
                    return Err(BoundaryError(format!(
                        "Particle is out of domain at pos {:?}",
                        pos
                    )));
                }
            }
        }
        Ok(())
    }
}

impl<C, F, const D: usize> SortCells<C> for CartesianSubDomainRods<F, D>
where
    C: Position<Matrix<F, Dyn, Const<D>, VecStorage<F, Dyn, Const<D>>>>,
    F: 'static
        + nalgebra::Field
        + Clone
        + core::fmt::Debug
        + num::FromPrimitive
        + num::ToPrimitive
        + num::Float
        + Copy,
{
    type VoxelIndex = [usize; D];

    fn get_voxel_index_of(&self, cell: &C) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.pos().row_sum().transpose() / F::from_usize(cell.pos().nrows()).unwrap();
        let index = self.subdomain.get_index_of(pos)?;
        Ok(index)
    }
}

impl<F, const D: usize> RodMechanics<F, D> {
    /// Divides a [RodMechanics] struct into two thus separating their positions
    ///
    /// ```
    /// # use cellular_raza_building_blocks::*;
    /// use nalgebra::MatrixXx2;
    /// let n_vertices = 7;
    /// let mut pos = MatrixXx2::zeros(n_vertices);
    /// pos
    ///     .row_iter_mut()
    ///     .enumerate()
    ///     .for_each(|(n_row, mut r)| r[0] += n_row as f32 * 0.5);
    /// let mut m1 = RodMechanics {
    ///     pos,
    ///     vel: MatrixXx2::zeros(n_vertices),
    ///     diffusion_constant: 0.0,
    ///     spring_tension: 0.1,
    ///     rigidity: 0.05,
    ///     spring_length: 0.5,
    ///     damping: 0.0,
    /// };
    /// let radius = 0.25;
    /// let m2 = m1.divide(radius)?;
    ///
    /// let last_pos_m1 = m1.pos.row(6);
    /// let first_pos_m2 = m2.pos.row(0);
    /// assert!(((last_pos_m1 - first_pos_m2).norm() - 2.0 * radius).abs() < 1e-3);
    /// # Result::<(), cellular_raza_concepts::DivisionError>::Ok(())
    /// ```
    pub fn divide(
        &mut self,
        radius: F,
    ) -> Result<RodMechanics<F, D>, cellular_raza_concepts::DivisionError>
    where
        F: num::Float + nalgebra::RealField + FromPrimitive + std::iter::Sum,
    {
        use itertools::Itertools;
        let pos = self.pos();
        let c1 = self;
        let mut c2 = c1.clone();

        let n_rows = c1.pos.nrows();
        // Calculate the fraction of how much we need to scale down the individual spring lengths
        // in order for the distances to still work.
        let two = F::one() + F::one();
        let one_half = F::one() / two;
        let div_factor = one_half - radius / (F::from_usize(n_rows).unwrap() * c1.spring_length);

        // Shrink spring length
        let new_spring_length = div_factor * c1.spring_length;
        c1.spring_length = new_spring_length;
        c2.spring_length = new_spring_length;

        // Set starting point
        c1.pos.set_row(0, &pos.row(0));
        c2.pos
            .set_row(c2.pos.nrows() - 1, &pos.row(c2.pos.nrows() - 1));

        let segments: Vec<_> = pos
            .row_iter()
            .tuple_windows::<(_, _)>()
            .map(|(x, y)| (x - y).norm())
            .collect();
        let segment_length = (segments.iter().map(|&x| x).sum::<F>() - two * radius)
            / F::from_usize(c2.pos.nrows() - 1).unwrap()
            / two;

        for n_vertex in 0..c2.pos.nrows() {
            // Get smallest index k such that the beginning of the new segment is "farther" than the
            // original vertex at this index k.
            let k = (0..segments.len())
                .filter(|n| {
                    segments.iter().map(|&x| x).take(*n).sum::<F>()
                        <= F::from_usize(n_vertex).unwrap() * segment_length
                })
                .max()
                .unwrap();
            let q = (F::from_usize(n_vertex).unwrap() * segment_length
                - segments.iter().map(|&x| x).take(k).sum::<F>())
                / segments[k];
            c1.pos.set_row(
                n_vertex,
                &(pos.row(k) * (F::one() - q) + pos.row(k + 1) * q),
            );

            let m = (0..c2.pos.nrows())
                .filter(|n| {
                    segments.iter().rev().map(|&x| x).take(*n).sum::<F>()
                        <= F::from_usize(n_vertex).unwrap() * segment_length
                })
                .max()
                .unwrap();
            let p = (F::from_usize(n_vertex).unwrap() * segment_length
                - segments.iter().rev().map(|&x| x).take(m).sum::<F>())
                / segments[c2.pos.nrows() - m - 2];
            c2.pos.set_row(
                c2.pos.nrows() - n_vertex - 1,
                &(pos.row(c2.pos.nrows() - m - 1) * (F::one() - p)
                    + pos.row(c2.pos.nrows() - m - 2) * p),
            );
        }
        Ok(c2)
    }
}
