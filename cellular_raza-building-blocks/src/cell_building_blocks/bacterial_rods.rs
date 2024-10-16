use crate::{CartesianCuboid, CartesianSubDomain};
use cellular_raza_concepts::*;

use num::FromPrimitive;
use serde::{ser::SerializeStruct, Deserialize, Serialize};

/// A mechanical model for Bacterial Rods
///
/// See the [Bacterial Rods](https://cellular-raza.com/showcase/bacterial-rods) example for more
/// detailed information.
#[derive(Clone, Debug, PartialEq)]
pub struct RodMechanics<F, const D1: usize, const D2: usize> {
    /// The current position
    pub pos: nalgebra::SMatrix<F, D1, D2>,
    /// The current velocity
    pub vel: nalgebra::SMatrix<F, D1, D2>,
    /// Controls magnitude of stochastic motion
    pub diffusion_constant: F,
    /// Spring tension between individual vertices
    pub spring_tension: F,
    /// Stiffness at each joint connecting two edges
    pub angle_stiffness: F,
    /// Target spring length
    pub spring_length: F,
    /// Daming constant
    pub damping: F,
}

impl<F, const D1: usize, const D2: usize>
    Mechanics<
        nalgebra::SMatrix<F, D1, D2>,
        nalgebra::SMatrix<F, D1, D2>,
        nalgebra::SMatrix<F, D1, D2>,
        F,
    > for RodMechanics<F, D1, D2>
where
    F: nalgebra::RealField + Clone + num::Float,
    rand_distr::StandardNormal: rand_distr::Distribution<F>,
{
    fn calculate_increment(
        &self,
        force: nalgebra::SMatrix<F, D1, D2>,
    ) -> Result<(nalgebra::SMatrix<F, D1, D2>, nalgebra::SMatrix<F, D1, D2>), CalcError> {
        use core::ops::AddAssign;
        let one_half = F::one() / (F::one() + F::one());

        // Calculate internal force between the two points of the Agent
        let mut total_force = force;

        // Calculate force exerted by spring action between individual vertices
        let dist_internal =
            self.pos.rows(0, self.pos.nrows() - 1) - self.pos.rows(1, self.pos.nrows() - 1);
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

        // Calculate force exerted by angle-contributions
        use itertools::Itertools;
        dist_internal
            .row_iter()
            .tuple_windows::<(_, _)>()
            .enumerate()
            .for_each(|(i, (conn1, conn2))| {
                let angle = conn1.angle(&-conn2);
                let force_d = conn1.normalize() - conn2.normalize();
                let force_direction = if !force_d.norm().is_zero() {
                    force_d.normalize()
                } else {
                    force_d
                };
                let force = force_direction * self.angle_stiffness * (F::pi() - angle);
                total_force.row_mut(i).add_assign(-force * one_half);
                total_force.row_mut(i + 1).add_assign(force);
                total_force.row_mut(i + 2).add_assign(-force * one_half);
            });

        // Calculate damping force
        total_force -= self.vel * self.damping;
        Ok((self.vel.clone(), total_force))
    }

    fn get_random_contribution(
        &self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: F,
    ) -> Result<(nalgebra::SMatrix<F, D1, D2>, nalgebra::SMatrix<F, D1, D2>), RngError> {
        let distr = match rand_distr::Normal::new(F::zero(), <F as num::Float>::sqrt(dt)) {
            Ok(e) => Ok(e),
            Err(e) => Err(cellular_raza_concepts::RngError(format!("{e}"))),
        }?;
        let dpos = nalgebra::SMatrix::<F, D1, D2>::from_distribution(&distr, rng)
            * <F as num::Float>::powi(F::one() + F::one(), -2)
            * self.diffusion_constant
            / dt;
        let dvel = nalgebra::SMatrix::<F, D1, D2>::zeros();

        Ok((dpos, dvel))
    }
}

impl<F: Clone, const D1: usize, const D2: usize> Position<nalgebra::SMatrix<F, D1, D2>>
    for RodMechanics<F, D1, D2>
{
    fn pos(&self) -> nalgebra::SMatrix<F, D1, D2> {
        self.pos.clone()
    }

    fn set_pos(&mut self, position: &nalgebra::SMatrix<F, D1, D2>) {
        self.pos = position.clone();
    }
}

impl<F: Clone, const D1: usize, const D2: usize> Velocity<nalgebra::SMatrix<F, D1, D2>>
    for RodMechanics<F, D1, D2>
{
    fn velocity(&self) -> nalgebra::SMatrix<F, D1, D2> {
        self.vel.clone()
    }

    fn set_velocity(&mut self, velocity: &nalgebra::SMatrix<F, D1, D2>) {
        self.vel = velocity.clone();
    }
}

/// Automatically derives a [Interaction] suitable for rods from a point-wise interaction.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RodInteraction<I>(pub I);

impl<I, F, Inf, const D1: usize, const D2: usize>
    Interaction<
        nalgebra::SMatrix<F, D1, D2>,
        nalgebra::SMatrix<F, D1, D2>,
        nalgebra::SMatrix<F, D1, D2>,
        Inf,
    > for RodInteraction<I>
where
    I: Interaction<
        nalgebra::SVector<F, D2>,
        nalgebra::SVector<F, D2>,
        nalgebra::SVector<F, D2>,
        Inf,
    >,
    F: 'static + nalgebra::RealField + Copy + core::fmt::Debug + num::Zero,
{
    fn get_interaction_information(&self) -> Inf {
        self.0.get_interaction_information()
    }

    fn calculate_force_between(
        &self,
        own_pos: &nalgebra::SMatrix<F, D1, D2>,
        own_vel: &nalgebra::SMatrix<F, D1, D2>,
        ext_pos: &nalgebra::SMatrix<F, D1, D2>,
        ext_vel: &nalgebra::SMatrix<F, D1, D2>,
        ext_inf: &Inf,
    ) -> Result<(nalgebra::SMatrix<F, D1, D2>, nalgebra::SMatrix<F, D1, D2>), CalcError> {
        use core::ops::AddAssign;
        use itertools::Itertools;
        let mut force_own = nalgebra::SMatrix::<F, D1, D2>::zeros();
        let mut force_ext = nalgebra::SMatrix::<F, D1, D2>::zeros();
        for (i, p1) in own_pos.row_iter().enumerate() {
            for (j, (p2_n0, p2_n1)) in ext_pos.row_iter().tuple_windows::<(_, _)>().enumerate() {
                // Calculate the closest point of the external position
                let (_, nearest_point, rel_length) = crate::nearest_point_from_point_to_line(
                    &p1.transpose(),
                    &(p2_n0.transpose(), p2_n1.transpose()),
                );

                let (f_own, f_ext) = self.0.calculate_force_between(
                    &p1.transpose().into(),
                    &own_vel.row(i).transpose().into(),
                    &nearest_point.into(),
                    &ext_vel.row(j).transpose().into(),
                    ext_inf,
                )?;

                force_own.row_mut(i).add_assign(f_own.transpose());
                force_ext
                    .row_mut(j)
                    .add_assign(f_ext.transpose() * (F::one() - rel_length));
                force_ext
                    .row_mut((j + 1) % D1)
                    .add_assign(f_ext.transpose() * rel_length);
            }
        }
        Ok((force_own, force_ext))
    }
}

/// Cells are represented by rods
#[derive(Domain)]
pub struct CartesianCuboidRods<F, const D1: usize, const D2: usize> {
    /// The base-cuboid which is being repurposed
    #[DomainRngSeed]
    pub domain: CartesianCuboid<F, D2>,
}

impl<C, F, const D1: usize, const D2: usize> SortCells<C> for CartesianCuboidRods<F, D1, D2>
where
    C: Position<nalgebra::SMatrix<F, D1, D2>>,
    F: 'static
        + nalgebra::Field
        + Clone
        + core::fmt::Debug
        + num::FromPrimitive
        + num::ToPrimitive
        + num::Float
        + Copy,
{
    type VoxelIndex = [usize; D2];

    fn get_voxel_index_of(&self, cell: &C) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.pos().row_sum().transpose() / F::from_usize(D1).unwrap();
        let index = self.domain.get_voxel_index_of_raw(&pos)?;
        Ok(index)
    }
}

impl<F, const D1: usize, const D2: usize> DomainCreateSubDomains<CartesianSubDomainRods<F, D1, D2>>
    for CartesianCuboidRods<F, D1, D2>
where
    F: 'static + num::Float + core::fmt::Debug + num::FromPrimitive,
{
    type SubDomainIndex = usize;
    type VoxelIndex = [usize; D2];

    fn create_subdomains(
        &self,
        n_subdomains: std::num::NonZeroUsize,
    ) -> Result<
        impl IntoIterator<
            Item = (
                Self::SubDomainIndex,
                CartesianSubDomainRods<F, D1, D2>,
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
                    CartesianSubDomainRods::<F, D1, D2> { subdomain },
                    voxels,
                )
            }))
    }
}

/// The corresponding SubDomain of the [CartesianCuboidRods] domain.
#[derive(Clone, SubDomain)]
pub struct CartesianSubDomainRods<F, const D1: usize, const D2: usize> {
    /// Base subdomain as created by the [CartesianCuboid] domain.
    #[Base]
    pub subdomain: CartesianSubDomain<F, D2>,
}

impl<F, const D1: usize, const D2: usize>
    SubDomainMechanics<nalgebra::SMatrix<F, D1, D2>, nalgebra::SMatrix<F, D1, D2>>
    for CartesianSubDomainRods<F, D1, D2>
where
    F: nalgebra::RealField + num::Float,
{
    fn apply_boundary(
        &self,
        pos: &mut nalgebra::SMatrix<F, D1, D2>,
        vel: &mut nalgebra::SMatrix<F, D1, D2>,
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

#[derive(Deserialize)]
#[serde(rename(
    serialize = "CartesianSubDomainRods",
    deserialize = "CartesianSubDomainRods",
))]
struct __CartesianSubDomainRodsSerde<F, const D1: usize, const D2: usize>
where
    F: 'static + Clone + core::fmt::Debug + PartialEq + nalgebra::Scalar,
    CartesianSubDomain<F, D2>: for<'a> Deserialize<'a>,
{
    subdomain: CartesianSubDomain<F, D2>,
}

impl<F, const D1: usize, const D2: usize> From<__CartesianSubDomainRodsSerde<F, D1, D2>>
    for CartesianSubDomainRods<F, D1, D2>
where
    F: 'static + Clone + core::fmt::Debug + PartialEq + for<'a> Deserialize<'a>,
{
    fn from(s: __CartesianSubDomainRodsSerde<F, D1, D2>) -> Self {
        CartesianSubDomainRods {
            subdomain: s.subdomain,
        }
    }
}

impl<F, const D1: usize, const D2: usize> Serialize for CartesianSubDomainRods<F, D1, D2>
where
    F: nalgebra::Scalar + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.subdomain.serialize(serializer)
    }
}

impl<'de, F, const D1: usize, const D2: usize> Deserialize<'de>
    for CartesianSubDomainRods<F, D1, D2>
where
    F: nalgebra::Scalar + for<'a> Deserialize<'a>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let s = __CartesianSubDomainRodsSerde::deserialize(deserializer)?;
        let subdomain = s.into();
        Ok(subdomain)
    }
}

impl<C, F, const D1: usize, const D2: usize> SortCells<C> for CartesianSubDomainRods<F, D1, D2>
where
    C: Position<nalgebra::SMatrix<F, D1, D2>>,
    F: 'static
        + nalgebra::Field
        + Clone
        + core::fmt::Debug
        + num::FromPrimitive
        + num::ToPrimitive
        + num::Float
        + Copy,
{
    type VoxelIndex = [usize; D2];

    fn get_voxel_index_of(&self, cell: &C) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.pos().row_sum().transpose() / F::from_usize(D1).unwrap();
        let index = self.subdomain.get_index_of(pos)?;
        Ok(index)
    }
}

#[derive(Deserialize)]
#[serde(rename(serialize = "RodMechanics", deserialize = "RodMechanics",))]
struct __RodMechanicsSerde<
    F: 'static + Clone + core::fmt::Debug + PartialEq,
    const D1: usize,
    const D2: usize,
> {
    pos: nalgebra::SMatrix<F, D1, D2>,
    vel: nalgebra::SMatrix<F, D1, D2>,
    diffusion_constant: F,
    spring_tension: F,
    angle_stiffness: F,
    spring_length: F,
    damping: F,
}

impl<F, const D1: usize, const D2: usize> From<__RodMechanicsSerde<F, D1, D2>>
    for RodMechanics<F, D1, D2>
where
    F: 'static + Clone + core::fmt::Debug + PartialEq,
{
    fn from(value: __RodMechanicsSerde<F, D1, D2>) -> Self {
        RodMechanics {
            pos: value.pos,
            vel: value.vel,
            diffusion_constant: value.diffusion_constant,
            spring_tension: value.spring_tension,
            angle_stiffness: value.angle_stiffness,
            spring_length: value.spring_length,
            damping: value.damping,
        }
    }
}

impl<F, const D1: usize, const D2: usize> Serialize for RodMechanics<F, D1, D2>
where
    F: nalgebra::Scalar + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("RodMechanics", 6)?;
        state.serialize_field("pos", &self.pos)?;
        state.serialize_field("pos", &self.pos)?;
        state.serialize_field("vel", &self.vel)?;
        state.serialize_field("diffusion_constant", &self.diffusion_constant)?;
        state.serialize_field("spring_tension", &self.spring_tension)?;
        state.serialize_field("angle_stiffness", &self.angle_stiffness)?;
        state.serialize_field("spring_length", &self.spring_length)?;
        state.serialize_field("damping", &self.damping)?;
        state.end()
    }
}

impl<'de, F, const D1: usize, const D2: usize> Deserialize<'de> for RodMechanics<F, D1, D2>
where
    F: nalgebra::Scalar + for<'a> Deserialize<'a>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let r = __RodMechanicsSerde::deserialize(deserializer)?;
        let rodmechanics = r.into();
        Ok(rodmechanics)
    }
}

impl<F, const D1: usize, const D2: usize> RodMechanics<F, D1, D2> {
    /// Divides a [RodMechanics] struct into two thus separating their positions
    ///
    /// ```
    /// # use cellular_raza_building_blocks::*;
    /// let mut m1 = RodMechanics {
    ///     pos: nalgebra::SMatrix::<f32, 7, 2>::from_fn(|r, c| if c == 0 {0.5 * r as f32} else {0.0}),
    ///     vel: nalgebra::SMatrix::<f32, 7, 2>::zeros(),
    ///     diffusion_constant: 0.0,
    ///     spring_tension: 0.1,
    ///     angle_stiffness: 0.05,
    ///     spring_length: 0.5,
    ///     damping: 0.0,
    /// };
    /// let radius = 0.25;
    /// let m2 = m1.divide(radius)?;
    ///
    /// let last_pos_m1 = m1.pos.row(6);
    /// let first_pos_m2 = m2.pos.row(0);
    /// println!("{} {}", m1.pos, m2.pos);
    /// println!("{} {}", (last_pos_m1 - first_pos_m2).norm(), 2.0 * radius);
    /// assert!(((last_pos_m1 - first_pos_m2).norm() - 2.0 * radius).abs() < 1e-3);
    /// # Result::<(), cellular_raza_concepts::DivisionError>::Ok(())
    /// ```
    pub fn divide(
        &mut self,
        radius: F,
    ) -> Result<RodMechanics<F, D1, D2>, cellular_raza_concepts::DivisionError>
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
        c1.spring_length = div_factor * c1.spring_length;
        c2.spring_length = div_factor * c1.spring_length;

        // Set starting point
        c1.pos.set_row(0, &pos.row(0));
        c2.pos.set_row(D1 - 1, &pos.row(D1 - 1));

        let segments: Vec<_> = pos
            .row_iter()
            .tuple_windows::<(_, _)>()
            .map(|(x, y)| (x - y).norm())
            .collect();
        let segment_length = (segments.iter().map(|&x| x).sum::<F>() - two * radius)
            / F::from_usize(D1 - 1).unwrap()
            / two;

        for n_vertex in 0..D1 {
            // Get smallest index k such that the beginning of the new segment is "farther" than the
            // original vertex at this index k.
            let k = (0..D1)
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

            let m = (0..D1)
                .filter(|n| {
                    segments.iter().rev().map(|&x| x).take(*n).sum::<F>()
                        <= F::from_usize(n_vertex).unwrap() * segment_length
                })
                .max()
                .unwrap();
            let p = (F::from_usize(n_vertex).unwrap() * segment_length
                - segments.iter().rev().map(|&x| x).take(m).sum::<F>())
                / segments[D1 - m - 2];
            c2.pos.set_row(
                D1 - n_vertex - 1,
                &(pos.row(D1 - m - 1) * (F::one() - p) + pos.row(D1 - m - 2) * p),
            );
        }
        Ok(c2)
    }
}
