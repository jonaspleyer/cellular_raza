use std::collections::HashMap;

use cellular_raza::prelude::*;

use nalgebra::{Matrix6xX, Vector3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub const N_CELLS: usize = 8;

pub const CELL_DAMPING: f64 = 0.4;
pub const CELL_RADIUS: f64 = 6.0;
pub const PARTICLE_MASS: f64 = 0.15;

pub const CELL_MECHANICS_RELATIVE_INTERACTION_RANGE: f64 = 1.0;
pub const CELL_MECHANICS_POTENTIAL_STRENGTH: f64 = 5.0;

pub const CELL_GROWTH_INCREMENT: f64 = 0.5;
pub const CELL_DIVISION_RADIUS: f64 = 12.0;
pub const CELL_DEVOUR_RATE: f64 = 0.01;

pub const DT: f64 = 0.1;
pub const N_TIMES: u64 = 2_000;
pub const SAVE_INTERVAL: u64 = 10;

pub const DOMAIN_SIZE: f64 = 100.0;

#[derive(CellAgent, Clone, Deserialize, Serialize)]
struct Cell {
    #[Interaction]
    interaction: MorsePotential,
    #[Mechanics]
    mechanics: NewtonDamped3D,
    particles: ParticleVec,
    retain_particle_indices: Vec<usize>,
}

fn reflect_at(
    // The particle which should be reflected
    particle: &mut nalgebra::SVectorViewMut<f64, 6>,
    // Stützpunkt
    q: &nalgebra::SVectorView<f64, 3>,
    // Normal pointing from q into the prohibited space
    dir: &nalgebra::SVectorView<f64, 3>,
) -> nalgebra::SVector<f64, 3> {
    let pos = particle.view((0, 0), (3, 1));
    let dir = dir.normalize();
    // Connect from given reflection point to particle position
    let diff = pos - q;
    // Length of penetration normal to dir
    let d = diff.dot(&dir).abs();

    use core::ops::SubAssign;
    let mut pos = particle.view_mut((0, 0), (3, 1));
    pos.sub_assign(2.0 * d * dir);

    let mut vel = particle.view_mut((3, 0), (3, 1));
    let vel_change = 2.0 * vel.dot(&dir) * dir;
    vel.sub_assign(vel_change);

    // Return the change in velocity
    vel_change
}

impl Cycle for Cell {
    fn update_cycle(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Self,
    ) -> Option<CycleEvent> {
        if cell.interaction.radius >= CELL_DIVISION_RADIUS {
            Some(CycleEvent::Division)
        } else {
            for i in 0..cell.particles.ncols() {
                if rng.random_bool(dt * CELL_DEVOUR_RATE) {
                    cell.interaction.radius += CELL_GROWTH_INCREMENT;
                    cell.interaction.cutoff =
                        cell.interaction.radius * CELL_MECHANICS_RELATIVE_INTERACTION_RANGE;
                } else {
                    cell.retain_particle_indices.push(i);
                }
            }
            None
        }
    }

    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Self) -> Result<Self, DivisionError> {
        // Calculate new radius (divide the area in half)
        let r = cell.interaction.radius;
        let r_new = r / core::f64::consts::SQRT_2;
        // Calculate random direction
        let phi = rng.random_range(0.0..2.0 * core::f64::consts::PI);
        // let theta = rng.random_range(0.0..core::f64::consts::PI);
        let theta = core::f64::consts::FRAC_PI_2;
        let dir = Vector3::from([
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ]);
        // Define new positions
        let p = cell.mechanics.pos;
        let p1 = p + r_new * dir;
        let p2 = p - r_new * dir;

        // TODO Define new particles

        // Create new cell
        cell.interaction.radius = r_new;
        let c1 = cell;
        let mut c2 = c1.clone();
        c1.mechanics.pos = p1;
        c2.mechanics.pos = p2;

        Ok(c2)
    }
}

/// The ParticleVec consists of entries
///
/// ```text
/// v[0] = pos0;
/// v[1] = pos1;
/// v[2] = pos2;
/// v[3] = vel0;
/// v[4] = vel1;
/// v[5] = vel2;
/// ```
type ParticleVec<F = f64> = Matrix6xX<F>;

impl Intracellular<ParticleVec> for Cell {
    fn get_intracellular(&self) -> ParticleVec {
        self.particles.clone()
    }

    fn set_intracellular(&mut self, intracellular: ParticleVec) {
        // Apply collision detection
        self.particles = intracellular;
        for n in 0..self.particles.ncols() {
            let pos = self.particles.view((0, n), (3, 1));
            let diff = pos - self.mechanics.pos;
            let d = diff.norm() - self.interaction.radius;

            // Calculate new point for particle
            if d >= 0.0 {
                let di = diff.normalize();

                let vel_change = reflect_at(
                    &mut self.particles.column_mut(n),
                    &(self.mechanics.pos + self.interaction.radius * di).as_view(),
                    &di.as_view(),
                );
                self.mechanics.vel += vel_change * PARTICLE_MASS / self.mechanics.mass;
            }
        }

        let retained_particles =
            ParticleVec::from_fn(self.retain_particle_indices.len(), |i, n| {
                let j = self.retain_particle_indices[n];
                self.particles[(i, j)]
            });
        self.particles = retained_particles;
        self.retain_particle_indices.clear();
    }
}

impl Reactions<ParticleVec> for Cell {
    fn calculate_intracellular_increment(
        &self,
        intracellular: &ParticleVec,
    ) -> Result<ParticleVec, CalcError> {
        let mut dintra = intracellular.clone();
        dintra.swap_rows(0, 3);
        dintra.swap_rows(1, 4);
        dintra.swap_rows(2, 5);

        let mut dvel = dintra.view_mut((3, 0), (3, intracellular.ncols()));
        dvel *= 0.0;

        Ok(dintra)
    }
}

impl ReactionsExtra<ParticleVec, ParticleVec> for Cell {
    fn calculate_combined_increment(
        &self,
        intracellular: &ParticleVec,
        extracellular: &ParticleVec,
    ) -> Result<(ParticleVec, ParticleVec), CalcError> {
        Ok((intracellular * 0.0, extracellular * 0.0))
    }
}

#[derive(Clone)]
struct Cuboid3D<F, const D: usize> {
    base: CartesianCuboid<F, D>,
    particles: ParticleVec<F>,
}

impl<F, C, Ci, const D: usize> Domain<C, CuboidSubDomain<F, D>, Ci> for Cuboid3D<F, D>
where
    CartesianCuboid<F, D>: Domain<C, CartesianSubDomain<F, D>, Ci>,
    F: 'static + num::Float + core::fmt::Debug + core::ops::SubAssign + core::ops::DivAssign,
{
    type SubDomainIndex =
        <CartesianCuboid<F, D> as Domain<C, CartesianSubDomain<F, D>, Ci>>::SubDomainIndex;
    type VoxelIndex =
        <CartesianCuboid<F, D> as Domain<C, CartesianSubDomain<F, D>, Ci>>::VoxelIndex;

    fn decompose(
        self,
        n_subdomains: core::num::NonZeroUsize,
        cells: Ci,
    ) -> Result<DecomposedDomain<Self::SubDomainIndex, CuboidSubDomain<F, D>, C>, DecomposeError>
    {
        let Cuboid3D { base, particles } = self;
        let DecomposedDomain {
            n_subdomains,
            index_subdomain_cells,
            neighbor_map,
            rng_seed,
        } = base.decompose(n_subdomains, cells)?;

        let index_subdomain_cells = index_subdomain_cells
            .into_iter()
            .map(|(ind, base, cells)| {
                // TODO fixme this is completely wrong
                let particles = particles.clone();
                let particle_indices: HashMap<[usize; D], Vec<usize>> = particles
                    .column_iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let mut q = [F::zero(); D];
                        for i in 0..D {
                            q[i] = p[i];
                        }
                        let index = base.get_index_of(q).unwrap();
                        (index, i)
                    })
                    .fold(HashMap::new(), |mut acc, (index, n_col)| {
                        acc.entry(index)
                            .and_modify(|x: &mut Vec<usize>| x.push(n_col))
                            .or_insert(vec![n_col]);
                        acc
                    });

                let sbd = CuboidSubDomain {
                    base,
                    particles,
                    particle_indices,
                };
                (ind, sbd, cells)
            })
            .collect();

        Ok(DecomposedDomain {
            n_subdomains,
            index_subdomain_cells,
            neighbor_map,
            rng_seed,
        })
    }
}

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
struct CuboidSubDomain<F, const D: usize> {
    #[Base]
    #[SortCells]
    #[Mechanics]
    base: CartesianSubDomain<F, D>,
    particles: ParticleVec<F>,
    #[serde(skip)]
    particle_indices: HashMap<[usize; D], Vec<usize>>,
}

impl SubDomainReactions<Vector3<f64>, ParticleVec, f64> for CuboidSubDomain<f64, 3> {
    type NeighborValue = ParticleVec;
    type BorderInfo = ParticleVec;

    fn treat_increments<I, J>(
        &mut self,
        _neighbors: I,
        _sources: J,
    ) -> Result<(), cellular_raza::concepts::CalcError>
    where
        I: IntoIterator<Item = Self::NeighborValue>,
        J: IntoIterator<Item = (Vector3<f64>, ParticleVec)>,
    {
        // Bounce off cells
        Ok(())
    }

    fn update_fluid_dynamics(&mut self, dt: f64) -> Result<(), cellular_raza::concepts::CalcError> {
        // TODO psi
        for n in 0..self.particles.ncols() {
            for i in 0..3 {
                self.particles[(i, n)] += dt * self.particles[(i + 3, n)];
            }

            // Put in new voxel
            let index = self
                .base
                .get_index_of([self.particles[0], self.particles[1], self.particles[2]])
                .map_err(|e| cellular_raza::prelude::CalcError(e.to_string()))?;

            // Handle particle-cell collisions

            // Apply boundary conditions
            let mut pos = [
                self.particles[(0, n)],
                self.particles[(1, n)],
                self.particles[(2, n)],
            ];
            let mut vel = [
                self.particles[(3, n)],
                self.particles[(4, n)],
                self.particles[(5, n)],
            ];
            self.base
                .apply_boundary(&mut pos, &mut vel)
                .map_err(|e| cellular_raza::prelude::CalcError(e.to_string()))?;

            for i in 0..3 {
                self.particles[(i, n)] = pos[i];
                self.particles[(i + 3, n)] = vel[i];
            }
        }
        Ok(())
    }

    fn get_extracellular_at_pos(&self, pos: &Vector3<f64>) -> Result<ParticleVec, CalcError> {
        let index = self
            .base
            .get_index_of([pos[0], pos[1], pos[2]])
            .map_err(|e| CalcError(e.to_string()))?;

        let particles = self
            .particle_indices
            .get(&index)
            .map(|inds| {
                let mut x = ParticleVec::zeros(inds.len());
                inds.iter()
                    .enumerate()
                    .for_each(|(i, j)| x.set_column(i, &self.particles.column(*j)));
                x
            })
            .unwrap_or_default();

        Ok(particles)
    }

    fn get_neighbor_value(&self, _: Self::BorderInfo) -> Self::NeighborValue {
        todo!()
    }

    fn get_border_info(&self) -> Self::BorderInfo {
        todo!()
    }
}

fn main() -> Result<(), SimulationError> {
    // Define the seed
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let cells = (0..N_CELLS)
        .map(|_| {
            let pos = Vector3::from([
                rng.random_range(0.2 * DOMAIN_SIZE..0.8 * DOMAIN_SIZE),
                rng.random_range(0.2 * DOMAIN_SIZE..0.8 * DOMAIN_SIZE),
                DOMAIN_SIZE / 2.0,
            ]);
            Cell {
                mechanics: NewtonDamped3D {
                    pos,
                    vel: Vector3::from([
                        rng.random_range(-0.1..0.1),
                        rng.random_range(-0.1..0.1),
                        0.0,
                    ]),
                    damping_constant: CELL_DAMPING,
                    mass: 1.0,
                },
                interaction: MorsePotential {
                    radius: CELL_RADIUS,
                    potential_stiffness: 0.1,
                    strength: CELL_MECHANICS_POTENTIAL_STRENGTH,
                    cutoff: CELL_MECHANICS_RELATIVE_INTERACTION_RANGE * CELL_RADIUS,
                },
                // particles: MatrixXx3::from_fn_generic(4, 3, |_, m| pos[m]),
                particles: ParticleVec::from_fn(20, |n, _| {
                    if n < 3 {
                        pos[n] + rng.random_range(-1.0..1.0) * CELL_RADIUS
                    } else if n == 5 {
                        0.0
                    } else {
                        rng.random_range(-1.0..1.0)
                    }
                }),
                retain_particle_indices: Vec::with_capacity(20),
            }
        })
        .collect::<Vec<_>>();

    let base = CartesianCuboid::from_boundaries_and_n_voxels([0.0; 3], [DOMAIN_SIZE; 3], [1; 3])?;

    let n_particles_outer = 100;

    let mut particles = ParticleVec::zeros(n_particles_outer);
    for mut pi in particles.column_iter_mut() {
        pi[0] = rng.random_range(0.0..DOMAIN_SIZE);
        pi[1] = rng.random_range(0.0..DOMAIN_SIZE);
        // pi[2] = rng.random_range(0.0..DOMAIN_SIZE);
        pi[2] = DOMAIN_SIZE / 2.0;

        pi[3] = rng.random_range(-1.0..1.0);
        pi[4] = rng.random_range(-1.0..1.0);
        // pi[5] = rng.random_range(-1.0..1.0);
        pi[5] = 0.0;
    }

    let domain = Cuboid3D { base, particles };

    let time = cellular_raza::core::time::FixedStepsize::from_partial_save_steps(
        0.0,
        DT,
        N_TIMES,
        SAVE_INTERVAL,
    )?;
    let storage_builder = StorageBuilder::new().location("out/extracellular_particles");

    let settings = cellular_raza::core::backend::chili::Settings {
        n_threads: 1.try_into().unwrap(),
        time,
        storage: storage_builder,
        progressbar: Some("".into()),
    };

    let custom_update_fun = |sbox: &mut _| -> Result<(), SimulationError> {
        let sbox: &mut SubDomainBox<_, CuboidSubDomain<f64, 3>, _, _, _, _> = sbox;
        for (v_index, p_indices) in sbox.subdomain.particle_indices.iter() {
            let plain_index = sbox.voxel_index_to_plain_index[v_index];
            for (cbox, _) in sbox.voxels.get_mut(&plain_index).unwrap().cells.iter_mut() {
                let cell: &mut Cell = &mut cbox.cell;
                for n in p_indices.iter() {
                    let pos = sbox.subdomain.particles.view((0, *n), (3, 1));
                    let vel = sbox.subdomain.particles.view((3, *n), (3, 1));
                    let cpos = cell.mechanics.pos;
                    let radius = cell.interaction.radius;

                    let diff = pos - cpos;
                    let dist = diff.norm();
                    // Calculate new point for particle
                    if dist < radius {
                        // Determine point at which particle invaded circle
                        let x = pos - cpos;
                        let v = vel.normalize();
                        let s = -v.dot(&x)
                            - (v.dot(&x).powi(2) + (radius.powi(2) - x.norm_squared())).sqrt();
                        let q = pos + s * v;

                        // Perform reflection
                        let mut particle = sbox.subdomain.particles.column_mut(*n);
                        let vel_change = reflect_at(
                            &mut particle,
                            &q.as_view(),
                            &(cpos - &q).normalize().as_view(),
                        );
                        cell.mechanics.vel += vel_change * PARTICLE_MASS / cell.mechanics.mass;
                    }
                }
            }
        }
        Ok(())
    };

    run_simulation!(
        domain,
        agents: cells,
        settings,
        aspects: [Mechanics, Interaction, Reactions, ReactionsExtra, Cycle],
        zero_reactions_default: |c: &Cell| ParticleVec::zeros(c.particles.ncols()),
        custom_update: custom_update_fun,
    )?;
    Ok(())
}

#[test]
fn test_reflect_at_1() {
    let mut particle = ParticleVec::from_fn(1, |n, _| [1.0, 0.0, 0.0, 1.0, 0.0, 0.0][n]);
    let q = Vector3::from([0.0, 0.0, 0.0]);
    let dir = Vector3::from([1.0, 0.0, 0.0]);
    reflect_at(&mut particle.column_mut(0), &q.as_view(), &dir.as_view());
    assert!(particle[0] == -1.0);
    assert!(particle[1] == 0.0);
    assert!(particle[3] == -1.0);
    assert!(particle[4] == 0.0);
}

#[test]
fn test_reflect_at_2() {
    let mut particle = ParticleVec::from_fn(1, |n, _| [1.0, 1.0, 0.0, 1.0, 0.5, 0.0][n]);
    let q = Vector3::from([0.5, 0.5, 0.0]);
    let dir = Vector3::from([0.5, 0.5, 0.0]);
    reflect_at(&mut particle.column_mut(0), &q.as_view(), &dir.as_view());
    assert!((particle[0] - 0.0).abs() < 1e-8);
    assert!((particle[1] - 0.0).abs() < 1e-8);
    assert!((particle[3] + 0.5).abs() < 1e-8);
    assert!((particle[4] + 1.0).abs() < 1e-8);
}
