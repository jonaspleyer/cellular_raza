use cellular_raza_concepts::errors::{CalcError, RngError};
use cellular_raza_concepts::mechanics::Mechanics;

use itertools::Itertools;
use nalgebra::SVector;

use serde::{Deserialize, Serialize};

/// Simple newtonian dynamics governed by mass and damping.
///
/// The equation of motion is given by
/// \\begin{equation}
///     m \ddot{\vec{x}} = \vec{F} - \lambda \dot{\vec{x}}
/// \\end{equation}
/// where $\vec{F}$ is calculated by the [Interaction](cellular_raza_concepts::interaction::Interaction) trait.
/// The parameter $m$ describes the mass of the object while $\lambda$ is the damping constant.
/// If the cell is growing, we need to increase the mass $m$.
/// By interacting with the outside world, we can adapt $\lambda$ to external values although this is rarely desirable.
/// Both operations need to be implemented by other concepts such as [Cycle](cellular_raza_concepts::cycle::Cycle).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NewtonDamped<const D: usize> {
    /// Current position $\vec{x}$ given by a vector of dimension `D`.
    pub pos: SVector<f64, D>,
    /// Current velocity $\dot{\vec{x}}$ given by a vector of dimension `D`.
    pub vel: SVector<f64, D>,
    /// Damping constant $\lambda$.
    pub damping_constant: f64,
    /// Mass $m$ of the object.
    pub mass: f64,
}

impl<const D: usize> Mechanics<SVector<f64, D>, SVector<f64, D>, SVector<f64, D>> for NewtonDamped<D> {
    fn pos(&self) -> SVector<f64, D> {
        self.pos
    }

    fn velocity(&self) -> SVector<f64, D> {
        self.vel
    }

    fn set_pos(&mut self, p: &SVector<f64, D>) {
        self.pos = *p;
    }

    fn set_velocity(&mut self, v: &SVector<f64, D>) {
        self.vel = *v;
    }

    fn calculate_increment(
        &self,
        force: SVector<f64, D>,
    ) -> Result<(SVector<f64, D>, SVector<f64, D>), CalcError> {
        let dx = self.vel;
        let dv = force / self.mass - self.damping_constant * self.vel;
        Ok((dx, dv))
    }
}

/// An empty struct to signalize that no velocity needs to be updated.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NoVelocity;

impl core::ops::Add for NoVelocity {
    type Output = Self;

    fn add(self, _: Self) -> Self::Output {
        NoVelocity
    }
}

impl<F> core::ops::Mul<F> for NoVelocity {
    type Output = Self;

    fn mul(self, _: F) -> Self::Output {
        NoVelocity
    }
}


/// Brownian motion of particles represented by a spherical potential in arbitrary dimension.
///
/// # Parameters
/// | Symbol | Parameter | Description |
/// | --- | --- | --- |
/// | $\vec{x}$ | `pos` | Position of the particle. |
/// | $D$ | `diffusion_constant` | Dampening constant of each particle. |
/// | $k_BT$ | `kb_temperature` | Product of temperature and boltzmann constant $k_B T$. |
///
/// # Position Update
/// Positions are numerically integrated.
/// We assume an overdamped context, meaning that we utilize the [NoVelocity] struct to
/// avoid updating the velocities and save computational resources.
/// The differential equation which is solved corresponds to a euclidean equation of motion
/// with dampening and a random part.
/// \\begin{align}
///     \frac{\partial}{\partial t}\vec{x} &= \vec{v}(t) + v_r(t)\vec{d}(t)\\\\
///     \frac{\partial}{\partial t}\vec{v} &= \frac{1}{m}\vec{F}(x, t) - \lambda\vec{v}(t)
/// \\end{align}
/// By choosing the `random_update_time` $t_r$ larger than the integration step, we can
/// resolve smaller timesteps to more accurately solve the equations.
/// This procedure is recommended.
/// In this scheme, both $v_r$ and $\vec{d}$ depend on time in the sence that their values
/// are changed at discrete time events.
/// The notation is slightly different to the usually used for stochastic processes.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Brownian<const D: usize> {
    /// Current position of the particle $\vec{x}$.
    pub pos: SVector<f64, D>,
    /// Diffusion constant $D$.
    pub diffusion_constant: f64,
    /// The product of temperature and boltzmann constant $k_B T$.
    pub kb_temperature: f64,
    /// The steps it takes for the particle to update its random vector
    pub particle_random_update_interval: usize,
    random_vector: SVector<f64, D>,
}

impl<const D: usize> Brownian<D> {
    /// Constructs a new [Brownian] mechanics model for the specified dimension.
    pub fn new(
        pos: SVector<f64, D>,
        diffusion_constant: f64,
        kb_temperature: f64,
        particle_random_update_interval: usize,
    ) -> Self {
        use num::Zero;
        Self {
            pos,
            diffusion_constant,
            kb_temperature,
            particle_random_update_interval,
            random_vector: SVector::<f64, D>::zero(),
        }
    }
}

impl<const D: usize> Mechanics<SVector<f64, D>, SVector<f64, D>, SVector<f64, D>> for Brownian<D> {
    fn pos(&self) -> SVector<f64, D> {
        self.pos
    }

    fn velocity(&self) -> SVector<f64, D> {
        use num::Zero;
        SVector::<f64, D>::zero()
    }

    fn set_pos(&mut self, pos: &SVector<f64, D>) {
        self.pos = *pos;
    }

    fn set_velocity(&mut self, _velocity: &SVector<f64, D>) {}

    fn set_random_variable(
        &mut self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: f64
    ) -> Result<Option<f64>, RngError> {
        use rand::Rng;

        let mut random_array = [0_f64; D];
        let distr = match rand_distr::Normal::new(0.0, dt) {
            Ok(e) => Ok(e),
            Err(e) => Err(RngError(format!("{e}"))),
        }?;
        rng
            .sample_iter(distr)
            .zip(random_array.iter_mut())
            .for_each(|(r, arr)| *arr = r);
        self.random_vector = random_array.into();
        Ok(Some(self.particle_random_update_interval as f64 * dt))
    }

    fn calculate_increment(
        &self,
        force: SVector<f64, D>
    ) -> Result<(SVector<f64, D>, SVector<f64, D>), CalcError> {
        use num::Zero;
        let dx = -self.diffusion_constant/self.kb_temperature*force
            + (2.0*self.diffusion_constant).sqrt()*self.random_vector;
        Ok((
            dx,
            SVector::<f64, D>::zero()
        ))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VertexMechanics2D<const D: usize> {
    points: nalgebra::SMatrix<f64, D, 2>,
    velocity: nalgebra::SMatrix<f64, D, 2>,
    cell_boundary_lengths: nalgebra::SVector<f64, D>,
    spring_tensions: nalgebra::SVector<f64, D>,
    cell_area: f64,
    central_pressure: f64,
    dampening_constant: f64,
}

pub type VertexVector2<const D: usize> = nalgebra::SMatrix<f64, D, 2>;
pub type VertexConnections2<const D: usize> = nalgebra::SVector<f64, D>;

impl<const D: usize> VertexMechanics2D<D> {
    /// Creates a new vertex model in equilibrium conditions.
    ///
    /// The specified parameters are then used to carefully calculate relating properties of the model.
    /// We outline the formulas used.
    /// Given the number of vertices \\(N\\) in our model (specified by the const generic argument of the [VertexMechanics2D] struct),
    /// the resulting average angle when all nodes are equally distributed is
    /// \\[
    ///     \Delta\varphi = \frac{2\pi}{N}
    /// \\]
    /// Given the total area of the cell ([regular polygon](https://en.wikipedia.org/wiki/Regular_polygon)) \\(A\\), we can calculate the distance from the center point to the individual vertices by inverting
    /// \\[
    ///     A = r^2 \sin\left(\frac{\pi}{N}\right)\cos\left(\frac{\pi}{N}\right).
    /// \\]
    /// This formula can be derived by elementary geometric arguments.
    /// We can then generate the points \\(\vec{p}\_i\\) which make up the cell-boundary by using
    /// \\[
    ///     \vec{p}\_i = \vec{p}\_{mid} + r(\cos(i \Delta\varphi), \sin(i\Delta\varphi))^T.
    /// \\]
    /// From these points, their distance is calculated and passed as the individual boundary lengths.
    /// When randomization is turned on, these points will be slightly randomized in their radius and angle which might lead to non-equilibrium configurations.
    /// Pressure, dampening and spring tensions are not impacted by randomization.
    pub fn new(
        middle: SVector<f64, 2>,
        cell_area: f64,
        rotation_angle: f64,
        spring_tensions: f64,
        central_pressure: f64,
        dampening_constant: f64,
        randomize: Option<(f64, rand_chacha::ChaCha8Rng)>,
    ) -> Self {
        use rand::Rng;
        // Restrict the randomize variable between 0 and 1
        let r = match randomize {
            Some((rand, _)) => rand.clamp(0.0, 1.0),
            _ => 0.0,
        };
        let rng = || -> f64 {
            match randomize {
                Some((_, mut rng)) => rng.gen_range(0.0..1.0),
                None => 0.0,
            }
        };
        // Randomize the overall rotation angle
        let rotation_angle = (1.0 - r * rng.clone()()) * rotation_angle;
        // Calculate the angle fraction used to determine the points of the polygon
        let angle_fraction = std::f64::consts::PI / D as f64;
        // Calculate the radius from cell area
        let radius = (cell_area / D as f64 / angle_fraction.sin() / angle_fraction.cos()).sqrt();
        // TODO this needs to be calculated again
        let points = VertexVector2::<D>::from_row_iterator(
            (0..D)
                .map(|n| {
                    let angle = rotation_angle
                        + 2.0 * angle_fraction * n as f64 * (1.0 - r * rng.clone()());
                    let radius_modified = radius * (1.0 + 0.5 * r * (1.0 - rng.clone()()));
                    [
                        middle.x + radius_modified * angle.cos(),
                        middle.y + radius_modified * angle.sin(),
                    ]
                    .into_iter()
                })
                .flatten(),
        );
        // Randomize the boundary lengths
        let cell_boundary_lengths = VertexConnections2::<D>::from_iterator(
            points
                .row_iter()
                .circular_tuple_windows()
                .map(|(p1, p2)| (p1 - p2).norm()),
        );
        VertexMechanics2D {
            points,
            velocity: VertexVector2::<D>::zeros(),
            cell_boundary_lengths,
            spring_tensions: VertexConnections2::<D>::from_element(spring_tensions),
            cell_area,
            central_pressure,
            dampening_constant,
        }
    }

    pub fn get_cell_area(&self) -> f64 {
        self.cell_area
    }

    pub fn set_cell_area(&mut self, cell_area: f64) {
        // Calculate the relative difference to current area
        match self.cell_area {
            a if a == 0.0 => {
                let new_interaction_parameters = Self::new(
                    self.points
                        .row_iter()
                        .map(|v| v.transpose())
                        .sum::<nalgebra::Vector2<f64>>(),
                    cell_area,
                    0.0,
                    self.spring_tensions.sum() / self.spring_tensions.len() as f64,
                    self.central_pressure,
                    self.dampening_constant,
                    None,
                );
                *self = new_interaction_parameters;
            }
            _ => {
                let relative_length_difference = (cell_area.abs() / self.cell_area.abs()).sqrt();
                // Calculate the new length of the cell boundary lengths
                self.cell_boundary_lengths
                    .iter_mut()
                    .for_each(|length| *length *= relative_length_difference);
                self.cell_area = cell_area;
            }
        };
    }
}

impl VertexMechanics2D<4> {
    pub fn fill_rectangle(
        cell_area: f64,
        spring_tensions: f64,
        central_pressure: f64,
        dampening_constant: f64,
        rectangle: [SVector<f64, 2>; 2],
    ) -> Vec<Self> {
        let cell_side_length: f64 = cell_area.sqrt();
        let cell_side_length_padded: f64 = cell_side_length * 1.04;

        let number_of_cells_x: u64 =
            ((rectangle[1].x - rectangle[0].x) / cell_side_length_padded).floor() as u64;
        let number_of_cells_y: u64 =
            ((rectangle[1].y - rectangle[0].y) / cell_side_length_padded).floor() as u64;

        let start_x: f64 = rectangle[0].x
            + 0.5
                * ((rectangle[1].x - rectangle[0].x)
                    - number_of_cells_x as f64 * cell_side_length_padded);
        let start_y: f64 = rectangle[0].y
            + 0.5
                * ((rectangle[1].y - rectangle[0].y)
                    - number_of_cells_y as f64 * cell_side_length_padded);

        use itertools::iproduct;
        let filled_rectangle = iproduct!(0..number_of_cells_x, 0..number_of_cells_y)
            .map(|(i, j)| {
                let corner = (
                    start_x + (i as f64) * cell_side_length_padded,
                    start_y + (j as f64) * cell_side_length_padded,
                );

                let points = VertexVector2::<4>::from_row_iterator([
                    corner.0,
                    corner.1,
                    corner.0 + cell_side_length,
                    corner.1,
                    corner.0 + cell_side_length,
                    corner.1 + cell_side_length,
                    corner.0,
                    corner.1 + cell_side_length,
                ]);
                let cell_boundary_lengths =
                    VertexConnections2::<4>::from_iterator((0..2 * 4).map(|_| cell_side_length));

                VertexMechanics2D {
                    points,
                    velocity: VertexVector2::<4>::zeros(),
                    cell_boundary_lengths,
                    spring_tensions: VertexConnections2::<4>::from_element(spring_tensions),
                    cell_area,
                    central_pressure,
                    dampening_constant,
                }
            })
            .collect::<Vec<_>>();

        filled_rectangle
    }
}

impl<const D: usize> Mechanics<VertexVector2<D>, VertexVector2<D>, VertexVector2<D>>
    for VertexMechanics2D<D>
{
    fn pos(&self) -> VertexVector2<D> {
        self.points.clone()
    }

    fn velocity(&self) -> VertexVector2<D> {
        self.velocity.clone()
    }

    fn set_pos(&mut self, pos: &VertexVector2<D>) {
        self.points = pos.clone();
    }

    fn set_velocity(&mut self, velocity: &VertexVector2<D>) {
        self.velocity = velocity.clone();
    }

    fn calculate_increment(
        &self,
        force: VertexVector2<D>,
    ) -> Result<(VertexVector2<D>, VertexVector2<D>), CalcError> {
        // Calculate the total internal force
        let middle = self.points.row_sum() / self.points.shape().0 as f64;
        let current_ara: f64 = 0.5_f64
            * self
                .points
                .row_iter()
                .circular_tuple_windows()
                .map(|(p1, p2)| p1.transpose().perp(&p2.transpose()))
                .sum::<f64>();

        let mut internal_force = self.points.clone() * 0.0;
        for (index, (point_1, point_2, point_3)) in self
            .points
            .row_iter()
            .circular_tuple_windows::<(_, _, _)>()
            .enumerate()
        {
            let tension_12 = self.spring_tensions[index];
            let tension_23 = self.spring_tensions[(index + 1) % self.spring_tensions.len()];
            let mut force_2 = internal_force.row_mut((index + 1) % self.points.shape().0);

            // Calculate forces arising from springs between points
            let p_21 = point_2 - point_1;
            let p_23 = point_2 - point_3;
            let force1 =
                p_21.normalize() * (self.cell_boundary_lengths[index] - p_21.norm()) * tension_12;
            let force2 = p_23.normalize()
                * (self.cell_boundary_lengths[(index + 1) % self.cell_boundary_lengths.len()]
                    - p_23.norm())
                * tension_23;

            // Calculate force arising from internal pressure
            let middle_to_point_2 = point_2 - middle;
            let force3 = middle_to_point_2.normalize()
                * (self.cell_area - current_ara)
                * self.central_pressure;

            // Combine forces
            force_2 += force1 + force2 + force3;
        }
        let dx = self.velocity.clone();
        let dv = force + internal_force - self.dampening_constant * self.velocity.clone();
        Ok((dx, dv))
    }
}
