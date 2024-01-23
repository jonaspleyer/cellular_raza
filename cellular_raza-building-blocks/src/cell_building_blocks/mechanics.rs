use cellular_raza_concepts::{CalcError, Mechanics, RngError};

use itertools::Itertools;
use nalgebra::SVector;

use serde::{Deserialize, Serialize};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

macro_rules! implement_newton_damped_mechanics(
    ($struct_name:ident, $d:literal) => {implement_newton_damped_mechanics!($struct_name, $d, f64);};
    ($struct_name:ident, $d:literal, $float_type:ty) => {
        /// Simple newtonian dynamics governed by mass and damping.
        ///
        /// The equation of motion is given by
        /// \\begin{equation}
        ///     m \ddot{\vec{x}} = \vec{F} - \lambda \dot{\vec{x}}
        /// \\end{equation}
        /// where $\vec{F}$ is calculated by the
        /// [Interaction](cellular_raza_concepts::Interaction) trait.
        /// The parameter $m$ describes the mass of the object while $\lambda$ is
        /// the damping constant.
        /// If the cell is growing, we need to increase the mass $m$.
        /// By interacting with the outside world, we can adapt $\lambda$ to external values
        /// although this is rarely desirable.
        /// Both operations need to be implemented by other concepts such as
        /// [Cycle](cellular_raza_concepts::Cycle).
        #[derive(Clone, Debug, Serialize, Deserialize)]
        #[cfg_attr(feature = "pyo3", pyclass)]
        pub struct $struct_name {
            /// Current position $\vec{x}$ given by a vector of dimension `D`.
            pub pos: SVector<$float_type, $d>,
            /// Current velocity $\dot{\vec{x}}$ given by a vector of dimension `D`.
            pub vel: SVector<$float_type, $d>,
            /// Damping constant $\lambda$.
            pub damping_constant: $float_type,
            /// Mass $m$ of the object.
            pub mass: $float_type,
        }

        #[cfg(feature = "pyo3")]
        #[pymethods]
        impl $struct_name {
            #[doc = "Create a new "]
            #[doc = stringify!($struct_name)]
            /// from position, velocity, damping constant and mass
            #[new]
            pub fn new(
                pos: [$float_type; $d],
                vel: [$float_type; $d],
                damping_constant: $float_type,
                mass: $float_type,
            ) -> Self {
                Self {
                    pos: pos.into(),
                    vel: vel.into(),
                    damping_constant,
                    mass,
                }
            }

            #[getter]
            fn get_pos(&self) -> [$float_type; $d] {
                self.pos.into()
            }

            #[getter]
            fn get_vel(&self) -> [$float_type; $d] {
                self.vel.into()
            }

            #[getter]
            fn get_damping_constant(&self) -> $float_type {
                self.damping_constant
            }

            #[getter]
            fn get_mass(&self) -> $float_type {
                self.mass
            }

            #[setter]
            fn set_pos(&mut self, pos: [$float_type; $d]) {
                self.pos = pos.into();
            }

            #[setter]
            fn set_vel(&mut self, vel: [$float_type; $d]) {
                self.vel = vel.into();
            }

            #[setter]
            fn set_damping_constant(&mut self, damping_constant: $float_type) {
                self.damping_constant = damping_constant;
            }

            #[setter]
            fn set_mass(&mut self, mass: $float_type) {
                self.mass = mass;
            }
        }

        impl Mechanics<SVector<$float_type, $d>, SVector<$float_type, $d>, SVector<$float_type, $d>, $float_type> for $struct_name
        {
            fn pos(&self) -> SVector<$float_type, $d> {
                self.pos
            }

            fn velocity(&self) -> SVector<$float_type, $d> {
                self.vel
            }

            fn set_pos(&mut self, p: &SVector<$float_type, $d>) {
                self.pos = *p;
            }

            fn set_velocity(&mut self, v: &SVector<$float_type, $d>) {
                self.vel = *v;
            }

            fn calculate_increment(
                &self,
                force: SVector<$float_type, $d>,
            ) -> Result<(SVector<$float_type, $d>, SVector<$float_type, $d>), CalcError> {
                let dx = self.vel;
                let dv = force / self.mass - self.damping_constant * self.vel;
                Ok((dx, dv))
            }
        }
    }
);

implement_newton_damped_mechanics!(NewtonDamped1D, 1);
implement_newton_damped_mechanics!(NewtonDamped2D, 2);
implement_newton_damped_mechanics!(NewtonDamped3D, 3);

implement_newton_damped_mechanics!(NewtonDamped1DF32, 1, f32);
implement_newton_damped_mechanics!(NewtonDamped2DF32, 2, f32);
implement_newton_damped_mechanics!(NewtonDamped3DF32, 3, f32);

/// An empty struct to signalize that no velocity needs to be updated.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "pyo3", pyclass)]
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

fn generate_random_vector<const D: usize>(
    rng: &mut rand_chacha::ChaCha8Rng,
    distribution_width: f64,
) -> Result<SVector<f64, D>, RngError> {
    let distr = match rand_distr::Normal::new(0.0, distribution_width) {
        Ok(e) => Ok(e),
        Err(e) => Err(RngError(format!("{e}"))),
    }?;
    let random_dir = SVector::<f64, D>::from_distribution(&distr, rng);
    Ok(random_dir)
}

macro_rules! implement_brownian_mechanis(
    ($struct_name:ident, $d:literal) => {
        /// Brownian motion of particles represented by a spherical potential in arbitrary dimension.
        ///
        /// # Parameters
        /// | Symbol | Parameter | Description |
        /// | --- | --- | --- |
        /// | $\vec{x}$ | `pos` | Position of the particle. |
        /// | $D$ | `diffusion_constant` | Dampening constant of each particle. |
        /// | $k_BT$ | `kb_temperature` | Product of temperature and boltzmann constant $k_B T$. |
        /// | $\Delta t$ | 'update_interval` | A multiple of the integration constant `dt` which determines how often a new random direction for movement is chosen. |
        ///
        /// # Position Update
        /// We integrate the standard brownian motion stochastic differential equation.
        /// \\begin{equation}
        ///     \dot{\vec{x}} = -\frac{D}{k_B T}\nabla V(x) + \sqrt{2D}R(t)
        /// \\end{equation}
        /// The variable `update_interval` $n_t$ determines how often a new random direction for travel
        /// is generated.
        /// The new random vector is then also sampled by a distribution with greater width.
        /// If we choose this value larger than one, we can
        /// resolve smaller timesteps to more accurately solve the equations.
        #[derive(Clone, Debug, Deserialize, Serialize)]
        #[cfg_attr(feature = "pyo3", pyclass)]
        pub struct $struct_name {
            /// Current position of the particle $\vec{x}$.
            pub pos: SVector<f64, $d>,
            /// Diffusion constant $D$.
            pub diffusion_constant: f64,
            /// The product of temperature and boltzmann constant $k_B T$.
            pub kb_temperature: f64,
            /// The steps it takes for the particle to update its random vector
            pub update_interval: usize,
            random_vector: SVector<f64, $d>,
        }

        #[cfg(feature = "pyo3")]
        #[cfg_attr(feature = "pyo3", pymethods)]
        impl $struct_name {
            /// Constructs a new [Brownian] mechanics model for the specified dimension.
            #[new]
            pub fn new(
                pos: [f64; $d],
                diffusion_constant: f64,
                kb_temperature: f64,
                update_interval: usize,
            ) -> Self {
                use num::Zero;
                Self {
                    pos: pos.into(),
                    diffusion_constant,
                    kb_temperature,
                    update_interval,
                    random_vector: SVector::<f64, $d>::zero(),
                }
            }
        }

        // TODO use NoVelocity struct
        impl Mechanics<SVector<f64, $d>, SVector<f64, $d>, SVector<f64, $d>> for $struct_name {
            fn pos(&self) -> SVector<f64, $d> {
                self.pos
            }

            fn velocity(&self) -> SVector<f64, $d> {
                use num::Zero;
                SVector::<f64, $d>::zero()
            }

            fn set_pos(&mut self, pos: &SVector<f64, $d>) {
                self.pos = *pos;
            }

            fn set_velocity(&mut self, _velocity: &SVector<f64, $d>) {}

            fn set_random_variable(
                &mut self,
                rng: &mut rand_chacha::ChaCha8Rng,
                dt: f64,
            ) -> Result<Option<f64>, RngError> {
                self.random_vector = generate_random_vector(rng, self.update_interval as f64 * dt)?;
                Ok(Some(self.update_interval as f64 * dt))
            }

            fn calculate_increment(
                &self,
                force: SVector<f64, $d>,
            ) -> Result<(SVector<f64, $d>, SVector<f64, $d>), CalcError> {
                use num::Zero;
                let dx = self.diffusion_constant / self.kb_temperature * force
                    + 2_f64.sqrt() * self.diffusion_constant.sqrt() * self.random_vector;
                Ok((dx, SVector::<f64, $d>::zero()))
            }
        }
    }
);

implement_brownian_mechanis!(Brownian1D, 1);
implement_brownian_mechanis!(Brownian2D, 2);
implement_brownian_mechanis!(Brownian3D, 3);

macro_rules! define_langevin_nd(
    ($struct_name:ident, $d:literal) => {
        /// Langevin dynamics in
        #[doc = stringify!($d)]
        /// dimensions
        #[cfg_attr(feature = "pyo3", pyclass)]
        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct $struct_name {
            /// Current position
            pub pos: SVector<f64, $d>,
            /// Current velocity
            pub vel: SVector<f64, $d>,
            /// Mass of the object
            pub mass: f64,
            /// Damping constant
            pub damping: f64,
            /// Product of Boltzmann constant and temperature
            pub kb_temperature: f64,
            /// Number of steps to do before updating the internal random vector again
            pub update_interval: usize,
            random_vector: SVector<f64, $d>,
        }

        impl Mechanics<SVector<f64, $d>, SVector<f64, $d>, SVector<f64, $d>> for $struct_name {
            fn pos(&self) -> SVector<f64, $d> {
                self.pos
            }

            fn set_pos(&mut self, pos: &SVector<f64, $d>) {
                self.pos = *pos;
            }

            fn velocity(&self) -> SVector<f64, $d> {
                self.vel
            }

            fn set_velocity(&mut self, velocity: &SVector<f64, $d>) {
                self.vel = *velocity;
            }

            fn set_random_variable(
                &mut self,
                rng: &mut rand_chacha::ChaCha8Rng,
                dt: f64,
            ) -> Result<Option<f64>, RngError> {
                self.random_vector = generate_random_vector(rng, 2.0 * self.kb_temperature)?;// TODO * self.update_interval as f64 * dt)?;
                Ok(Some(self.update_interval as f64 * dt))
            }

            fn calculate_increment(
                &self,
                force: SVector<f64, $d>,
            ) -> Result<(SVector<f64, $d>, SVector<f64, $d>), CalcError> {
                let dx = self.vel;
                let dv =
                    -self.damping / self.mass * self.vel + 1.0 / self.mass * (self.random_vector + force);
                Ok((dx, dv))
            }
        }

        #[cfg(feature = "pyo3")]
        #[pymethods]
        impl $struct_name {
            /// Creates a new [Langevin] struct from position, velocity, mass, damping,
            /// kb_temperature and the update interval of the mechanics aspect.
            #[new]
            pub fn new(
                pos: [f64; $d],
                vel: [f64; $d],
                mass: f64,
                damping: f64,
                kb_temperature: f64,
                update_interval: usize,
            ) -> Self {
                Self {
                    pos: pos.into(),
                    vel: vel.into(),
                    mass,
                    damping,
                    kb_temperature,
                    update_interval,
                    random_vector: [0.0; $d].into(),
                }
            }

            #[getter(pos)]
            /// Get position of object
            pub fn get_position(&self) -> [f64; $d] {
                self.pos.into()
            }

            #[setter(pos)]
            /// Set position of object
            pub fn set_position(&mut self, pos: [f64; $d]) {
                self.pos = pos.into();
            }

            #[getter(damping)]
            /// Get damping constant of object
            pub fn get_damping(&self) -> f64 {
                self.damping
            }

            #[setter(damping)]
            /// Set the damping constant of the object
            pub fn set_damping(&mut self, damping: f64) {
                self.damping = damping;
            }

            #[getter(mass)]
            /// Get mass of the object
            pub fn get_mass(&self) -> f64 {
                self.mass
            }

            #[setter(mass)]
            /// Set mass of the object
            pub fn set_mass(&mut self, mass: f64) {
                self.mass = mass;
            }

            #[getter(kb_temperature)]
            /// Get the product of Boltzmann constant and temperature
            pub fn get_kb_temperature(&self) -> f64 {
                self.kb_temperature
            }

            #[setter(kb_temperature)]
            /// Define product of Boltzmann constant and temperature
            pub fn set_kb_temperature(&mut self, kb_temperature: f64) {
                self.kb_temperature = kb_temperature;
            }

            #[getter(update_interval)]
            /// Get the update interval after which a new random vector is chosen
            pub fn get_update_interval(&self) -> usize {
                self.update_interval
            }

            #[setter(update_interval)]
            /// Sets the update interval
            pub fn set_update_interval(&mut self, update_interval: usize) {
                self.update_interval = update_interval;
            }

            fn __repr__(&self) -> String {
                format!("{self:#?}")
            }
        }
    }
);

define_langevin_nd!(Langevin1D, 1);
define_langevin_nd!(Langevin2D, 2);
define_langevin_nd!(Langevin3D, 3);

/// Mechanics model which represents cells as vertices with edges between them.
///
/// The vertices are attached to each other with springs and a given length between each
/// vertex.
/// Furthermore, we define a central pressure that acts when the total cell area is greater
/// or smaller than the desired one.
/// Each vertex is damped individually by the same constant.
// TODO include more formulas for this model
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

/// Alias for the spatial representation of a cell
pub type VertexVector2<const D: usize> = nalgebra::SMatrix<f64, D, 2>;
/// Alias for a connection quantity between two individual vertices
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

    /// Obtain current cell area
    pub fn get_cell_area(&self) -> f64 {
        self.cell_area
    }

    /// Set the current cell area
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
    /// Fill a specified rectangle with cells of 4 vertices
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
