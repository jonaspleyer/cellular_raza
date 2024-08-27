use cellular_raza_concepts::{CalcError, Mechanics, RngError};

use itertools::Itertools;
use nalgebra::{SMatrix, SVector};

use serde::{Deserialize, Serialize};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

macro_rules! implement_newton_damped_mechanics(
    (
        $struct_name:ident,
        $d:literal
    ) => {
        implement_newton_damped_mechanics!($struct_name, $d, f64);
    };
    (
        $struct_name:ident,
        $d:literal,
        $float_type:ty
    ) => {
        /// Newtonian dynamics governed by mass and damping.
        ///
        /// # Parameters
        /// | Symbol | Parameter | Description |
        /// | --- | --- | --- |
        /// | $\vec{x}$ | `pos` | Position of the particle. |
        /// | $\dot{\vec{x}}$ | `vel` | Velocity of the particle. |
        /// | $\lambda$ | `damping` | Damping constant |
        /// | $m$ | `mass` | Mass of the particle. |
        ///
        /// # Equations
        /// The equation of motion is given by
        /// \\begin{equation}
        ///     m \ddot{\vec{x}} = \vec{F} - \lambda \dot{\vec{x}}
        /// \\end{equation}
        /// where $\vec{F}$ is the force as calculated by the
        /// [Interaction](cellular_raza_concepts::Interaction) trait.
        ///
        /// # Comments
        /// If the cell is growing, we need to increase the mass $m$.
        /// By interacting with the outside world, we can adapt $\lambda$ to external values
        /// although this is rarely desirable.
        /// Both operations need to be implemented by other concepts such as
        /// [Cycle](cellular_raza_concepts::Cycle).
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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

        impl $struct_name {
            #[doc = "Create a new "]
            #[doc = stringify!($struct_name)]
            /// from position, velocity, damping constant and mass
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

        }

        #[cfg(feature = "pyo3")]
        #[pymethods]
        #[cfg_attr(docsrs, doc(cfg(feature = "pyo3")))]
        impl $struct_name {
            #[new]
            fn _new(
                pos: [$float_type; $d],
                vel: [$float_type; $d],
                damping_constant: $float_type,
                mass: $float_type,
            ) -> Self {
                Self::new(pos, vel, damping_constant, mass)
            }


            /// [pyo3] getter for `pos`
            #[getter]
            pub fn get_pos(&self) -> [$float_type; $d] {
                self.pos.into()
            }

            /// [pyo3] getter for `vel`
            #[getter]
            pub fn get_vel(&self) -> [$float_type; $d] {
                self.vel.into()
            }

            /// [pyo3] getter for `damping_constant`
            #[getter]
            pub fn get_damping_constant(&self) -> $float_type {
                self.damping_constant
            }

            /// [pyo3] getter for `mass`
            #[getter]
            pub fn get_mass(&self) -> $float_type {
                self.mass
            }

            /// [pyo3] setter for `pos`
            #[setter]
            pub fn set_pos(&mut self, pos: [$float_type; $d]) {
                self.pos = pos.into();
            }

            /// [pyo3] setter for `vel`
            #[setter]
            pub fn set_vel(&mut self, vel: [$float_type; $d]) {
                self.vel = vel.into();
            }

            /// [pyo3] setter for `damping_constant`
            #[setter]
            pub fn set_damping_constant(&mut self, damping_constant: $float_type) {
                self.damping_constant = damping_constant;
            }

            /// [pyo3] setter for `mass`
            #[setter]
            pub fn set_mass(&mut self, mass: $float_type) {
                self.mass = mass;
            }
        }

        impl Mechanics<
            SVector<$float_type, $d>,
            SVector<$float_type, $d>,
            SVector<$float_type, $d>,
            $float_type
        > for $struct_name
        {
            fn get_random_contribution(
                &self,
                _: &mut rand_chacha::ChaCha8Rng,
                _dt: $float_type,
            ) -> Result<(SVector<$float_type, $d>, SVector<$float_type, $d>), RngError> {
                Ok((num::Zero::zero(), num::Zero::zero()))
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

        impl cellular_raza_concepts::Position<SVector<$float_type, $d>> for $struct_name {
            fn pos(&self) -> SVector<$float_type, $d> {
                self.pos
            }

            fn set_pos(&mut self, pos: &SVector<$float_type, $d>) {
                self.pos = *pos;
            }
        }

        impl cellular_raza_concepts::Velocity<SVector<$float_type, $d>> for $struct_name {
            fn velocity(&self) -> SVector<$float_type, $d> {
                self.vel
            }

            fn set_velocity(&mut self, velocity: &SVector<$float_type, $d>) {
                self.vel = *velocity;
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

/// Generate a vector corresponding to a wiener process.
///
/// This function calculates a statically sized random vector with dimension `D`.
/// It uses a [rand_distr::StandardNormal] distribution and divides the result by `dt` such that
/// the correct incremental wiener process is obtained.
pub fn wiener_process<F, const D: usize>(
    rng: &mut rand_chacha::ChaCha8Rng,
    dt: F,
) -> Result<SVector<F, D>, RngError>
where
    F: core::ops::DivAssign + nalgebra::Scalar + num::Float,
    rand_distr::StandardNormal: rand_distr::Distribution<F>,
{
    let std_dev = dt.sqrt();
    let distr = match rand_distr::Normal::new(F::zero(), std_dev) {
        Ok(e) => Ok(e),
        Err(e) => Err(RngError(format!("{e}"))),
    }?;
    let random_dir = SVector::<F, D>::from_distribution(&distr, rng);
    Ok(random_dir / dt)
}

macro_rules! implement_brownian_mechanics(
    ($struct_name:ident, $d:literal, $float_type:ty) => {
        /// Brownian motion of particles
        ///
        /// # Parameters & Variables
        /// | Symbol | Struct Field | Description |
        /// | --- | --- | --- |
        /// | $D$ | `diffusion_constant` | Damping constant of each particle. |
        /// | $k_BT$ | `kb_temperature` | Product of temperature $T$ and Boltzmann constant $k_B$. |
        /// | | | |
        /// | $\vec{x}$ | `pos` | Position of the particle. |
        /// | $R(t)$ | (automatically generated) | Gaussian process |
        ///
        /// # Equations
        /// We integrate the standard brownian motion stochastic differential equation.
        /// \\begin{equation}
        ///     \dot{\vec{x}} = -\frac{D}{k_B T}\nabla V(x) + \sqrt{2D}R(t)
        /// \\end{equation}
        /// The new random vector is then also sampled by a distribution with greater width.
        /// If we choose this value larger than one, we can
        /// resolve smaller timesteps to more accurately solve the equations.
        #[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
        #[cfg_attr(feature = "pyo3", pyclass)]
        pub struct $struct_name {
            /// Current position of the particle $\vec{x}$.
            pub pos: SVector<$float_type, $d>,
            /// Diffusion constant $D$.
            pub diffusion_constant: $float_type,
            /// The product of temperature and boltzmann constant $k_B T$.
            pub kb_temperature: $float_type,
        }

        impl $struct_name {
            /// Constructs a new
            #[doc = concat!("[", stringify!($struct_name), "]")]
            pub fn new(
                pos: [$float_type; $d],
                diffusion_constant: $float_type,
                kb_temperature: $float_type,
            ) -> Self {
                Self {
                    pos: pos.into(),
                    diffusion_constant,
                    kb_temperature,
                }
            }
        }

        #[cfg(feature = "pyo3")]
        #[pymethods]
        #[cfg_attr(docsrs, doc(cfg(feature = "pyo3")))]
        impl $struct_name {
            #[new]
            fn _new(
                pos: [$float_type; $d],
                diffusion_constant: $float_type,
                kb_temperature: $float_type,
            ) -> Self {
                Self::new(pos, diffusion_constant, kb_temperature)
            }


            /// [pyo3] setter for `pos`
            #[setter]
            pub fn set_pos(&mut self, pos: [$float_type; $d]) {
                self.pos = pos.into();
            }

            /// [pyo3] setter for `diffusion_constant`
            #[setter]
            pub fn set_diffusion_constant(&mut self, diffusion_constant: $float_type) {
                self.diffusion_constant = diffusion_constant;
            }

            /// [pyo3] setter for `kb_temperature`
            #[setter]
            pub fn set_kb_temperature(&mut self, kb_temperature: $float_type) {
                self.kb_temperature = kb_temperature;
            }

            /// [pyo3] getter for `pos`
            #[getter]
            pub fn get_pos(&self) -> [$float_type; $d] {
                self.pos.into()
            }

            /// [pyo3] getter for `diffusion_constant`
            #[getter]
            pub fn get_diffusion_constant(&self) -> $float_type {
                self.diffusion_constant
            }

            /// [pyo3] getter for `kb_temperature`
            #[getter]
            pub fn get_kb_temperature(&self) -> $float_type {
                self.kb_temperature
            }
        }

        impl Mechanics<
            SVector<$float_type, $d>,
            SVector<$float_type, $d>,
            SVector<$float_type, $d>,
            $float_type
        > for $struct_name {
            fn get_random_contribution(
                &self,
                rng: &mut rand_chacha::ChaCha8Rng,
                dt: $float_type,
            ) -> Result<(SVector<$float_type, $d>, SVector<$float_type, $d>), RngError> {
                let dpos = (2.0 as $float_type * self.diffusion_constant).sqrt()
                    * wiener_process(
                    rng,
                    dt
                )?;
                let dvel = SVector::<$float_type, $d>::zeros();
                Ok((dpos, dvel))
            }

            fn calculate_increment(
                &self,
                force: SVector<$float_type, $d>,
            ) -> Result<(SVector<$float_type, $d>, SVector<$float_type, $d>), CalcError> {
                use num::Zero;
                let dx = self.diffusion_constant / self.kb_temperature * force;
                Ok((dx, SVector::<$float_type, $d>::zero()))
            }
        }

        impl cellular_raza_concepts::Position<SVector<$float_type, $d>> for $struct_name {
            fn pos(&self) -> SVector<$float_type, $d> {
                self.pos
            }

            fn set_pos(&mut self, pos: &SVector<$float_type, $d>) {
                self.pos = *pos;
            }

        }

        impl cellular_raza_concepts::Velocity<SVector<$float_type, $d>> for $struct_name {
            fn velocity(&self) -> SVector<$float_type, $d> {
                use num::Zero;
                SVector::<$float_type, $d>::zero()
            }

            fn set_velocity(&mut self, _velocity: &SVector<$float_type, $d>) {}
        }
    }
);

implement_brownian_mechanics!(Brownian1D, 1, f64);
implement_brownian_mechanics!(Brownian2D, 2, f64);
implement_brownian_mechanics!(Brownian3D, 3, f64);
implement_brownian_mechanics!(Brownian1DF32, 1, f32);
implement_brownian_mechanics!(Brownian2DF32, 2, f32);
implement_brownian_mechanics!(Brownian3DF32, 3, f32);

macro_rules! define_langevin_nd(
    ($struct_name:ident, $d:literal, $float_type:ident) => {
        /// Langevin dynamics
        ///
        /// # Parameters & Variables
        /// | Symbol | Struct Field | Description |
        /// |:---:| --- | --- |
        /// | $M$ | `mass` | Mass of the particle. |
        /// | $\gamma$ | `damping` | Damping constant |
        /// | $k_BT$ | `kb_temperature` | Product of temperature $T$ and Boltzmann constant $k_B$. |
        /// | | | |
        /// | $\vec{X}$ | `pos` | Position of the particle. |
        /// | $\dot{\vec{X}}$ | `vel` | Velocity of the particle. |
        /// | $R(t)$ | (automatically generated) | Gaussian process |
        ///
        /// # Equations
        ///
        /// \\begin{equation}
        ///     M \ddot{\mathbf{X}} = - \mathbf{\nabla} U(\mathbf{X}) - \gamma M\dot{\mathbf{X}} + \sqrt{2 M \gamma k_{\rm B} T}\mathbf{R}(t)
        /// \\end{equation}
        #[cfg_attr(feature = "pyo3", pyclass)]
        #[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
        pub struct $struct_name {
            /// Current position
            pub pos: SVector<$float_type, $d>,
            /// Current velocity
            pub vel: SVector<$float_type, $d>,
            /// Mass of the object
            pub mass: $float_type,
            /// Damping constant
            pub damping: $float_type,
            /// Product of Boltzmann constant and temperature
            pub kb_temperature: $float_type,
        }

        impl Mechanics<
            SVector<$float_type, $d>,
            SVector<$float_type, $d>,
            SVector<$float_type, $d>,
            $float_type
        > for $struct_name {
            fn get_random_contribution(
                &self,
                rng: &mut rand_chacha::ChaCha8Rng,
                dt: $float_type,
            ) -> Result<(SVector<$float_type, $d>, SVector<$float_type, $d>), RngError> {
                let dpos = (
                    2.0 as $float_type
                    * self.damping
                    * self.kb_temperature
                    / self.mass
                ).sqrt() * wiener_process(
                    rng,
                    dt
                )?;
                let dvel = SVector::<$float_type, $d>::zeros();
                Ok((dpos, dvel))
            }

            fn calculate_increment(
                &self,
                force: SVector<$float_type, $d>,
            ) -> Result<(SVector<$float_type, $d>, SVector<$float_type, $d>), CalcError> {
                let dx = self.vel;
                let dv1 =
                    - 1.0 as $float_type / self.mass * force;
                let dv2 =
                    - self.damping * self.vel;
                let dv = dv1 + dv2;
                Ok((dx, dv))
            }
        }

        impl cellular_raza_concepts::Position<SVector<$float_type, $d>> for $struct_name {
            fn pos(&self) -> SVector<$float_type, $d> {
                self.pos
            }

            fn set_pos(&mut self, pos: &SVector<$float_type, $d>) {
                self.pos = *pos;
            }
        }

        impl cellular_raza_concepts::Velocity<SVector<$float_type, $d>> for $struct_name {
            fn velocity(&self) -> SVector<$float_type, $d> {
                self.vel
            }

            fn set_velocity(&mut self, velocity: &SVector<$float_type, $d>) {
                self.vel = *velocity;
            }
        }

        impl $struct_name {
            /// Constructs a new
            #[doc = concat!("[", stringify!($struct_name), "]")]
            pub fn new(
                pos: [$float_type; $d],
                vel: [$float_type; $d],
                mass: $float_type,
                damping: $float_type,
                kb_temperature: $float_type,
            ) -> Self {
                Self {
                    pos: pos.into(),
                    vel: vel.into(),
                    mass,
                    damping,
                    kb_temperature,
                }
            }
        }

        #[cfg(feature = "pyo3")]
        #[pymethods]
        #[cfg_attr(docsrs, doc(cfg(feature = "pyo3")))]
        impl $struct_name {
            /// Creates a new [
            #[doc = stringify!($struct_name)]
            /// ] struct from position, velocity, mass, damping,
            /// kb_temperature and the update interval of the mechanics aspect.
            #[new]
            fn _new(
                pos: [$float_type; $d],
                vel: [$float_type; $d],
                mass: $float_type,
                damping: $float_type,
                kb_temperature: $float_type,
            ) -> Self {
                Self {
                    pos: pos.into(),
                    vel: vel.into(),
                    mass,
                    damping,
                    kb_temperature,
                }
            }

            #[getter(pos)]
            /// [pyo3] getter for `position`
            pub fn get_position(&self) -> [$float_type; $d] {
                self.pos.into()
            }

            #[setter(pos)]
            /// [pyo3] setter for `position`
            pub fn set_position(&mut self, pos: [$float_type; $d]) {
                self.pos = pos.into();
            }

            #[getter(damping)]
            /// [pyo3] getter for `damping`
            pub fn get_damping(&self) -> $float_type {
                self.damping
            }

            #[setter(damping)]
            /// [pyo3] setter for `damping`
            pub fn set_damping(&mut self, damping: $float_type) {
                self.damping = damping;
            }

            #[getter(mass)]
            /// [pyo3] getter for `mass`
            pub fn get_mass(&self) -> $float_type {
                self.mass
            }

            #[setter(mass)]
            /// [pyo3] setter for `mass`
            pub fn set_mass(&mut self, mass: $float_type) {
                self.mass = mass;
            }

            #[getter(kb_temperature)]
            /// [pyo3] getter for `kb_temperature`
            pub fn get_kb_temperature(&self) -> $float_type {
                self.kb_temperature
            }

            #[setter(kb_temperature)]
            /// [pyo3] setter for `kb_temperature`
            pub fn set_kb_temperature(&mut self, kb_temperature: $float_type) {
                self.kb_temperature = kb_temperature;
            }

            fn __repr__(&self) -> String {
                format!("{self:#?}")
            }
        }
    }
);

define_langevin_nd!(Langevin1D, 1, f64);
define_langevin_nd!(Langevin2D, 2, f64);
define_langevin_nd!(Langevin3D, 3, f64);
define_langevin_nd!(Langevin1DF32, 1, f32);
define_langevin_nd!(Langevin2DF32, 2, f32);
define_langevin_nd!(Langevin3DF32, 3, f32);

/// Mechanics model which represents cells as vertices with edges between them.
///
/// The vertices are attached to each other with springs and a given length between each
/// vertex.
/// Furthermore, we define a central pressure that acts when the total cell area is greater
/// or smaller than the desired one.
/// Each vertex is damped individually by the same constant.
// TODO include more formulas for this model
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct VertexMechanics2D<const D: usize> {
    points: nalgebra::SMatrix<f64, D, 2>,
    velocity: nalgebra::SMatrix<f64, D, 2>,
    /// Boundary lengths of individual edges
    pub cell_boundary_lengths: nalgebra::SVector<f64, D>,
    /// Spring tensions of individual edges
    pub spring_tensions: nalgebra::SVector<f64, D>,
    /// Total cell area
    pub cell_area: f64,
    /// Central pressure going from middle of the cell outwards
    pub central_pressure: f64,
    /// Damping constant
    pub damping_constant: f64,
    /// Controls the random motion of the entire cell
    pub diffusion_constant: f64,
}

impl<const D: usize> VertexMechanics2D<D> {
    /// Creates a new vertex model in equilibrium conditions.
    ///
    /// The specified parameters are then used to carefully calculate relating properties of the model.
    /// We outline the formulas used.
    /// Given the number of vertices \\(N\\) in our model (specified by the const generic argument
    /// of the [VertexMechanics2D] struct),
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
    /// Pressure, damping and spring tensions are not impacted by randomization.
    pub fn new(
        middle: SVector<f64, 2>,
        cell_area: f64,
        rotation_angle: f64,
        spring_tensions: f64,
        central_pressure: f64,
        damping_constant: f64,
        diffusion_constant: f64,
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
        let points = nalgebra::SMatrix::<f64, D, 2>::from_row_iterator(
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
        let cell_boundary_lengths = SVector::<f64, D>::from_iterator(
            points
                .row_iter()
                .circular_tuple_windows()
                .map(|(p1, p2)| (p1 - p2).norm()),
        );
        VertexMechanics2D {
            points,
            velocity: nalgebra::SMatrix::<f64, D, 2>::zeros(),
            cell_boundary_lengths,
            spring_tensions: SVector::<f64, D>::from_element(spring_tensions),
            cell_area,
            central_pressure,
            damping_constant,
            diffusion_constant,
        }
    }

    /// Calculates the boundary length of the regular polygon given the total area in equilibrium.
    ///
    /// The formula used is
    /// $$\\begin{align}
    ///     A &= \frac{L^2}{4n\tan\left(\frac{\pi}{n}\right)}\\\\
    ///     L &= \sqrt{4An\tan\left(\frac{\pi}{n}\right)}
    /// \\end{align}$$
    /// where $A$ is the total area, $n$ is the number of vertices and $L$ is the total boundary
    /// length.
    pub fn calculate_boundary_length(cell_area: f64) -> f64 {
        (4.0 * cell_area * (std::f64::consts::PI / D as f64).tan() * D as f64).sqrt()
    }

    /// Calculates the cell area of the regular polygon in equilibrium.
    ///
    /// The formula used is identical the the one of [Self::calculate_boundary_length].
    pub fn calculate_cell_area(boundary_length: f64) -> f64 {
        D as f64 * boundary_length.powf(2.0) / (4.0 * (std::f64::consts::PI / D as f64).tan())
    }

    /// Calculates the current area of the cell
    pub fn get_current_cell_area(&self) -> f64 {
        0.5_f64
            * self
                .points
                .row_iter()
                .circular_tuple_windows()
                .map(|(p1, p2)| p1.transpose().perp(&p2.transpose()))
                .sum::<f64>()
    }

    /// Calculate the current polygons boundary length
    pub fn calculate_current_boundary_length(&self) -> f64 {
        self.points
            .row_iter()
            .tuple_windows::<(_, _)>()
            .map(|(p1, p2)| {
                let dist = (p2 - p1).norm();
                dist
            })
            .sum::<f64>()
    }

    /// Obtain current cell area
    pub fn get_cell_area(&self) -> f64 {
        self.cell_area
    }

    /// Set the current cell area and adjust the length of edges such that the cell is still in
    /// equilibrium.
    pub fn set_cell_area_and_boundary_length(&mut self, cell_area: f64) {
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
                    self.damping_constant,
                    self.diffusion_constant,
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

    /// Change the internal cell area
    pub fn set_cell_area(&mut self, cell_area: f64) {
        self.cell_area = cell_area;
    }
}

impl VertexMechanics2D<6> {
    /// Fills the area of a given rectangle with hexagonal cells. Their orientation is such that
    /// the top border has a flat top.
    ///
    /// The produced pattern will like similar to this.
    /// ```ignore
    /// __________________________________
    /// |   ___       ___          ___   |
    /// |  /   \     /   \        /   \  |
    /// | /     \___/     \_ ..._/     \ |
    /// | \     /   \     /      \     / |
    /// |  \___/     \___/        \___/  |
    /// |  /   \     /   \        /   \  |
    /// | .     .   .     .      .     . |
    /// ```
    /// The padding around the generated cells will be determined automatically.
    pub fn fill_rectangle_flat_top(
        cell_area: f64,
        spring_tensions: f64,
        central_pressure: f64,
        damping_constant: f64,
        diffusion_constant: f64,
        rectangle: [SVector<f64, 2>; 2],
    ) -> Vec<Self> {
        // If the supplied area is larger than the total area, return nothing
        let side_x = rectangle[1].x - rectangle[0].x;
        let side_y = rectangle[1].y - rectangle[0].y;
        if cell_area > side_x * side_y {
            return Vec::new();
        }
        let segment_length = Self::calculate_boundary_length(cell_area) / 6.0;
        let radius_outer = Self::outer_radius_from_cell_area(cell_area);
        let radius_inner = Self::inner_radius_from_cell_area(cell_area);

        // Check if only one single hexagon fits into the domain in any dimension
        let n_max_x = (side_x - 2.0 * radius_outer).div_euclid(3.0 / 2.0 * radius_outer) as usize;
        let n_max_y = side_y.div_euclid(2.0 * radius_inner);
        let total_width_x = 2.0 * radius_outer + (n_max_x - 1) as f64 * 3.0 / 2.0 * radius_outer;
        let total_width_y = n_max_y as f64 * 2.0 * radius_inner;

        let pad_x = (side_x - total_width_x) / 2.0;
        let pad_y = (side_y - total_width_y) / 2.0;
        let padding = nalgebra::RowVector2::from([pad_x, pad_y]);

        let mut generated_models = vec![];
        for n_x in 0..n_max_x {
            for n_y in 0..n_max_y as usize - n_x % 2 {
                let middle = rectangle[0].transpose()
                    + padding
                    + nalgebra::RowVector2::from([
                        (1.0 + 3.0 / 2.0 * n_x as f64) * radius_outer,
                        (1 + 2 * n_y + n_x % 2) as f64 * radius_inner,
                    ]);
                let mut pos = nalgebra::SMatrix::<f64, 6, 2>::zeros();
                for i in 0..6 {
                    let phi = 2.0 * std::f64::consts::PI * i as f64 / 6.0;
                    pos.set_row(
                        i,
                        &(middle
                            + radius_outer * nalgebra::RowVector2::from([phi.cos(), phi.sin()])),
                    );
                }
                generated_models.push(Self {
                    points: pos,
                    velocity: SMatrix::zeros(),
                    cell_boundary_lengths: SVector::from_element(segment_length),
                    spring_tensions: SVector::from_element(spring_tensions),
                    cell_area,
                    central_pressure,
                    damping_constant,
                    diffusion_constant,
                });
            }
        }
        generated_models
    }
}

#[cfg(test)]
mod test_vertex_mechanics_6n {
    #[test]
    fn test_fill_too_small() {
        use crate::VertexMechanics2D;
        use nalgebra::Vector2;
        let models = VertexMechanics2D::<6>::fill_rectangle_flat_top(
            200.0,
            0.0,
            0.0,
            0.0,
            0.0,
            [Vector2::from([1.0, 1.0]), Vector2::from([2.0, 2.0])],
        );
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_fill_multiple() {
        use crate::VertexMechanics2D;
        use cellular_raza_concepts::Position;
        use nalgebra::Vector2;
        let models = VertexMechanics2D::<6>::fill_rectangle_flat_top(
            36.0,
            0.0,
            0.0,
            0.0,
            0.0,
            [Vector2::from([0.0; 2]), Vector2::from([100.0; 2])],
        );
        use itertools::Itertools;
        for (m1, m2) in models.into_iter().circular_tuple_windows() {
            if m1.pos().row_mean().transpose().x == m2.pos().row_mean().transpose().x {
                let max = m1
                    .pos()
                    .row_iter()
                    .map(|row| row.transpose().y)
                    .max_by(|x0, x1| x0.partial_cmp(x1).unwrap())
                    .unwrap();
                let min = m2
                    .pos()
                    .row_iter()
                    .map(|row| row.transpose().y)
                    .min_by(|x0, x1| x0.partial_cmp(x1).unwrap())
                    .unwrap();
                assert!((max - min).abs() < 1e-7);
            }
        }
    }
}

impl VertexMechanics2D<4> {
    /// Fill a specified rectangle with cells of 4 vertices
    pub fn fill_rectangle(
        cell_area: f64,
        spring_tensions: f64,
        central_pressure: f64,
        damping_constant: f64,
        diffusion_constant: f64,
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

                let points = nalgebra::SMatrix::<f64, 4, 2>::from_row_iterator([
                    corner.0,
                    corner.1,
                    corner.0 + cell_side_length,
                    corner.1,
                    corner.0 + cell_side_length,
                    corner.1 + cell_side_length,
                    corner.0,
                    corner.1 + cell_side_length,
                ]);
                let cell_boundary_lengths = nalgebra::SVector::<f64, 4>::from_iterator(
                    (0..2 * 4).map(|_| cell_side_length),
                );

                VertexMechanics2D {
                    points,
                    velocity: nalgebra::SMatrix::<f64, 4, 2>::zeros(),
                    cell_boundary_lengths,
                    spring_tensions: nalgebra::SVector::<f64, 4>::from_element(spring_tensions),
                    cell_area,
                    central_pressure,
                    damping_constant,
                    diffusion_constant,
                }
            })
            .collect::<Vec<_>>();

        filled_rectangle
    }
}

impl<const D: usize> VertexMechanics2D<D> {
    /// Calculates the outer circle radius of the Regular Polygon given its area.
    pub fn outer_radius_from_cell_area(cell_area: f64) -> f64 {
        // let segment_length = Self::calculate_boundary_length(cell_area) / D as f64;
        // segment_length / (std::f64::consts::PI / D as f64).tan() / 2.0
        let boundary_length = Self::calculate_boundary_length(cell_area);
        Self::outer_radius_from_boundary_length(boundary_length)
    }

    /// Calculates the outer circle radius of the Regular Polygon given its boundary length.
    pub fn outer_radius_from_boundary_length(boundary_length: f64) -> f64 {
        let segment_length = boundary_length / D as f64;
        segment_length / (std::f64::consts::PI / D as f64).sin() / 2.0
    }

    /// Calculates the inner circle radius of the Regular Polygon given its area.
    pub fn inner_radius_from_cell_area(cell_area: f64) -> f64 {
        let boundary_length = Self::calculate_boundary_length(cell_area);
        Self::inner_radius_from_boundary_length(boundary_length)
    }

    /// Calculates the inner circle radius of the Regular Polygon given its boundary length.
    pub fn inner_radius_from_boundary_length(boundary_length: f64) -> f64 {
        let segment_length = boundary_length / D as f64;
        segment_length / (std::f64::consts::PI / D as f64).tan() / 2.0
    }
}

impl<const D: usize>
    Mechanics<
        nalgebra::SMatrix<f64, D, 2>,
        nalgebra::SMatrix<f64, D, 2>,
        nalgebra::SMatrix<f64, D, 2>,
    > for VertexMechanics2D<D>
{
    fn calculate_increment(
        &self,
        force: nalgebra::SMatrix<f64, D, 2>,
    ) -> Result<(nalgebra::SMatrix<f64, D, 2>, nalgebra::SMatrix<f64, D, 2>), CalcError> {
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
        let dv = force + internal_force - self.damping_constant * self.velocity.clone();
        Ok((dx, dv))
    }

    fn get_random_contribution(
        &self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: f64,
    ) -> Result<(nalgebra::SMatrix<f64, D, 2>, nalgebra::SMatrix<f64, D, 2>), RngError> {
        let dvel = nalgebra::SMatrix::<f64, D, 2>::zeros();
        if dt != 0.0 {
            let random_vector: SVector<f64, 2> = wiener_process(rng, dt)?;
            let mut dpos = nalgebra::SMatrix::<f64, D, 2>::zeros();
            dpos.row_iter_mut().for_each(|mut r| {
                r *= 0.0;
                r += random_vector.transpose();
            });
            Ok((dpos, self.diffusion_constant * dvel))
        } else {
            Ok((nalgebra::SMatrix::<f64, D, 2>::zeros(), dvel))
        }
    }
}

impl<const D: usize> cellular_raza_concepts::Position<nalgebra::SMatrix<f64, D, 2>>
    for VertexMechanics2D<D>
{
    fn pos(&self) -> nalgebra::SMatrix<f64, D, 2> {
        self.points.clone()
    }

    fn set_pos(&mut self, pos: &nalgebra::SMatrix<f64, D, 2>) {
        self.points = pos.clone();
    }
}

impl<const D: usize> cellular_raza_concepts::Velocity<nalgebra::SMatrix<f64, D, 2>>
    for VertexMechanics2D<D>
{
    fn velocity(&self) -> nalgebra::SMatrix<f64, D, 2> {
        self.velocity.clone()
    }

    fn set_velocity(&mut self, velocity: &nalgebra::SMatrix<f64, D, 2>) {
        self.velocity = velocity.clone();
    }
}
