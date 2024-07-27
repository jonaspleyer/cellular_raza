use cellular_raza::building_blocks::NewtonDamped2D;
use cellular_raza::concepts::{
    CalcError, CellAgent, Interaction, Mechanics, Position, RngError, Velocity,
};

use cellular_raza_building_blocks::CartesianCuboid;
use cellular_raza_core::backend::chili::{run_simulation, Settings};
use cellular_raza_core::storage::{StorageBuilder, StorageInterfaceLoad, StorageOption};
use cellular_raza_core::time::FixedStepsize;
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

pub const KILOGRAMM: f64 = 1.0;
pub const METRE: f64 = 1.0;
pub const KILOMETRE: f64 = 1e3 * METRE;
pub const SECOND: f64 = 1.0;
pub const MINUTE: f64 = 60.0 * SECOND;
pub const HOUR: f64 = 60.0 * MINUTE;
pub const DAY: f64 = 24.0 * HOUR;
pub const NEWTON: f64 = 1.0 * KILOGRAMM * METRE / SECOND / SECOND;
pub const GRAVITATIONAL_CONSTANT: f64 = 6.6743e-11 * NEWTON * METRE * METRE / KILOGRAMM / KILOGRAMM;

#[derive(CellAgent, PartialEq, Debug, Clone, Deserialize, Serialize)]
struct MassParticle {
    #[Mechanics]
    mechanics: NewtonDamped2D,
    id: usize,
}

impl Interaction<Vector2<f64>, Vector2<f64>, Vector2<f64>, f64> for MassParticle {
    fn get_interaction_information(&self) -> f64 {
        self.mechanics.mass
    }
    fn calculate_force_between(
        &self,
        own_pos: &Vector2<f64>,
        _own_vel: &Vector2<f64>,
        ext_pos: &Vector2<f64>,
        _ext_vel: &Vector2<f64>,
        ext_mass: &f64,
    ) -> Result<(Vector2<f64>, Vector2<f64>), CalcError> {
        let z = own_pos - ext_pos;
        let r2 = z.norm_squared();
        let dir = z.normalize();
        // let force = GRAVITATIONAL_CONSTANT * dir * (self.mechanics.mass * ext_mass) / r2;
        let force = force_strength(r2, self.mechanics.mass, *ext_mass) * dir;
        assert!(!force.norm().is_nan());
        Ok((-force, force))
    }
}

fn force_strength(distance_pow2: f64, m1: f64, m2: f64) -> f64 {
    GRAVITATIONAL_CONSTANT * (m1 * m2) / distance_pow2
}

// TODO revisit this test and then enable again
// possibly decide to fix position of earth
fn two_body_problem() {
    // ======= Parameters of the problem
    let distance_to_moon = 385e3 * KILOMETRE;
    let angular_velocity = 2.0 * std::f64::consts::PI / (27.3 * DAY) * distance_to_moon;
    let m_moon = 7.3476e22 * KILOGRAMM;
    let m_earth = 5.972e24 * KILOGRAMM;

    // Time values to solve for
    // One revelation is approximately 28 days.
    let t0 = 0.0 * DAY;
    let dt = 0.00002 * DAY;
    let n_steps = 500;
    let time_series: Vec<_> = (0..n_steps).map(|i| t0 + i as f64 * dt).collect();

    let particles = [
        // This is the moon
        MassParticle {
            mechanics: NewtonDamped2D {
                pos: [-distance_to_moon, 0.0].into(),
                vel: [0.0, angular_velocity].into(),
                damping_constant: 0.0,
                mass: m_moon,
            },
            id: 0,
        },
        // This is earth
        MassParticle {
            mechanics: NewtonDamped2D {
                pos: [0.0; 2].into(),
                vel: [0.0; 2].into(),
                damping_constant: 0.0,
                mass: m_earth,
            },
            id: 1,
        },
    ];
    let domain = CartesianCuboid::from_boundaries_and_n_voxels(
        [-500_000.0 * KILOMETRE; 2],
        [500_000.0 * KILOMETRE; 2],
        [1; 2],
    )
    .unwrap();
    let time = FixedStepsize::from_partial_save_points(t0, dt, time_series.clone()).unwrap();
    let storage = StorageBuilder::new().priority([StorageOption::Memory]);
    let settings = Settings {
        time,
        storage,
        show_progressbar: false,
        n_threads: 1.try_into().unwrap(),
    };
    let storager = run_simulation!(
        agents: particles,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction],
    )
    .unwrap();
    let positions =
        storager
            .cells
            .load_all_elements()
            .unwrap()
            .into_iter()
            .map(|(iteration, elements)| {
                let mut elements: Vec<_> = elements.into_iter().map(|x| x).collect();
                elements.sort_by_key(|(id, _)| id.1);
                let mut matrix = SMatrix::<f64, 4, 2>::zeros();
                elements
                    .into_iter()
                    .enumerate()
                    .for_each(|(n_cell, (_, (cbox, _)))| {
                        use core::ops::AddAssign;
                        matrix
                            .row_mut(2 * n_cell)
                            .add_assign(cbox.mechanics.pos.transpose());
                        matrix
                            .row_mut(2 * n_cell + 1)
                            .add_assign(cbox.mechanics.vel.transpose());
                    });
                (iteration, matrix)
            });

    use nalgebra::SMatrix;
    // We store the positions in row 0 and 2
    // and the velocities in row 1 and 3
    let exact_rhs = |y: &SMatrix<f64, 4, 2>,
                     dy: &mut SMatrix<f64, 4, 2>,
                     _: &f64,
                     _: &()|
     -> Result<(), ode_integrate::concepts::errors::CalcError> {
        let z = y.row(0) - y.row(2);
        let distance_pow2 = z.norm_squared();
        let dir = z.normalize();
        let force_strength = force_strength(distance_pow2, m_moon, m_earth);
        let force = dir * force_strength;
        // Increment position and velocity of particle1
        use core::ops::AddAssign;
        // Clear previous values
        *dy *= 0.0;
        dy.row_mut(0).add_assign(&y.row(1));
        dy.row_mut(1).add_assign(&force / m_moon);
        // Incremetn position and velocity of paticle2
        dy.row_mut(2).add_assign(&y.row(3));
        dy.row_mut(3).add_assign(&-force / m_earth);
        Ok(())
    };
    let y0 = SMatrix::<f64, 4, 2>::from_rows(&[
        nalgebra::RowVector2::from([-distance_to_moon, 0.0]),
        nalgebra::RowVector2::from([0.0, angular_velocity]),
        nalgebra::RowVector2::from([0.0; 2]),
        nalgebra::RowVector2::from([0.0; 2]),
    ]);
    let res = ode_integrate::prelude::solve_ode_time_series_single_step_add(
        &y0,
        &time_series,
        &exact_rhs,
        &(),
        ode_integrate::prelude::Rk4,
    )
    .unwrap();
    for ((_, res_cr), res_ode) in positions.into_iter().zip(res.into_iter()) {
        // Obtain positions from both solutions
        let p_moon_cr = res_cr.row(0);
        let p_earth_cr = res_cr.row(2);
        let p_moon_ode = res_ode.row(0);
        let p_earth_ode = res_ode.row(2);

        // Compare the solutions
        let n1 = (p_moon_cr - p_moon_ode).norm();
        let n2 = (p_earth_cr - p_earth_ode).norm();
        assert!(n1 < distance_to_moon * dt * 1e-5);
        assert!(n2 < distance_to_moon * dt * 1e-5);
    }
}
