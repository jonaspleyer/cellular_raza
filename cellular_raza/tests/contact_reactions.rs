use cellular_raza::building_blocks::CartesianCuboid;
use cellular_raza::concepts::*;
use cellular_raza::core::{
    backend::chili::{Settings, SimulationError},
    storage::{StorageBuilder, StorageOption},
    time::FixedStepsize,
};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ReactionCell {
    pos: nalgebra::Vector2<f64>,
    index: (usize, usize),
    intracellular: f64,
    diffusion_constant: f64,
    dx: f64,
}

impl Position<nalgebra::Vector2<f64>> for ReactionCell {
    fn pos(&self) -> nalgebra::Vector2<f64> {
        self.pos
    }

    fn set_pos(&mut self, pos: &nalgebra::Vector2<f64>) {
        self.pos = *pos;
    }
}

impl Intracellular<f64> for ReactionCell {
    fn get_intracellular(&self) -> f64 {
        self.intracellular
    }

    fn set_intracellular(&mut self, intracellular: f64) {
        if self.index == (5, 5) {
            println!("{}", intracellular);
        }
        self.intracellular = intracellular;
    }
}

impl ReactionsContact<f64, nalgebra::Vector2<f64>> for ReactionCell {
    fn calculate_contact_increment(
        &self,
        own_intracellular: &f64,
        ext_intracellular: &f64,
        own_pos: &nalgebra::Vector2<f64>,
        ext_pos: &nalgebra::Vector2<f64>,
        _: &(),
    ) -> Result<(f64, f64), CalcError> {
        let diff = if (own_pos - ext_pos).norm() < 1.05 * self.dx {
            self.diffusion_constant * (ext_intracellular - own_intracellular) / self.dx.powf(2.0)
        } else {
            0.0
        };
        Ok((diff, -diff))
    }

    fn get_contact_information(&self) -> () {}
}

#[test]
fn contact_diffusion_2d_numerical() -> Result<(), SimulationError> {
    // Parameters of the simulation
    let domain_size = 100.0;
    let n_agents_side: usize = 3;
    let dx = domain_size / n_agents_side as f64;
    let v0 = 10.0;
    let diffusion_constant = 0.2;

    // Time Parameters
    let t0 = 0.0;
    let dt = 0.01;
    let save_interval = 0.01;
    let t_max: f64 = 1.0;
    let time_series: Vec<_> = (0..(t_max / save_interval).round() as usize + 1)
        .map(|i| t0 + i as f64 * save_interval)
        .collect();

    let agents = (0..n_agents_side.pow(2)).map(|n_agent: usize| {
        let n_x = n_agent % n_agents_side;
        let n_y = n_agent.div_euclid(n_agents_side);
        ReactionCell {
            pos: [0.5 * dx + n_x as f64 * dx, 0.5 * dx + n_y as f64 * dx].into(),
            index: (n_x, n_y),
            intracellular: if n_x == n_y
                && n_x > 0
                && n_y > 0
                && 2 * n_agent == (n_agents_side.pow(2) - 1)
            {
                v0
            } else {
                0.0
            },
            diffusion_constant,
            dx,
        }
    });
    let domain = CartesianCuboid::from_boundaries_and_n_voxels(
        [0.0; 2],
        [domain_size; 2],
        [n_agents_side; 2],
    )?;
    let storage = StorageBuilder::new().priority([StorageOption::Memory]);
    let time = FixedStepsize::from_partial_save_points(t0, dt, time_series.clone())?;
    let settings = Settings {
        time,
        storage,
        show_progressbar: false,
        n_threads: 1.try_into().unwrap(),
    };
    let storager = cellular_raza::core::backend::chili::run_simulation!(
        agents: agents,
        domain: domain,
        settings: settings,
        aspects: [ReactionsContact],
    )?;
    use cellular_raza::core::storage::StorageInterfaceLoad;
    let grids: Vec<_> = storager
        .cells
        .load_all_elements()?
        .into_iter()
        .map(|(iteration, elements)| {
            let mut grid =
                nalgebra::DMatrix::<f64>::from_element(n_agents_side, n_agents_side, -1.0);
            elements.into_iter().for_each(|(_, (cbox, _))| {
                let index = cbox.cell.index;
                let value = cbox.cell.intracellular;
                *grid.index_mut(index) = value;
            });
            (iteration, grid)
        })
        .collect();
    // Compare results with pure numerical simulation
    use core::ops::AddAssign;
    let mut y0 = nalgebra::DMatrix::<f64>::zeros(n_agents_side, n_agents_side);
    y0.index_mut((
        (n_agents_side - 1).div_euclid(2),
        (n_agents_side - 1).div_euclid(2),
    ))
    .add_assign(v0);
    // Solve this numerically
    let rhs_ode = |y: &nalgebra::DMatrix<f64>,
                   dy_ext: &mut nalgebra::DMatrix<f64>,
                   _: &f64,
                   _: &()|
     -> Result<(), ode_integrate::concepts::errors::CalcError> {
        let mut dy = nalgebra::DMatrix::<f64>::zeros(n_agents_side, n_agents_side);

        for n in 0..n_agents_side {
            for m in 0..n_agents_side {
                // Calculate first order derivatives along both dimensions
                // These are 4 values in total.
                let dy_n0 = if n == 0 {
                    0.0
                } else {
                    y[(n, m)] - y[(n - 1, m)]
                };
                let dy_n1 = if n == n_agents_side - 1 {
                    0.0
                } else {
                    y[(n + 1, m)] - y[(n, m)]
                };
                let dy_m0 = if m == 0 {
                    0.0
                } else {
                    y[(n, m)] - y[(n, m - 1)]
                };
                let dy_m1 = if m == n_agents_side - 1 {
                    0.0
                } else {
                    y[(n, m + 1)] - y[(n, m)]
                };
                let dy_n = dy_n1 - dy_n0;
                let dy_m = dy_m1 - dy_m0;
                dy.index_mut((n, m)).add_assign(dy_n + dy_m);
            }
        }
        *dy_ext = diffusion_constant / dx.powf(2.0) * dy;
        Ok(())
    };
    let results = ode_integrate::prelude::solve_ode_time_series_single_step_add(
        &y0,
        &time_series,
        &rhs_ode,
        &(),
        ode_integrate::prelude::Rk4,
    )
    .unwrap();
    for (y1, (_, y2)) in results.iter().skip(1).zip(grids.iter()) {
        let e = n_agents_side.pow(2) as f64 * diffusion_constant * dt / dx.powi(2);
        assert!((y1 - y2).norm() < e);
    }
    Ok(())
}

mod two_component_contact_reaction {
    use cellular_raza::building_blocks::*;
    use cellular_raza::concepts::*;
    use cellular_raza::core::{backend::chili::*, storage::*, time::*};
    use serde::{Deserialize, Serialize};
    #[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
    struct ContactCell {
        intracellular: nalgebra::Vector2<f64>,
        alpha0: f64,
        upper_limit: f64,
        #[Position]
        mechanics: NewtonDamped1DF32,
    }
    impl Intracellular<nalgebra::Vector2<f64>> for ContactCell {
        fn get_intracellular(&self) -> nalgebra::Vector2<f64> {
            self.intracellular
        }
        fn set_intracellular(&mut self, intracellular: nalgebra::Vector2<f64>) {
            self.intracellular = intracellular;
        }
    }
    impl ReactionsContact<nalgebra::Vector2<f64>, nalgebra::Vector1<f32>> for ContactCell {
        fn calculate_contact_increment(
            &self,
            own_intracellular: &nalgebra::Vector2<f64>,
            ext_intracellular: &nalgebra::Vector2<f64>,
            _own_pos: &nalgebra::Vector1<f32>,
            _ext_pos: &nalgebra::Vector1<f32>,
            _rinf: &(),
        ) -> Result<(nalgebra::Vector2<f64>, nalgebra::Vector2<f64>), CalcError> {
            let calculate_incr = |y: f64| -> f64 { self.alpha0 * y * (1.0 - y / self.upper_limit) };
            let own_dr = [self.alpha0, calculate_incr(own_intracellular[1])].into();
            let ext_dr = [self.alpha0, calculate_incr(ext_intracellular[1])].into();
            Ok((own_dr, ext_dr))
        }
        fn get_contact_information(&self) -> () {}
    }

    fn run_cellular_raza(
        alpha0: f64,
        y0: [f64; 2],
        upper_limit: f64,
        n_agents: usize,
        t0: f64,
        dt: f64,
        save_interval: usize,
        t_max: f64,
    ) -> Result<Vec<(f64, Vec<[f64; 2]>)>, SimulationError> {
        // Define initial values
        let y0 = nalgebra::Vector2::from(y0);

        // Agents
        let agents = (0..n_agents).map(|_| ContactCell {
            alpha0,
            intracellular: y0,
            upper_limit,
            mechanics: NewtonDamped1DF32 {
                pos: [0.5].into(),
                vel: [0.0].into(),
                damping_constant: 0.0,
                mass: 0.0,
            },
        });

        // Specify simulation domain, time and only store results intermediately in memory
        let domain = CartesianCuboid::from_boundaries_and_n_voxels([0.0; 1], [1.0; 1], [1; 1])?;
        let time = FixedStepsize::from_partial_save_freq(t0, dt, t_max, save_interval)?;
        let storage = StorageBuilder::new().priority([StorageOption::Memory]);
        let settings = Settings {
            n_threads: 1.try_into().unwrap(),
            show_progressbar: false,
            storage,
            time,
        };

        // Run full simulation and return storager to access results
        let storager = run_simulation!(
            agents: agents,
            settings: settings,
            domain: domain,
            aspects: [ReactionsContact],
        )?;

        // Gather cellular_raza results
        Ok(storager
            .cells
            .load_all_elements()?
            .into_iter()
            .map(|(iteration, elements)| {
                (
                    t0 + iteration as f64 * dt,
                    elements
                        .into_iter()
                        .map(|(_, (cbox, _))| cbox.cell.get_intracellular().into())
                        .collect(),
                )
            })
            .collect())
    }

    fn compare_results(
        production: f64,
        y0: [f64; 2],
        upper_limit: f64,
        n_agents: usize,
        t0: f64,
        dt: f64,
        save_interval: usize,
        t_max: f64,
        #[allow(unused)] save_filename: &str,
    ) -> Result<(), SimulationError> {
        // Define exact solution
        let q = (upper_limit - y0[1]) / y0[1];
        let exact_solution_derivative = |t: f64, n_deriv: i32| -> nalgebra::Vector2<f64> {
            let linear_growth = if n_deriv == 0 {
                y0[0] + (n_agents - 1) as f64 * production * (t - t0)
            } else {
                0.0
            };
            let logistic_curve = (1..n_deriv).product::<i32>() as f64
                * upper_limit
                * q.powi(n_deriv)
                * (1.0 + q * (-production * (n_agents - 1) as f64 * (t - t0)).exp())
                    .powi(-(n_deriv + 1));
            nalgebra::Vector2::from([linear_growth, logistic_curve])
        };

        // Estimate upper bound on local and global truncation error
        let lipschitz_constant = nalgebra::vector![
            (n_agents - 1) as f64 * production,
            y0[1] / (t_max - t0)
                * (1.0 / (1.0 + q * (-production * (n_agents - 1) as f64 * t0).exp())
                    - 1.0 / (1.0 + q * (-production * (n_agents - 1) as f64 * (t_max - t0)).exp()))
                .abs()
                .max(10.0 * std::f64::EPSILON)
        ];
        let fourth_derivative_bound = |t: f64| -> nalgebra::Vector2<f64> {
            nalgebra::vector![
                // This should be zero but due to machine precision we need to insert something here.
                // Since we are doing multiple operations such as add, multiply etc. we need more than
                // the minimal amount of precision
                80.0 * f64::EPSILON,
                exact_solution_derivative((t - 2.0 * dt).min(t0), 4)[1]
            ]
        };

        // Calculate upper bound on local and global truncation error
        let local_truncation_error = |t: f64| -> nalgebra::Vector2<f64> {
            &fourth_derivative_bound(t) * (3f64 / 8.0 * dt.powi(4))
        };
        let global_truncation_error = |t: f64| -> nalgebra::Vector2<f64> {
            nalgebra::Vector2::from([
                ((lipschitz_constant[0] * t).exp() - 1.0) * local_truncation_error(t)[0]
                    / dt
                    / lipschitz_constant[0],
                ((lipschitz_constant[1] * t).exp() - 1.0) * local_truncation_error(t)[1]
                    / dt
                    / lipschitz_constant[1],
            ])
        };

        // Obtain solutions from cellular_raza
        let solutions_cr = run_cellular_raza(
            production,
            y0,
            upper_limit,
            n_agents,
            t0,
            dt,
            save_interval,
            t_max,
        )?;

        // Compare the results
        let mut results = vec![];
        for (t, res_cr) in solutions_cr {
            let res_ex = exact_solution_derivative(t, 0);
            let e_global = global_truncation_error(t);
            let e_local = local_truncation_error(t);
            for r in res_cr.iter() {
                let d0 = (r[0] - res_ex[0]).abs();
                let d1 = (r[1] - res_ex[1]).abs();
                assert!(d0 < e_global[0]);
                println!("{} {}", d1, e_global[1]);
                assert!(d1 < e_global[1]);
            }
            results.push((t, res_ex, e_global, e_local, res_cr));
        }

        #[cfg(not(debug_assertions))]
        save_results(results, save_filename);

        Ok(())
    }

    #[allow(unused)]
    fn save_results(
        results: Vec<(
            f64,
            nalgebra::Vector2<f64>,
            nalgebra::Vector2<f64>,
            nalgebra::Vector2<f64>,
            Vec<[f64; 2]>,
        )>,
        save_filename: &str,
    ) {
        use std::fs::File;
        use std::io::prelude::*;
        let mut file = File::create(save_filename).unwrap();
        for (t, res_ex, e_global, e_local, res_cr) in results {
            write!(
                file,
                "{},{},{},{},{},{},{}",
                t, e_global[0], e_global[1], e_local[0], e_local[1], res_ex[0], res_ex[1]
            )
            .unwrap();
            for r_cr in res_cr {
                write!(file, ",{},{}", r_cr[0], r_cr[1]).unwrap();
            }
            writeln!(file, "").unwrap();
        }
    }

    #[test]
    fn test_config0() {
        // Simulation parameters
        let production = 0.2;
        let y0 = [1.0, 2.0];
        let upper_limit = 12.0;
        let t0 = 3.0;
        let dt = 0.21;
        let save_interval = 2;
        let t_max = 20.000001;
        let n_agents = 2;
        compare_results(
            production,
            y0,
            upper_limit,
            n_agents,
            t0,
            dt,
            save_interval,
            t_max,
            "tests/contact_reactions-config0.csv",
        )
        .unwrap();
    }

    #[test]
    fn test_config1() {
        // Simulation parameters
        let production = 0.3;
        let y0 = [1.0, 2.0];
        let upper_limit = 5.0;
        let t0 = 34.0;
        let dt = 0.01;
        let save_interval = 10;
        let t_max = 38.000001;
        let n_agents = 3;
        compare_results(
            production,
            y0,
            upper_limit,
            n_agents,
            t0,
            dt,
            save_interval,
            t_max,
            "tests/contact_reactions-config1.csv",
        )
        .unwrap();
    }

    #[test]
    fn test_config2() {
        // Simulation parameters
        let production = 1.0;
        let y0 = [1.0, 1.1];
        let upper_limit = 2.0;
        let t0 = 170.0;
        let dt = 0.1;
        let save_interval = 2;
        let t_max = 199.000001;
        let n_agents = 12;
        compare_results(
            production,
            y0,
            upper_limit,
            n_agents,
            t0,
            dt,
            save_interval,
            t_max,
            "tests/contact_reactions-config2.csv",
        )
        .unwrap();
    }
}
