use cellular_raza::prelude::*;
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, CellAgent)]
struct Agent {
    #[Mechanics]
    mechanics: Langevin2D,
    #[Interaction]
    interaction: BoundLennardJones,
}

trait Collision<Pos> {
    fn are_in_contact(&self, pos: &Pos) -> bool;
}

fn intersect(p1: &Vector2<f64>, p2: &Vector2<f64>, radius: &f64) -> bool {
    let dist = (p1 - p2).norm();
    dist < 2. * radius
}

fn collision_response(
    p1_pre: &Vector2<f64>,
    p1: &Vector2<f64>,
    v1: &Vector2<f64>,
    p2_pre: &Vector2<f64>,
    p2: &Vector2<f64>,
    v2: &Vector2<f64>,
    radius: &f64,
) -> (Vector2<f64>, Vector2<f64>, Vector2<f64>, Vector2<f64>) {
    // Calculate point at which the two paths would intersect
    // First calculate the two incremental velocitities between points t and t+1
    let dv1 = p1 - p1_pre;
    let dv2 = p2 - p2_pre;
    let dv = dv1 - dv2;
    let dp = p1_pre - p2_pre;

    // println!("{dv} {dp}");

    // The two equations for the line segments are
    // h1(s) = p1_pre + s * dv1;
    // h2(s) = p2_pre + s * dv2;
    //
    // The difference at any point in time is
    // dist(s) = h1(s) - h2(s)
    // dist(s) = p1_pre - p2_pre + s * (dv1 - dv2)
    // When this vector's length is equal to 2*r, we have reached our point
    //
    // This is equivalent to
    //   (p1_pre_1 - p2_pre_1 + s * (dv1_1 - dv2_1))^2
    // + (p1_pre_2 - p2_pre_2 + s * (dv1_2 - dv2_2))^2
    // = 4 * r^2
    //
    // Substitute
    // (a, c) = dp = p1_pre - p2_pre
    // (b, d) = dv = dv1 - dv2
    //
    // Then
    // (a+sb)² + (c+sd)² = 4r²
    // s²(b²+d²) + 2s(ab + cd) + a² + c² = 4r²
    // s² + 2s(ab + cd)/(b²+d²) + (a² + c² - 4r²)/(b²+d²) = 0
    // p = (ab + cd)/(b²+d²)
    // q = (a² + c² - 4r²)/(b²+d²)
    // Or in vector notation:
    // p = dv . dp / |dv|²
    // q = (|dp|² - 4r²)/|dv|²

    let dvn = dv.norm_squared();
    let p = dp.dot(&dv) / dvn;
    let q = (dp.norm_squared() - 4.0 * radius.powf(2.0)) / dvn;

    // By solving the quadratic equation we obtain the two solutions
    // s1 = - p + sqrt(p² - q)
    // s1 = - p - sqrt(p² - q)

    let sqrt_term = (p.powf(2.0) - q).sqrt();
    let s1 = -p + sqrt_term;
    let s2 = -p - sqrt_term;
    let s = if (0.0..1.0).contains(&s1) { s1 } else { s2 };

    // Now we have calculated the points of impact
    let p1_impact = p1_pre + s * dv1;
    let p2_impact = p2_pre + s * dv2;

    // Calculate the difference
    let diff = p2_impact - p1_impact;
    let dir = diff.normalize();

    assert!((dir.norm() - 1.) < 1e-3);

    // Calculate the new velocities by inverting the direction along the connecting line
    let v1_new = v1 - 2. * v1.dot(&dir).abs() * dir;
    let v2_new = v2 + 2. * v2.dot(&dir).abs() * dir;

    let p1_new = p1_pre + s * dv1 + (1. - s) * (dv1 - dv1.dot(&dir).abs() * dir);
    let p2_new = p2_pre + s * dv2 + (1. - s) * (dv2 + dv2.dot(&dir).abs() * dir);

    (p1_new, v1_new, p2_new, v2_new)
}

fn main() -> Result<(), SimulationError> {
    // Agents setup
    let mut p1 = Vector2::zeros();
    let mut p2 = Vector2::from([1.0, 0.0]);

    let mut v1 = Vector2::from([0.0, 0.1]);
    let mut v2 = Vector2::from([0.0, -0.1]);

    let mass1 = 1.0;
    let mass2 = 0.5;

    let force = |p1: &Vector2<f64>, p2: &Vector2<f64>| -> Vector2<f64> {
        let dir = p2 - p1;
        let d = dir.norm();
        1.0 / (1.0 + d) * dir
        // p2 - p1
    };

    let radius = 0.2;

    let dt = 0.01f64;
    for n in 0..1_000_000 {
        let p1_pre = p1;
        let p2_pre = p2;

        let f = force(&p1, &p2);

        // Euler integrator
        v1 += f * dt;
        v2 -= f * dt;
        p1 += v1 * dt;
        p2 += v2 * dt;

        // Apply collision detection and reponse
        if intersect(&p1, &p2, &radius) {
            (p1, v1, p2, v2) = collision_response(&p1_pre, &p1, &v1, &p2_pre, &p2, &v2, &radius);
        }
        if n % 2 == 0 {
            println!("{} {} {} {}", p1[0], p1[1], p2[0], p2[1]);
        }
    }

    /* let agent = Agent {
        mechanics: Langevin2D {
            pos: Vector2::from([0.0, 0.0]),
            vel: Vector2::from([0.0, 0.0]),
            mass: 1.0,
            damping: 0.1,
            kb_temperature: 0.00,
        },
        interaction: BoundLennardJones {
            epsilon: 0.1,
            sigma: 0.5,
            bound: 0.1,
            cutoff: 2.0,
        },
    };

    let domain_size = 10.0;
    let n_voxels = 4;
    let mut agents = [agent.clone(), agent];
    agents[0].mechanics.pos = Vector2::from([4.5, 5.0]);
    agents[1].mechanics.pos = Vector2::from([5.5, 5.0]);

    // Domain Setup
    let domain =
        CartesianCuboid::from_boundaries_and_n_voxels([0.0; 2], [domain_size; 2], [n_voxels; 2])?;

    // Storage Setup
    let storage_builder =
        cellular_raza::prelude::StorageBuilder::new().priority([StorageOption::Memory]);

    // Time Setup
    let t0 = 0.0;
    let dt = 0.001;
    let save_points: Vec<_> = (0..101).map(|n| n as f64).collect();
    let time_stepper = cellular_raza::prelude::time::FixedStepsize::from_partial_save_points(
        t0,
        dt,
        save_points.clone(),
    )?;

    let settings = Settings {
        n_threads: 1.try_into().unwrap(),
        time: time_stepper,
        storage: storage_builder,
        progressbar: Some("Running Simulation".into()),
    };

    let container = run_simulation!(
        domain: domain,
        agents: agents,
        settings: settings,
        aspects: [Mechanics, Interaction],
    )?;

    let mut idents = container.cells.load_all_element_histories()?.into_iter();
    let id0 = idents.next().unwrap().0;
    let id1 = idents.next().unwrap().0;
    for (_, cells) in container.cells.load_all_elements()? {
        let p0 = cells[&id0].0.cell.mechanics.pos;
        let p1 = cells[&id1].0.cell.mechanics.pos;
        println!("{:.3?} {:.3?}", p0, p1);
    }*/

    Ok(())
}
