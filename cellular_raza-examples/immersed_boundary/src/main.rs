use cellular_raza::prelude::CalcError;
use rand_chacha::ChaCha8Rng;

trait Reactions<I> {
    fn increment_intracellular(&mut self, increment: &I);
    fn calculate_intracellular_increment(&self, rng: &mut ChaCha8Rng) -> Result<I, CalcError>;
}

trait ReactionsExtra<I, E> {
    fn increment_intracellular(&mut self, increment: &I);
    fn calculate_combined_increment(
        &self,
        extracellular: &E,
        rng: &mut ChaCha8Rng,
    ) -> Result<(I, E), CalcError>;
}

fn calculate_intermediate_velocity(
    u: &ndarray::Array3<f64>,
    h_n: &ndarray::Array3<f64>,
    h_n_min_one: &ndarray::Array3<f64>,
    nabla_pressure_min_one: &ndarray::Array3<f64>,
    dt: f64,
) -> ndarray::Array3<f64> {
    u + dt * (1.5 * h_n - 0.5 * h_n_min_one + 0.5 * nabla_pressure_min_one)
}

#[allow(unused)]
fn main() {
    let n_x = 10;
    let n_y = 10;
    let n_components = 3;

    let d = 1.0;
    let mut velocity_ib_points = ndarray::Array3::<f64>::zeros([n_x, n_y, n_components]);
    let mut i = ndarray::Array3::<f64>::zeros([n_x, n_y, n_components]);
    let mut f_intermediate = i.clone();
    let mut V = f_intermediate.clone();
    let mut I = f_intermediate.clone();
    let mut nabla_pressure_intermediate = f_intermediate.clone();
    let mut u_star = f_intermediate.clone();
    let dt = 1.0;

    for _ in 0..10 {
        // u_star = calculate_intermediate_velocite(&u, &h_n, &h_n_min_one, &nabla_pressure_min_one);
        f_intermediate = d * (&V - &I * (&u_star - 1.5 * dt * &nabla_pressure_intermediate));
        println!("{}", f_intermediate);
    }
}
