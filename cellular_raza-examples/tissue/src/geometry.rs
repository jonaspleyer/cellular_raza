use nalgebra::{Matrix2, Vector2};

//// See Supplement 1 of https://doi.org/10.1371/journal.pcbi.1005569
pub fn calculate_circumcenter_derivative(
    ri: Vector2<f64>,
    rj: Vector2<f64>,
    rk: Vector2<f64>,
) -> [Matrix2<f64>; 3] {
    let rjk = rj - rk;
    let rki = rk - ri;
    let rij = ri - rj;

    let rjk_2 = rjk.norm_squared();
    let rki_2 = rki.norm_squared();
    let rij_2 = rij.norm_squared();

    let l2_sum = rjk_2 + rki_2 + rij_2;
    let lambda_1 = rjk_2 * (l2_sum - 2.0 * rjk_2);
    let lambda_2 = rki_2 * (l2_sum - 2.0 * rki_2);
    let lambda_3 = rij_2 * (l2_sum - 2.0 * rij_2);

    let lambda_total = lambda_1 + lambda_2 + lambda_3;

    // Gradients of lambdas
    let dl1_dri = 2.0 * rjk_2 * (-rki + rij);
    let dl2_dri = -2.0 * (rjk_2 + rij_2 - 2.0 * rki_2) * rki + 2.0 * rki_2 * rij;
    let dl3_dri = 2.0 * (rjk_2 + rki_2 - 2.0 * rij_2) * rij - 2.0 * rij_2 * rki;

    let dl1_drj = 2.0 * (rki_2 + rij_2 - 2.0 * rjk_2) * rjk - 2.0 * rjk_2 * rij;
    let dl2_drj = 2.0 * rki_2 * (rjk - rij);
    let dl3_drj = -2.0 * (rjk_2 + rki_2 - 2.0 * rij_2) * rij + 2.0 * rij_2 * rjk;

    let dl1_drk = -2.0 * (rki_2 + rij_2 - 2.0 * rjk_2) * rjk + 2.0 * rjk_2 * rki;
    let dl2_drk = 2.0 * (rjk_2 + rij_2 - 2.0 * rki_2) * rki - 2.0 * rki_2 * rjk;
    let dl3_drk = 2.0 * rij_2 * (-rjk + rki);

    let d_lam_dri = dl1_dri + dl2_dri + dl3_dri;
    let d_lam_drj = dl1_drj + dl2_drj + dl3_drj;
    let d_lam_drk = dl1_drk + dl2_drk + dl3_drk;

    let inv_l_total_2 = 1.0 / (lambda_total * lambda_total);

    let quotient_grad = |l_n: f64, dl_n: Vector2<f64>, d_total: Vector2<f64>| {
        inv_l_total_2 * (lambda_total * dl_n - l_n * d_total)
    };

    let grads = [
        (
            quotient_grad(lambda_1, dl1_dri, d_lam_dri),
            quotient_grad(lambda_2, dl2_dri, d_lam_dri),
            quotient_grad(lambda_3, dl3_dri, d_lam_dri),
        ),
        (
            quotient_grad(lambda_1, dl1_drj, d_lam_drj),
            quotient_grad(lambda_2, dl2_drj, d_lam_drj),
            quotient_grad(lambda_3, dl3_drj, d_lam_drj),
        ),
        (
            quotient_grad(lambda_1, dl1_drk, d_lam_drk),
            quotient_grad(lambda_2, dl2_drk, d_lam_drk),
            quotient_grad(lambda_3, dl3_drk, d_lam_drk),
        ),
    ];

    let l_div_lam = [
        lambda_1 / lambda_total,
        lambda_2 / lambda_total,
        lambda_3 / lambda_total,
    ];

    let mut gradients = [Matrix2::<f64>::zeros(); 3];
    for p in 0..3 {
        let (dg1, dg2, dg3) = grads[p];

        // Compute 2x2 Matrix: sum(grad_n ⊗ r_n) + (I * lambda_p/Lambda)
        let mut m = dg1 * ri.transpose() + dg2 * rj.transpose() + dg3 * rk.transpose();

        m[(0, 0)] += l_div_lam[p];
        m[(1, 1)] += l_div_lam[p];

        gradients[p] = m;
    }
    gradients
}
