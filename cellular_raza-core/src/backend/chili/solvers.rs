use cellular_raza_concepts::{CalcError, Xapy};
use num::FromPrimitive;

#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{UpdateReactions, UpdateReactionsContact};

/// Classical euler solver for the [Mechanics](cellular_raza_concepts::Mechanics) trait.
///
/// The euler solver is the most simple solver and not stable for many problems.
/// Thus its usage is discouraged. For another general-purpose solver look at
/// [mechanics_adams_bashforth_2].
///
/// The update step follows the simple equations
/// \\begin{align}
///     x(t_{i+1}) &= x(t_i) + \Delta t \frac{d x}{d t}(t_i)\\\\
///     v(t_{i+1}) &= v(t_i) + \Delta t \frac{d v}{d t}(t_i)
/// \\end{align}
/// where $\Delta t$ is the step size and $dx/dt$ and $dv/dt$ are calculated by the
/// [calculate_increment](cellular_raza_concepts::Mechanics::calculate_increment) method.
#[cfg_attr(feature = "tracing", instrument(skip_all))]
pub fn mechanics_euler<C, A, Pos, Vel, For, Float>(
    cell: &mut C,
    aux_storage: &mut A,
    dt: Float,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> Result<(), super::SimulationError>
where
    A: super::aux_storage::UpdateMechanics<Pos, Vel, For, 0>,
    C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
    C: cellular_raza_concepts::Position<Pos>,
    C: cellular_raza_concepts::Velocity<Vel>,
    Pos: Xapy<Float> + Clone,
    Vel: Xapy<Float> + Clone,
    Float: num::Float + Copy,
{
    let force = aux_storage.get_current_force_and_reset();
    let velocity = cell.velocity();
    let position = cell.pos();

    let (dx, dv) = cell.calculate_increment(force)?;
    let (dx_rand, dv_rand) = cell.get_random_contribution(rng, dt)?;

    // Update values in the aux_storage
    aux_storage.set_last_position(dx.clone());
    aux_storage.set_last_velocity(dv.clone());

    // Calculate new position and velocity of cell
    let new_position = euler(position, dx, dt, dx_rand)?;
    let new_velocity = euler(velocity, dv, dt, dv_rand)?;
    cell.set_pos(&new_position);
    cell.set_velocity(&new_velocity);
    Ok(())
}

/// Note that the const generic for this struct is the order of the solver minus one.
/// This is due to the fact that the AuxStorage only stores one less step than the order of the
/// solver.
pub(crate) struct MechanicsAdamsBashforthSolver<const N: usize>;

pub(crate) trait AdamsBashforth<const N: usize> {
    #[allow(unused)]
    fn update<C, A, Pos, Vel, For, Float>(
        cell: &mut C,
        aux_storage: &mut A,
        dt: Float,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Result<(), super::SimulationError>
    where
        A: super::aux_storage::UpdateMechanics<Pos, Vel, For, N>,
        C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        C: cellular_raza_concepts::Position<Pos>,
        C: cellular_raza_concepts::Velocity<Vel>,
        Pos: Xapy<Float> + Clone,
        Vel: Xapy<Float> + Clone,
        Float: num::Float + FromPrimitive;
}

impl AdamsBashforth<2> for MechanicsAdamsBashforthSolver<2> {
    #[allow(unused)]
    fn update<C, A, Pos, Vel, For, Float>(
        cell: &mut C,
        aux_storage: &mut A,
        dt: Float,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Result<(), super::SimulationError>
    where
        A: super::aux_storage::UpdateMechanics<Pos, Vel, For, 2>,
        C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        C: cellular_raza_concepts::Position<Pos>,
        C: cellular_raza_concepts::Velocity<Vel>,
        Pos: Xapy<Float> + Clone,
        Vel: Xapy<Float> + Clone,
        Float: num::Float + FromPrimitive,
    {
        mechanics_adams_bashforth_3(cell, aux_storage, dt, rng)
    }
}

impl AdamsBashforth<1> for MechanicsAdamsBashforthSolver<1> {
    fn update<C, A, Pos, Vel, For, Float>(
        cell: &mut C,
        aux_storage: &mut A,
        dt: Float,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Result<(), super::SimulationError>
    where
        A: super::aux_storage::UpdateMechanics<Pos, Vel, For, 1>,
        C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        C: cellular_raza_concepts::Position<Pos>,
        C: cellular_raza_concepts::Velocity<Vel>,
        Pos: Xapy<Float> + Clone,
        Vel: Xapy<Float> + Clone,
        Float: num::Float + FromPrimitive,
    {
        mechanics_adams_bashforth_2(cell, aux_storage, dt, rng)
    }
}

impl AdamsBashforth<0> for MechanicsAdamsBashforthSolver<0> {
    fn update<C, A, Pos, Vel, For, Float>(
        cell: &mut C,
        aux_storage: &mut A,
        dt: Float,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Result<(), super::SimulationError>
    where
        A: super::aux_storage::UpdateMechanics<Pos, Vel, For, 0>,
        C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        C: cellular_raza_concepts::Position<Pos>,
        C: cellular_raza_concepts::Velocity<Vel>,
        Pos: Xapy<Float> + Clone,
        Vel: Xapy<Float> + Clone,
        Float: num::Float + FromPrimitive,
    {
        mechanics_euler(cell, aux_storage, dt, rng)
    }
}

/// Three-step Adams-Bashforth method.
///
/// See also the [Wikipedia](https://en.wikipedia.org/wiki/Linear_multistep_method) article.
/// We track previous increments of the update steps and use these in order to update the next time
/// steps.
///
/// The equations for updating are given by
/// \\begin{equation}
///     y(t_{i+3}) = y(t_{i+2}) + \Delta t\left(\frac{23}{12}\frac{dy}{dt}(t_{i+2})  - \frac{16}{12}\frac{dy}{dt}(t_{i+1}) + \frac{5}{12}\frac{dy}{dt}(t_i)\right)
/// \\end{equation}
///
/// for both the position and velocity.
/// In the beginning of the simulation, when not enough previous increment values are known,
/// we resort to the [mechanics_adams_bashforth_2] and [mechanics_euler] solver.
#[cfg_attr(feature = "tracing", instrument(skip_all))]
pub fn mechanics_adams_bashforth_3<C, A, Pos, Vel, For, Float>(
    cell: &mut C,
    aux_storage: &mut A,
    dt: Float,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> Result<(), super::SimulationError>
where
    A: super::aux_storage::UpdateMechanics<Pos, Vel, For, 2>,
    C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
    C: cellular_raza_concepts::Position<Pos>,
    C: cellular_raza_concepts::Velocity<Vel>,
    Pos: Xapy<Float> + Clone,
    Vel: Xapy<Float> + Clone,
    Float: num::Float + FromPrimitive,
{
    let force = aux_storage.get_current_force_and_reset();
    let velocity = cell.velocity();
    let position = cell.pos();

    let (dx, dv) = cell.calculate_increment(force)?;
    let (dx_rand, dv_rand) = cell.get_random_contribution(rng, dt)?;

    // Update values in the aux_storage
    aux_storage.set_last_position(dx.clone());
    aux_storage.set_last_velocity(dv.clone());

    // Calculate new position and velocity of cell
    let n_previous_values = aux_storage.n_previous_values();
    let mut old_pos_increments = aux_storage.previous_positions();
    let mut old_vel_increments = aux_storage.previous_velocities();
    let (new_position, new_velocity) = match n_previous_values {
        2 => (
            adams_bashforth_3(
                position,
                [
                    dx,
                    old_pos_increments.next().unwrap().clone(),
                    old_pos_increments.next().unwrap().clone(),
                ],
                dt,
                dx_rand,
            )?,
            adams_bashforth_3(
                velocity,
                [
                    dv,
                    old_vel_increments.next().unwrap().clone(),
                    old_vel_increments.next().unwrap().clone(),
                ],
                dt,
                dv_rand,
            )?,
        ),
        1 => (
            adams_bashforth_2(
                position,
                [dx, old_pos_increments.next().unwrap().clone()],
                dt,
                dx_rand,
            )?,
            adams_bashforth_2(
                velocity,
                [dv, old_vel_increments.next().unwrap().clone()],
                dt,
                dv_rand,
            )?,
        ),
        _ => (
            euler(position, dx, dt, dx_rand)?,
            euler(velocity, dv, dt, dv_rand)?,
        ),
    };
    cell.set_pos(&new_position);
    cell.set_velocity(&new_velocity);
    Ok(())
}

/// Two-step Adams-Bashforth method.
///
/// See also the [Wikipedia](https://en.wikipedia.org/wiki/Linear_multistep_method) article.
/// We track previous increments of the update steps and use these in order to update the next time
/// steps.
///
/// The equations for updating are given by
/// \\begin{equation}
///     y(t_{i+2}) = y(t_{i+1}) + \Delta t\left(\frac{3}{12}\frac{dy}{dt}(t_{i+1})  - \frac{1}{2}\frac{dy}{dt}(t_{i})\right)
/// \\end{equation}
///
/// for both the position and velocity.
/// In the beginning of the simulation, when not enough previous increment values are known,
/// we resort to the [euler](mechanics_euler) solver.
pub fn mechanics_adams_bashforth_2<C, A, Pos, Vel, For, Float>(
    cell: &mut C,
    aux_storage: &mut A,
    dt: Float,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> Result<(), super::SimulationError>
where
    A: super::aux_storage::UpdateMechanics<Pos, Vel, For, 1>,
    C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
    C: cellular_raza_concepts::Position<Pos>,
    C: cellular_raza_concepts::Velocity<Vel>,
    Pos: Xapy<Float> + Clone,
    Vel: Xapy<Float> + Clone,
    Float: num::Float + FromPrimitive,
{
    let force = aux_storage.get_current_force_and_reset();
    let velocity = cell.velocity();
    let position = cell.pos();

    let (dx, dv) = cell.calculate_increment(force)?;
    let (dx_rand, dv_rand) = cell.get_random_contribution(rng, dt)?;

    // Update values in the aux_storage
    aux_storage.set_last_position(dx.clone());
    aux_storage.set_last_velocity(dv.clone());

    // Calculate new position and velocity of cell
    let n_previous_values = aux_storage.n_previous_values();
    let mut old_pos_increments = aux_storage.previous_positions();
    let mut old_vel_increments = aux_storage.previous_velocities();
    let (new_position, new_velocity) = match n_previous_values {
        1 => (
            adams_bashforth_2(
                position,
                [dx, old_pos_increments.next().unwrap().clone()],
                dt,
                dx_rand,
            )?,
            adams_bashforth_2(
                velocity,
                [dv, old_vel_increments.next().unwrap().clone()],
                dt,
                dv_rand,
            )?,
        ),
        _ => (
            euler(position, dx, dt, dx_rand)?,
            euler(velocity, dv, dt, dv_rand)?,
        ),
    };
    cell.set_pos(&new_position);
    cell.set_velocity(&new_velocity);
    Ok(())
}

#[inline]
fn euler<X, F>(x: X, dx: X, dt: F, dx_rand: X) -> Result<X, CalcError>
where
    X: Xapy<F>,
    F: num::Float + Copy,
{
    let x_new = dx.xapy(dt, &x).xapy(F::one(), &dx_rand.xa(dt));
    Ok(x_new)
}

#[inline]
fn adams_bashforth_3<X, F>(x: X, dx: [X; 3], dt: F, dx_rand: X) -> Result<X, CalcError>
where
    X: Xapy<F>,
    F: Copy + FromPrimitive + num::Float,
{
    let f0 = F::from_isize(23).unwrap() / F::from_isize(12).unwrap();
    let f1 = -F::from_isize(16).unwrap() / F::from_isize(12).unwrap();
    let f2 = F::from_isize(5).unwrap() / F::from_isize(12).unwrap();

    let [dx0, dx1, dx2] = dx;

    let x_new = dx0
        .xapy(f0, &dx1.xapy(f1, &dx2.xa(f2)))
        .xapy(dt, &x)
        .xapy(F::one(), &dx_rand.xa(dt));
    Ok(x_new)
}

#[inline]
fn adams_bashforth_2<X, F>(x: X, dx: [X; 2], dt: F, dx_rand: X) -> Result<X, CalcError>
where
    X: Xapy<F>,
    F: Copy + FromPrimitive + num::Float,
{
    let f0 = F::from_isize(3).unwrap() / F::from_isize(2).unwrap();
    let f1 = -F::from_isize(1).unwrap() / F::from_isize(2).unwrap();

    let [dx0, dx1] = dx;

    let x_new = dx0
        .xapy(f0, &dx1.xa(f1))
        .xapy(dt, &x)
        .xapy(F::one(), &dx_rand.xa(dt));
    Ok(x_new)
}

/// Note that the const generic for this struct is the order of the solver minus one.
/// This is due to the fact that the AuxStorage only stores one less step than the order of the
/// solver.
pub(crate) struct ReactionsRungeKuttaSolver<const N: usize>;

pub(crate) trait RungeKutta<const N: usize> {
    #[allow(unused)]
    fn update<C, A, Ri, Float>(
        cell: &mut C,
        aux_storage: &mut A,
        dt: Float,
    ) -> Result<(), super::SimulationError>
    where
        A: UpdateReactions<Ri>,
        C: cellular_raza_concepts::Reactions<Ri>,
        Float: num::Float,
        Ri: Xapy<Float>;
}

impl RungeKutta<1> for ReactionsRungeKuttaSolver<1> {
    #[allow(unused)]
    #[inline]
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    fn update<C, A, Ri, Float>(
        cell: &mut C,
        aux_storage: &mut A,
        dt: Float,
    ) -> Result<(), super::SimulationError>
    where
        A: UpdateReactions<Ri>,
        C: cellular_raza_concepts::Reactions<Ri>,
        Float: num::Float,
        Ri: Xapy<Float>,
    {
        // Constants
        let intra = cell.get_intracellular();

        // Calculate the intermediate steps
        let dintra = cell.calculate_intracellular_increment(&intra)?;

        // Update the internal value of the cell
        aux_storage.incr_conc(dintra);
        Ok(())
    }
}

impl RungeKutta<2> for ReactionsRungeKuttaSolver<2> {
    #[allow(unused)]
    #[inline]
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    fn update<C, A, Ri, Float>(
        cell: &mut C,
        aux_storage: &mut A,
        dt: Float,
    ) -> Result<(), super::SimulationError>
    where
        A: UpdateReactions<Ri>,
        C: cellular_raza_concepts::Reactions<Ri>,
        Float: num::Float,
        Ri: Xapy<Float>,
    {
        // Constants
        let two = Float::one() + Float::one();
        let intra = cell.get_intracellular();

        // Calculate the intermediate steps
        let dintra1 = cell.calculate_intracellular_increment(&intra)?;
        let dintra = cell.calculate_intracellular_increment(&dintra1.xapy(dt / two, &intra))?;

        // Update the internal value of the cell
        aux_storage.incr_conc(dintra);
        Ok(())
    }
}

impl RungeKutta<4> for ReactionsRungeKuttaSolver<4> {
    #[allow(unused)]
    #[inline]
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    fn update<C, A, Ri, Float>(
        cell: &mut C,
        aux_storage: &mut A,
        dt: Float,
    ) -> Result<(), super::SimulationError>
    where
        A: UpdateReactions<Ri>,
        C: cellular_raza_concepts::Reactions<Ri>,
        Float: num::Float,
        Ri: Xapy<Float>,
    {
        // Constants
        let two = Float::one() + Float::one();
        let six = two + two + two;

        let intra = cell.get_intracellular();

        // Calculate the intermediate steps
        let dintra1 = cell.calculate_intracellular_increment(&intra)?;
        let dintra2 = cell.calculate_intracellular_increment(&dintra1.xapy(dt / two, &intra))?;
        let dintra3 = cell.calculate_intracellular_increment(&dintra2.xapy(dt / two, &intra))?;
        let dintra4 = cell.calculate_intracellular_increment(&dintra3.xapy(dt, &intra))?;
        let dintra = dintra1.xapy(
            Float::one() / six,
            &dintra2.xapy(
                two / six,
                &dintra3.xapy(two / six, &dintra4.xa(Float::one() / six)),
            ),
        );

        // Update the internal value of the cell
        aux_storage.incr_conc(dintra);
        Ok(())
    }
}

/// Calculates the increment introduced by the
/// [ReactionsContact](cellular_raza_concepts::ReactionsContact) aspect.
#[inline]
#[cfg_attr(feature = "tracing", instrument(skip_all))]
pub fn reactions_contact_adams_bashforth_3rd<C, A, F, Ri>(
    _cell: &mut C,
    aux_storage: &mut A,
) -> Result<(), CalcError>
where
    A: UpdateReactions<Ri>,
    A: UpdateReactionsContact<Ri, 2>,
    Ri: Xapy<F> + Clone,
    F: FromPrimitive + num::Float,
{
    // let dintra0 = aux_storage
    let dintra = aux_storage.get_conc();
    let conc_zero = dintra.xa(F::zero());
    let dintra_contact = <A as UpdateReactionsContact<Ri, 2>>::get_current_increment(&aux_storage);
    aux_storage.set_last_increment(dintra_contact.clone());
    let n_previous_values = aux_storage.n_previous_values();
    let mut old_intracellular_increments = aux_storage.previous_increments();
    let dintra_new = match n_previous_values {
        2 => adams_bashforth_3(
            dintra,
            [
                dintra_contact,
                old_intracellular_increments.next().unwrap().clone(),
                old_intracellular_increments.next().unwrap().clone(),
            ],
            F::one(),
            conc_zero,
        )?,
        1 => adams_bashforth_2(
            dintra,
            [
                dintra_contact,
                old_intracellular_increments.next().unwrap().clone(),
            ],
            F::one(),
            conc_zero,
        )?,
        _ => euler(dintra, dintra_contact, F::one(), conc_zero)?,
    };
    aux_storage.set_conc(dintra_new);
    Ok(())
}

#[cfg(test)]
mod test_solvers_reactions {
    use rand::SeedableRng;

    use crate::backend::chili::{UpdateReactions, local_reactions_use_increment};

    use super::*;

    #[test]
    fn exponential_decay_rk4() -> Result<(), super::super::SimulationError> {
        use cellular_raza_concepts::*;
        struct PlainCell {
            intracellular: f64,
            lambda: f64,
        }
        impl Intracellular<f64> for PlainCell {
            fn get_intracellular(&self) -> f64 {
                self.intracellular
            }
            fn set_intracellular(&mut self, intracellular: f64) {
                self.intracellular = intracellular;
            }
        }
        impl Reactions<f64> for PlainCell {
            fn calculate_intracellular_increment(
                &self,
                intracellular: &f64,
            ) -> Result<f64, CalcError> {
                Ok(-self.lambda * intracellular)
            }
        }
        struct AuxStorage {
            increment: f64,
        }
        impl UpdateReactions<f64> for AuxStorage {
            fn get_conc(&self) -> f64 {
                self.increment
            }
            fn incr_conc(&mut self, incr: f64) {
                self.increment += incr;
            }
            fn set_conc(&mut self, conc: f64) {
                self.increment = conc;
            }
        }
        let y0 = 33.0;
        let lambda = 0.2;
        let dt = 0.1;
        let exact_solution = |t: f64| -> f64 { y0 * (-lambda * t).exp() };

        let mut cell = PlainCell {
            intracellular: y0,
            lambda,
        };
        let mut aux_storage = AuxStorage { increment: 0.0 };
        let mut results_cr = vec![(0.0, cell.get_intracellular())];
        let mut t = 0.0;
        for _ in 0..100 {
            ReactionsRungeKuttaSolver::<4>::update(&mut cell, &mut aux_storage, dt)?;
            // This rng is just a placeholder and will not be used.
            let mut _rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
            local_reactions_use_increment(&mut cell, &mut aux_storage, dt, &mut _rng)?;
            t += dt;
            results_cr.push((t, cell.get_intracellular()));
        }
        for (t, res) in results_cr {
            let res_exact = exact_solution(t);
            assert!((res - res_exact).abs() < 1e-6);
        }
        Ok(())
    }
}

#[cfg(test)]
mod test_solvers {
    use super::*;

    #[test]
    fn euler_exp_decay() {
        let y0 = 27.0;
        let lambda = 0.3;
        let dt = 0.001;
        let rhs = |y: f64| -> f64 { -lambda * y };
        let exact_solution = |t: f64| -> f64 { y0 * (-lambda * t).exp() };

        // The following truncation error is taken from wikipedia:
        // https://en.wikipedia.org/wiki/Euler_method#Global_truncation_error
        // Lipschitz constant of RHS
        let lipschitz_constant = lambda;
        // Upper bound on second derivative of y
        let second_derivative_bound = y0 * lambda.powi(2);

        let expected_global_truncation_error = |t: f64| -> f64 {
            dt * second_derivative_bound / (2.0 * lipschitz_constant)
                * ((lipschitz_constant * t).exp() - 1.0)
        };

        let mut y = y0;
        let mut t = 0.0;
        for _ in 0..100 {
            let dy = rhs(y);
            y = euler(y, dy, dt, 0.0).unwrap();
            t += dt;
            let e = expected_global_truncation_error(t);
            assert!((y - exact_solution(t)).abs() < e);
        }
    }

    #[test]
    fn adams_bashforth_2_harmonic_oscillator() {
        #[derive(Clone, Copy, Debug)]
        struct Vec2(f32, f32);
        impl<'a> core::ops::Add<&'a Vec2> for Vec2 {
            type Output = Vec2;
            fn add(self, rhs: &'a Vec2) -> Self::Output {
                Vec2(self.0 + rhs.0, self.1 + rhs.1)
            }
        }
        impl core::ops::Add<Vec2> for Vec2 {
            type Output = Self;
            fn add(self, rhs: Vec2) -> Self::Output {
                Vec2(self.0 + rhs.0, self.1 + rhs.1)
            }
        }
        impl<'a> core::ops::Mul<f32> for &'a Vec2 {
            type Output = Vec2;
            fn mul(self, rhs: f32) -> Self::Output {
                Vec2(self.0 * rhs, self.1 * rhs)
            }
        }
        impl core::ops::Neg for Vec2 {
            type Output = Vec2;
            fn neg(self) -> Self::Output {
                Self(-self.0, -self.1)
            }
        }
        impl num::Zero for Vec2 {
            fn is_zero(&self) -> bool {
                self.0.is_zero() && self.1.is_zero()
            }
            fn set_zero(&mut self) {
                self.0 = 0.0;
                self.1 = 0.0;
            }
            fn zero() -> Self {
                Vec2(0., 0.)
            }
        }

        // Define parameters and initial values
        let y0 = Vec2(2.8347, 0.0);
        let omega: f32 = 0.319;
        let dt: f32 = 0.045;

        // Write down rhs and exact solution
        let rhs = |y: Vec2| -> Vec2 { Vec2(y.1, -omega.powi(2) * y.0) };
        let exact_solution =
            |t: f32| -> Vec2 { Vec2(y0.0 * (omega * t).cos(), -y0.0 * omega * (omega * t).sin()) };
        // This is taken from this math.stackexchange post:
        // https://math.stackexchange.com/questions/1326502/determine-the-local-truncation-error-of-the-following-method
        // Third order derivatives
        let third_derivative_bound = Vec2(y0.0 * omega.powi(3), y0.0 * omega.powi(3));
        let lipschitz_constant = Vec2(1.0, omega);
        let local_truncation_error = &third_derivative_bound * (5f32 / 12.0 * dt.powi(2));
        // See this wikipedia article:
        // https://en.wikipedia.org/wiki/Truncation_error_(numerical_integration)#Relationship_between_local_and_global_truncation_errors
        let global_truncation_error = |t: f32| -> Vec2 {
            Vec2(
                ((lipschitz_constant.0 * t).exp() - 1.0) * local_truncation_error.0
                    / dt
                    / lipschitz_constant.0,
                ((lipschitz_constant.1 * t).exp() - 1.0) * local_truncation_error.1
                    / dt
                    / lipschitz_constant.1,
            )
        };

        // Numerically integrate equation
        let mut t = 0.0;
        let mut y = y0.clone();
        let mut dy_storage = [
            rhs(exact_solution(t - dt)),
            rhs(exact_solution(t - 2.0 * dt)),
        ];

        for i in 0..100 {
            let dy = rhs(y);
            dy_storage[1] = dy_storage[0];
            dy_storage[0] = dy;
            y = adams_bashforth_2(y, dy_storage, dt, Vec2(0.0, 0.0)).unwrap();
            t += dt;
            let e = global_truncation_error(t);
            let d1 = (y.0 - exact_solution(t).0).abs();
            let d2 = (y.1 - exact_solution(t).1).abs();
            if i > 0 {
                assert!(d1 < e.0);
                assert!(d2 < e.1);
            }
        }
    }
}
