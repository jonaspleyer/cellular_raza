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
) -> Result<(), CalcError>
where
    A: super::aux_storage::UpdateMechanics<Pos, Vel, For, 0>,
    C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
    C: cellular_raza_concepts::Position<Pos>,
    C: cellular_raza_concepts::Velocity<Vel>,
    Pos: Xapy<Float> + num::Zero + Clone,
    Vel: Xapy<Float> + num::Zero + Clone,
    Float: Copy,
{
    let force = aux_storage.get_current_force();
    let velocity = cell.velocity();
    let position = cell.pos();

    let (dx, dv) = cell.calculate_increment(force)?;

    // Update values in the aux_storage
    aux_storage.set_last_position(dx.clone());
    aux_storage.set_last_velocity(dv.clone());
    aux_storage.clear_forces();

    // Calculate new position and velocity of cell
    let new_position = euler(position, dx, dt)?;
    let new_velocity = euler(velocity, dv, dt)?;
    cell.set_pos(&new_position);
    cell.set_velocity(&new_velocity);
    Ok(())
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
) -> Result<(), CalcError>
where
    A: super::aux_storage::UpdateMechanics<Pos, Vel, For, 2>,
    C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
    C: cellular_raza_concepts::Position<Pos>,
    C: cellular_raza_concepts::Velocity<Vel>,
    Pos: Xapy<Float> + num::Zero + Clone,
    Vel: Xapy<Float> + num::Zero + Clone,
    Float: num::Float + FromPrimitive,
{
    let force = aux_storage.get_current_force();
    let velocity = cell.velocity();
    let position = cell.pos();

    let (dx, dv) = cell.calculate_increment(force)?;

    // Update values in the aux_storage
    aux_storage.set_last_position(dx.clone());
    aux_storage.set_last_velocity(dv.clone());
    aux_storage.clear_forces();

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
            )?,
            adams_bashforth_3(
                velocity,
                [
                    dv,
                    old_vel_increments.next().unwrap().clone(),
                    old_vel_increments.next().unwrap().clone(),
                ],
                dt,
            )?,
        ),
        1 => (
            adams_bashforth_2(
                position,
                [dx, old_pos_increments.next().unwrap().clone()],
                dt,
            )?,
            adams_bashforth_2(
                velocity,
                [dv, old_vel_increments.next().unwrap().clone()],
                dt,
            )?,
        ),
        _ => (euler(position, dx, dt)?, euler(velocity, dv, dt)?),
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
) -> Result<(), CalcError>
where
    A: super::aux_storage::UpdateMechanics<Pos, Vel, For, 1>,
    C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
    C: cellular_raza_concepts::Position<Pos>,
    C: cellular_raza_concepts::Velocity<Vel>,
    Pos: Xapy<Float> + num::Zero + Clone,
    Vel: Xapy<Float> + num::Zero + Clone,
    Float: num::Float + FromPrimitive,
{
    let force = aux_storage.get_current_force();
    let velocity = cell.velocity();
    let position = cell.pos();

    let (dx, dv) = cell.calculate_increment(force)?;

    // Update values in the aux_storage
    aux_storage.set_last_position(dx.clone());
    aux_storage.set_last_velocity(dv.clone());
    aux_storage.clear_forces();

    // Calculate new position and velocity of cell
    let n_previous_values = aux_storage.n_previous_values();
    let mut old_pos_increments = aux_storage.previous_positions();
    let mut old_vel_increments = aux_storage.previous_velocities();
    /* let (new_position, new_velocity) = match n_previous_values {
        1 => adams_bashforth_2(
            position,
            [dx, old_pos_increments.next().unwrap().clone()],
            velocity,
            [dv, old_vel_increments.next().unwrap().clone()],
            dt,
        )?,
        _ => euler(position, dx, velocity, dv, dt)?,
    };*/
    let (new_position, new_velocity) = match n_previous_values {
        1 => (
            adams_bashforth_2(
                position,
                [dx, old_pos_increments.next().unwrap().clone()],
                dt,
            )?,
            adams_bashforth_2(
                velocity,
                [dv, old_vel_increments.next().unwrap().clone()],
                dt,
            )?,
        ),
        _ => (euler(position, dx, dt)?, euler(velocity, dv, dt)?),
    };
    cell.set_pos(&new_position);
    cell.set_velocity(&new_velocity);
    Ok(())
}

#[inline]
fn euler<X, F>(
    x: X,
    dx: X,
    // v: Vel,
    // dv: Vel,
    dt: F,
) -> Result<X, CalcError>
// ) -> Result<(X, Vel), CalcError>
where
    X: Xapy<F> + num::Zero,
    F: Copy,
{
    let x_new = dx.xapy(dt, &x);
    Ok(x_new)
}

#[inline]
fn adams_bashforth_3<X, F>(
    x: X,
    dx: [X; 3],
    dt: F,
) -> Result<X, CalcError>
where
    X: Xapy<F> + num::Zero,
    F: Copy + FromPrimitive + num::Float,
{
    let f0 = F::from_isize(23).unwrap() / F::from_isize(12).unwrap();
    let f1 = -F::from_isize(16).unwrap() / F::from_isize(12).unwrap();
    let f2 = F::from_isize(5).unwrap() / F::from_isize(12).unwrap();

    let [dx0, dx1, dx2] = dx;

    let x_new = dx0
        .xapy(f0, &dx1.xapy(f1, &dx2.xapy(f2, &X::zero())))
        .xapy(dt, &x);
    Ok(x_new)
}

#[inline]
fn adams_bashforth_2<X, F>(
    x: X,
    dx: [X; 2],
    // v: Vel,
    // dv: [Vel; 2],
    dt: F,
) -> Result<X, CalcError>
where
    X: Xapy<F> + num::Zero,
    F: Copy + FromPrimitive + num::Float,
{
    let f0 = F::from_isize(3).unwrap() / F::from_isize(2).unwrap();
    let f1 = -F::from_isize(1).unwrap() / F::from_isize(2).unwrap();

    let [dx0, dx1] = dx;

    let x_new = dx0.xapy(f0, &dx1.xapy(f1, &X::zero())).xapy(dt, &x);
    Ok(x_new)
}

/// TODO
#[inline]
#[cfg_attr(feature = "tracing", instrument(skip_all))]
pub fn reactions_intracellular_runge_kutta_4th<C, A, F, Ri>(
    cell: &mut C,
    _aux_storage: &mut A,
    dt: F,
) -> Result<(), CalcError>
where
    C: cellular_raza_concepts::Reactions<Ri>,
    // A: super::UpdateReactions<Ri>,
    F: num::Float,
    Ri: num::Zero + Xapy<F>,
{
    // Constants
    let two = F::one() + F::one();
    let six = two + two + two;

    let intra = cell.get_intracellular();

    // Calculate the intermediate steps
    let dintra1 = cell.calculate_intracellular_increment(&intra)?;
    let dintra2 = cell.calculate_intracellular_increment(&dintra1.xapy(dt / two, &intra))?;
    let dintra3 = cell.calculate_intracellular_increment(&dintra2.xapy(dt / two, &intra))?;
    let dintra4 = cell.calculate_intracellular_increment(&dintra3.xapy(dt, &intra))?;
    let dintra = dintra1.xapy(
        F::one() / six,
        &dintra2.xapy(
            two / six,
            &dintra3.xapy(two / six, &dintra4.xapy(F::one() / six, &Ri::zero())),
        ),
    );

    // Update the internal value of the cell
    cell.set_intracellular(dintra.xapy(dt, &intra));
    Ok(())
}
