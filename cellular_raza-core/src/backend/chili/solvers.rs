use cellular_raza_concepts::errors::CalcError;

pub fn mechanics_euler<C, A, Pos, Vel, For, Float>(
    cell: &mut C,
    aux_storage: &mut A,
    dt: Float,
) -> Result<(), CalcError>
where
    A: super::aux_storage::UpdateMechanics<Pos, Vel, For, Float, 0>,
    C: cellular_raza_concepts::mechanics::Mechanics<Pos, Vel, For, Float>,
    Pos: core::ops::Mul<Float, Output = Pos>,
    Pos: core::ops::Add<Pos, Output = Pos>,
    Vel: core::ops::Mul<Float, Output = Vel>,
    Vel: core::ops::Add<Vel, Output = Vel>,
    Vel: num::Zero,
    Float: Copy,
{
    let force = aux_storage.get_current_force();
    let velocity = cell.velocity();
    let position = cell.pos();

    let (dx, dv) = cell.calculate_increment(force)?;
    let new_position = position + dx * dt;
    let new_velocity = velocity + dv * dt;
    cell.set_pos(&new_position);
    cell.set_velocity(&new_velocity);

    aux_storage.clear_forces();
    Ok(())
}

// Use the two-step Adams-Bashforth method. See also: https://en.wikipedia.org/wiki/Linear_multistep_method
// TODO We should be able to implement arbitrary steppers here
pub fn mechanics_adams_bashforth<C, A, Pos, Vel, For, Float, const N: usize>(
    cell: &mut C,
    aux_storage: &mut A,
    dt: Float,
) -> Result<(), CalcError>
where
    A: super::aux_storage::UpdateMechanics<Pos, Vel, For, Float, N>,
    C: cellular_raza_concepts::mechanics::Mechanics<Pos, Vel, For, Float>,
    Pos: core::ops::Mul<Float, Output = Pos>,
    Pos: core::ops::Add<Pos, Output = Pos>,
    Vel: core::ops::Mul<Float, Output = Vel>,
    Vel: core::ops::Add<Vel, Output = Vel>,
    Vel: num::Zero,
    Float: Copy,
{
    let force = aux_storage.get_current_force();
    // let velocity = aux_storage.previous_velocities().last().or_else(|| Some(&Vel::zero()));
    // let position = aux_storage.previous_positions().last().or_else(|| Some())
    let velocity = cell.velocity();
    let position = cell.pos();

    let (dx, dv) = cell.calculate_increment(force)?;
    let new_position = position + dx * dt;
    let new_velocity = velocity + dv * dt;
    cell.set_pos(&new_position);
    cell.set_velocity(&new_velocity);

    aux_storage.clear_forces();

    // TODO make this not be a euler solver!
    /* match (
        aux_storage.inc_pos_back_1.clone(),
        aux_storage.inc_pos_back_2.clone(),
        aux_storage.inc_vel_back_1.clone(),
        aux_storage.inc_vel_back_2.clone(),
    ) {
        // If all values are present, use the Adams-Bashforth 3rd order
        (
            Some(inc_pos_back_1),
            Some(inc_pos_back_2),
            Some(inc_vel_back_1),
            Some(inc_vel_back_2),
        ) => {
            cell.set_pos(
                &(cell.pos() + dx.clone() * (23.0 / 12.0) * *dt
                    - inc_pos_back_1 * (16.0 / 12.0) * *dt
                    + inc_pos_back_2 * (5.0 / 12.0) * *dt),
            );
            cell.set_velocity(
                &(cell.velocity() + dv.clone() * (23.0 / 12.0) * *dt
                    - inc_vel_back_1 * (16.0 / 12.0) * *dt
                    + inc_vel_back_2 * (5.0 / 12.0) * *dt),
            );
        }
        // Otherwise check and use the 2nd order
        (Some(inc_pos_back_1), None, Some(inc_vel_back_1), None) => {
            cell.set_pos(
                &(cell.pos() + dx.clone() * (3.0 / 2.0) * *dt
                    - inc_pos_back_1 * (1.0 / 2.0) * *dt),
            );
            cell.set_velocity(
                &(cell.velocity() + dv.clone() * (3.0 / 2.0) * *dt
                    - inc_vel_back_1 * (1.0 / 2.0) * *dt),
            );
        }
        // This case should only exists when the cell was first created
        // Then use the Euler Method
        _ => {
            cell.set_pos(&(cell.pos() + dx.clone() * *dt));
            cell.set_velocity(&(cell.velocity() + dv.clone() * *dt));
        }
    }

    // Afterwards update values in auxiliary storage
    aux_storage.force = For::zero();
    aux_storage.inc_pos_back_1 = Some(dx);
    aux_storage.inc_vel_back_1 = Some(dv);*/
    Ok(())
}
