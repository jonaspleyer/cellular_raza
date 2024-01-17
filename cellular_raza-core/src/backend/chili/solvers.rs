// Use the two-step Adams-Bashforth method. See also: https://en.wikipedia.org/wiki/Linear_multistep_method
// TODO We should be able to implement arbitrary steppers here
pub fn mechanics_adams_bashforth<C, A, Pos, Vel, For, Float, const N: usize> (
    cell: &mut C,
    aux_storage: &mut A,
    dt: Float,
)
where
    A: super::aux_storage::UpdateMechanics<Pos, Vel, For, Float, N>,
    C: cellular_raza_concepts::mechanics::Mechanics<Pos, Vel, For, Float>,
{
    // TODO
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
}
