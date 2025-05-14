use cellular_raza::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
pub enum RadiusBasedInteraction {
    Morse(MorsePotential),
    Mie(MiePotential),
}

impl<T, I> Interaction<T, T, T, I> for RadiusBasedInteraction
where
    MorsePotential: Interaction<T, T, T, I>,
    MiePotential: Interaction<T, T, T, I>,
{
    fn calculate_force_between(
        &self,
        own_pos: &T,
        own_vel: &T,
        ext_pos: &T,
        ext_vel: &T,
        inf: &I,
    ) -> Result<(T, T), CalcError> {
        use RadiusBasedInteraction::*;
        match self {
            Morse(pot) => pot.calculate_force_between(
                own_pos, own_vel, ext_pos, ext_vel, inf,
            ),
            Mie(pot) => pot.calculate_force_between(
                own_pos, own_vel, ext_pos, ext_vel, inf,
            ),
        }
    }
}

impl<I> InteractionInformation<I> for RadiusBasedInteraction
where
    MorsePotential: InteractionInformation<I>,
    MiePotential: InteractionInformation<I>,
{
    fn get_interaction_information(&self) -> I {
        use RadiusBasedInteraction::*;
        match self {
            Morse(pot) => {
                <MorsePotential as InteractionInformation<I>>::
                    get_interaction_information(&pot)
            }
            Mie(pot) => {
                <MiePotential as InteractionInformation<I>>::
                    get_interaction_information(&pot)
            }
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub enum SphericalInteraction {
    Morse(MorsePotential),
    BLJ(BoundLennardJones),
}

#[derive(Clone)]
pub enum IInf<I1, I2> {
    Morse(I1),
    BLJ(I2),
}

impl<T, I1, I2> Interaction<T, T, T, IInf<I1, I2>> for SphericalInteraction
where
    MorsePotential: Interaction<T, T, T, I1>,
    BoundLennardJones: Interaction<T, T, T, I2>,
{
    fn calculate_force_between(
        &self,
        own_pos: &T,
        own_vel: &T,
        ext_pos: &T,
        ext_vel: &T,
        ext_info: &IInf<I1, I2>,
    ) -> Result<(T, T), CalcError> {
        use SphericalInteraction::*;
        match (self, ext_info) {
            (Morse(pot), IInf::Morse(inf)) => pot.calculate_force_between(
                own_pos, own_vel, ext_pos, ext_vel, inf,
            ),
            (BLJ(pot), IInf::BLJ(inf)) => pot.calculate_force_between(
                own_pos, own_vel, ext_pos, ext_vel, inf,
            ),
            _ => Err(CalcError(format!(
                "interaction potential and obtained\
                information did not match"
            ))),
        }
    }
}

impl<I1, I2> InteractionInformation<IInf<I1, I2>> for SphericalInteraction
where
    MorsePotential: InteractionInformation<I1>,
    BoundLennardJones: InteractionInformation<I2>,
{
    fn get_interaction_information(&self) -> IInf<I1, I2> {
        use SphericalInteraction::*;
        match self {
            Morse(pot) => IInf::Morse(pot.get_interaction_information()),
            // In this case, the BLJ potential returns ().
            // Thus this is equivalent to
            // BLJ(_) => (),
            BLJ(pot) => IInf::BLJ(pot.get_interaction_information()),
        }
    }
}

#[test]
fn test_interaction_different_inf_types() {
    let i1 = SphericalInteraction::Morse(MorsePotential {
        radius: 1.0,
        potential_stiffness: 0.5,
        cutoff: 3.0,
        strength: 0.1,
    });
    let inf1 = <SphericalInteraction as InteractionInformation<
        _,
    >>::get_interaction_information(&i1);
    match inf1 {
        IInf::Morse(i1) => assert_eq!(i1, 1.0),
        IInf::BLJ(_) => assert!(false),
    }

    let i2 = SphericalInteraction::BLJ(BoundLennardJones {
        epsilon: 1.0,
        sigma: 1.0,
        bound: 4.0,
        cutoff: 4.0,
    });
    let inf2 = <SphericalInteraction as InteractionInformation<
        _
    >>::get_interaction_information(&i2);
    match inf2 {
        IInf::Morse(_) => assert!(false),
        IInf::BLJ(i2) => assert_eq!((), i2),
    }
}

#[allow(unused)]
#[derive(Clone)]
enum SphericalInteraction2 {
    Morse(MorsePotential),
    Mie(MiePotential),
    BLJ(BoundLennardJones),
}

enum IInf2<I1, I2, I3> {
    Morse(I1),
    Mie(I2),
    BLJ(I3),
}

impl<I1, I2, I3> InteractionInformation<IInf2<I1, I2, I3>>
    for SphericalInteraction2
where
    MorsePotential: InteractionInformation<I1>,
    MiePotential: InteractionInformation<I2>,
    BoundLennardJones: InteractionInformation<I3>,
{
    fn get_interaction_information(&self) -> IInf2<I1, I2, I3> {
        match self {
            SphericalInteraction2::Morse(pot) => {
                IInf2::Morse(pot.get_interaction_information())
            }
            SphericalInteraction2::Mie(pot) => {
                IInf2::Mie(pot.get_interaction_information())
            }
            SphericalInteraction2::BLJ(pot) => {
                IInf2::BLJ(pot.get_interaction_information())
            }
        }
    }
}

impl<Pos, Vel, For, I1, I2, I3> Interaction<Pos, Vel, For, IInf2<I1, I2, I3>>
    for SphericalInteraction2
where
    MorsePotential: Interaction<Pos, Vel, For, I1>,
    MiePotential: Interaction<Pos, Vel, For, I2>,
    BoundLennardJones: Interaction<Pos, Vel, For, I3>,
{
    fn calculate_force_between(
        &self,
        own_pos: &Pos,
        own_vel: &Vel,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_info: &IInf2<I1, I2, I3>,
    ) -> Result<(For, For), CalcError> {
        match (self, ext_info) {
            (SphericalInteraction2::Morse(pot), IInf2::Morse(ext_info)) => pot
                .calculate_force_between(
                    own_pos, own_vel, ext_pos, ext_vel, ext_info,
                ),
            (SphericalInteraction2::Mie(pot), IInf2::Mie(ext_info)) => pot
                .calculate_force_between(
                    own_pos, own_vel, ext_pos, ext_vel, ext_info,
                ),
            (SphericalInteraction2::BLJ(pot), IInf2::BLJ(ext_info)) => pot
                .calculate_force_between(
                    own_pos, own_vel, ext_pos, ext_vel, ext_info,
                ),
            _ => Err(CalcError(format!(
                "interaction type and information type are not matching"
            ))),
        }
    }
}

#[test]
fn test_interaction_all_different_types() {
    let int1 = SphericalInteraction2::Morse(MorsePotential {
        radius: 1.0,
        potential_stiffness: 0.3,
        cutoff: 4.0,
        strength: 0.3,
    });
    let int2 = SphericalInteraction2::Mie(MiePotential {
        radius: 1.0,
        strength: 0.01,
        bound: 10.0,
        cutoff: 8.0,
        en: 2.0,
        em: 3.0,
    });

    let inf1 = <SphericalInteraction2 as InteractionInformation<
        _,
    >>::get_interaction_information(&int1);
    let inf2 = <SphericalInteraction2 as InteractionInformation<
        _,
    >>::get_interaction_information(&int2);

    match (inf1, inf2) {
        (IInf2::Morse(_), IInf2::Mie(_)) => (),
        _ => assert!(false),
    }
}
