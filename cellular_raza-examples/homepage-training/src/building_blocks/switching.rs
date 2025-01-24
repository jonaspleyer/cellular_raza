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

    fn get_interaction_information(&self) -> I {
        use RadiusBasedInteraction::*;
        match self {
            Morse(pot) => {
                <MorsePotential as Interaction<T, T, T, I>>::
                    get_interaction_information(&pot)
            }
            Mie(pot) => {
                <MiePotential as Interaction<T, T, T, I>>::
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
    let inf1 = <SphericalInteraction as Interaction<
        nalgebra::Vector2<f64>,
        _,
        _,
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
    let inf2 = <SphericalInteraction as Interaction<
        nalgebra::Vector2<f64>,
        _,
        _,
        _,
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

impl<Pos, Vel, For, I1, I2, I3> Interaction<Pos, Vel, For, IInf2<I1, I2, I3>>
    for SphericalInteraction2
where
    MorsePotential: Interaction<Pos, Vel, For, I1>,
    MiePotential: Interaction<Pos, Vel, For, I2>,
    BoundLennardJones: Interaction<Pos, Vel, For, I3>,
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
fn test_interaction_box() {
    use core::ops::Deref;
    let agent = Agent {
        interaction: Box::new(MorsePotential {
            radius: 1.0,
            potential_stiffness: 0.3,
            cutoff: 3.0,
            strength: 0.1,
        }),
    };
    let inf = agent.get_interaction_information();
    assert_eq!(inf, 1.0);
}
