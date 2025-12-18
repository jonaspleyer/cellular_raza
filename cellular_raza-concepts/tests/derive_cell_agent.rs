use cellular_raza_concepts::{CalcError, InteractionInformation};

#[test]
fn derive_cycle() {
    use cellular_raza_concepts::*;
    use cellular_raza_concepts_derive::CellAgent;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    struct Agent;
    impl<NA> cellular_raza_concepts::Cycle<NA> for Agent {
        fn update_cycle(_rng: &mut ChaCha8Rng, _dt: &f64, _cell: &mut NA) -> Option<CycleEvent> {
            None
        }

        fn divide(_rng: &mut ChaCha8Rng, _cell: &mut NA) -> Result<NA, DivisionError> {
            unimplemented!()
        }
    }
    #[derive(CellAgent)]
    struct NewAgent1 {
        #[Cycle]
        _old_agent: Agent,
    }
    #[derive(CellAgent)]
    struct NewAgent2(#[Cycle] Agent);

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let mut new_agent1 = NewAgent1 { _old_agent: Agent };
    let mut new_agent2 = NewAgent2(Agent);
    assert!(<NewAgent1 as Cycle>::update_cycle(&mut rng, &0.1, &mut new_agent1).is_none());
    assert!(<NewAgent2 as Cycle>::update_cycle(&mut rng, &0.1, &mut new_agent2).is_none());
}

#[test]
fn derive_position() {
    use cellular_raza_concepts::Position;
    use cellular_raza_concepts_derive::CellAgent;
    struct PositionModel;
    impl Position<u8> for PositionModel {
        fn pos(&self) -> u8 {
            1
        }

        fn set_pos(&mut self, _pos: &u8) {}
    }
    #[derive(CellAgent)]
    struct NewAgent {
        #[Position]
        pos_model: PositionModel,
    }
    let new_agent = NewAgent {
        pos_model: PositionModel,
    };
    assert_eq!(new_agent.pos(), 1);
}

#[test]
fn derive_velocity() {
    use cellular_raza_concepts::Velocity;
    use cellular_raza_concepts_derive::CellAgent;
    struct VelocityModel;
    impl Velocity<i32> for VelocityModel {
        fn velocity(&self) -> i32 {
            1
        }

        fn set_velocity(&mut self, _velocity: &i32) {}
    }
    #[derive(CellAgent)]
    struct NewAgent {
        #[Velocity]
        velocity_model: VelocityModel,
    }
    let new_agent = NewAgent {
        velocity_model: VelocityModel,
    };
    assert_eq!(new_agent.velocity(), 1);
}

#[test]
fn derive_mechanics() {
    use cellular_raza_concepts::*;
    use cellular_raza_concepts_derive::CellAgent;
    struct MechanicsModel;
    impl cellular_raza_concepts::Mechanics<f32, f32, f32, f32> for MechanicsModel {
        fn calculate_increment(&self, force: f32) -> Result<(f32, f32), CalcError> {
            Ok((0.0, force))
        }

        fn get_random_contribution(
            &self,
            _rng: &mut rand_chacha::ChaCha8Rng,
            _dt: f32,
        ) -> Result<(f32, f32), RngError> {
            unimplemented!()
        }
    }
    #[derive(CellAgent)]
    struct NewAgent1 {
        #[Mechanics]
        mechanics: MechanicsModel,
    }

    let new_agent = NewAgent1 {
        mechanics: MechanicsModel,
    };
    let (dx, dv) = new_agent.calculate_increment(0.1).unwrap();
    assert_eq!(dx, 0.0);
    assert_eq!(dv, 0.1);
}

#[test]
fn derive_interaction() {
    use cellular_raza_concepts::*;
    use cellular_raza_concepts_derive::CellAgent;
    struct InteractionModel;
    impl cellular_raza_concepts::Interaction<f32, f32, f32, ()> for InteractionModel {
        fn calculate_force_between(
            &self,
            _own_pos: &f32,
            _ext_pos: &f32,
            _own_vel: &f32,
            _ext_vel: &f32,
            _ext_info: &(),
        ) -> Result<(f32, f32), CalcError> {
            unimplemented!()
        }
    }
    impl cellular_raza_concepts::InteractionInformation<()> for InteractionModel {
        fn get_interaction_information(&self) {}
    }
    #[derive(CellAgent)]
    struct NewAgent1 {
        #[Interaction]
        interaction: InteractionModel,
    }

    let newagent = NewAgent1 {
        interaction: InteractionModel,
    };
    assert_eq!(newagent.get_interaction_information(), ());
}

#[test]
fn derive_interaction_generics() {
    use cellular_raza_concepts::Interaction;
    use cellular_raza_concepts_derive::CellAgent;

    struct InteractionModel<const D: usize> {
        index: [usize; D],
    }

    impl<const D: usize> Interaction<f32, f32, f32, [usize; D]> for InteractionModel<D> {
        fn calculate_force_between(
            &self,
            _own_pos: &f32,
            _own_vel: &f32,
            _ext_pos: &f32,
            _ext_vel: &f32,
            _ext_info: &[usize; D],
        ) -> Result<(f32, f32), cellular_raza_concepts::CalcError> {
            Ok((0.0, 0.0))
        }
    }

    impl<const D: usize> InteractionInformation<[usize; D]> for InteractionModel<D> {
        fn get_interaction_information(&self) -> [usize; D] {
            self.index
        }
    }

    #[derive(CellAgent)]
    struct NewAgent<const D: usize> {
        #[Interaction]
        interaction: InteractionModel<D>,
    }

    let my_agent = NewAgent {
        interaction: InteractionModel { index: [1, 2, 3] },
    };
    assert_eq!(my_agent.get_interaction_information(), [1, 2, 3]);
}

#[test]
fn derive_neighbor_interaction() {
    use cellular_raza_concepts::NeighborSensing;
    use cellular_raza_concepts_derive::CellAgent;

    struct Senser;

    impl InteractionInformation<i8> for Senser {
        fn get_interaction_information(&self) -> i8 {
            0
        }
    }

    impl NeighborSensing<f32, usize, i8> for Senser {
        fn accumulate_information(
            &self,
            _: &f32,
            _: &f32,
            ext_inf: &i8,
            accumulator: &mut usize,
        ) -> Result<(), CalcError> {
            *accumulator += *ext_inf as usize;
            Ok(())
        }

        fn react_to_neighbors(&mut self, _: &usize) -> Result<(), CalcError> {
            Ok(())
        }

        fn clear_accumulator(accumulator: &mut usize) {
            *accumulator = 0;
        }
    }

    #[derive(CellAgent)]
    struct Agent {
        #[NeighborSensing]
        senser: Senser,
    }

    let senser = Senser;

    let mut accumulator = 0;
    for i in 0..10 {
        senser
            .accumulate_information(&0., &0., &i, &mut accumulator)
            .unwrap();
    }

    let mut cell = Agent { senser };
    cell.react_to_neighbors(&accumulator).unwrap();
    cell.get_interaction_information();
}
