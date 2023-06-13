#[macro_export]
macro_rules! implement_cell_types {
    ($pos:ty, $force:ty, $information:ty, $velocity:ty, $voxel:ty, $index:ty, [$($celltype:ident),+]) => {
        use serde::{Serialize,Deserialize};

        #[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
        pub enum CellAgentType {
            $($celltype($celltype)),+
        }

        impl cellular_raza_concepts::interaction::Interaction<$pos, $force, $information> for CellAgentType {
            fn get_interaction_information(&self) -> Option<$information> {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.get_interaction_information(),)+
                }
            }

            fn calculate_force_between(&self, own_pos: &$pos, ext_pos: &$pos, ext_information: &Option<$information>) -> Option<Result<$force, cellular_raza_concepts::errors::CalcError>> {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.calculate_force_between(own_pos, ext_pos, ext_information),)+
                }
            }
        }

        impl cellular_raza_concepts::cycle::Cycle<CellAgentType> for CellAgentType {
            fn update_cycle(rng: &mut rand_chacha::ChaCha8Rng, dt: &f64, cell: &mut CellAgentType) -> Option<cellular_raza_concepts::cycle::CycleEvent> {
                match c {
                    $(CellAgentType::$celltype(cell) => $celltype::update_cycle(rng, dt, cell),)+
                }
            }

            fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut CellAgentType) -> Result<Option<CellAgentType>, cellular_raza_concepts::errors::DivisionError> {
                match c {
                    $(CellAgentType::$celltype(cell) => match $celltype::divide(rng, cell)? {
                        Some(new_cell) => Ok(Some(CellAgentType::$celltype(new_cell))),
                        None => Ok(None),
                    },)+
                }
            }
        }

        impl cellular_raza_concepts::mechanics::Mechanics<$pos, $force, $velocity> for CellAgentType {
            fn pos(&self) -> $pos {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.pos(),)+
                }
            }

            fn velocity(&self) -> $velocity {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.velocity(),)+
                }
            }

            fn set_pos(&mut self, pos: &$pos) {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.set_pos(pos),)+
                }
            }

            fn set_velocity(&mut self, velocity: &$velocity) {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.set_velocity(velocity),)+
                }
            }

            fn calculate_increment(&self, force: $force) -> Result<($pos, $velocity), cellular_raza_concepts::errors::CalcError> {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.calculate_increment(force),)+
                }
            }
        }

        impl crate::plotting::spatial::PlotSelf for CellAgentType
        {
            fn plot_self<Db>(&self, root: &mut plotters::prelude::DrawingArea<Db, plotters::coord::cartesian::Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>>) -> Result<(), cellular_raza_concepts::errors::DrawingError>
            where
                Db: plotters::backend::DrawingBackend,
            {
                match self {
                    $(CellAgentType::$celltype(cell) => cell.plot_self(root),)+
                }
            }
        }

        unsafe impl Send for CellAgentType {}
        unsafe impl Sync for CellAgentType {}
    }
}

#[macro_export]
macro_rules! define_simulation_types {
    (
        Position:   $position:ty,
        Force:      $force:ty,
        Information:$information:ty,
        Velocity:   $velocity:ty,
        CellTypes:  [$($celltype:ident),+],
        Voxel:      $voxel:ty,
        Index:      $index:ty,
        Domain:     $domain:ty,
    ) => {
        // Create an enum containing all cell types
        implement_cell_types!($position, $force, $information, $velocity, $voxel, $index, [$($celltype),+]);

        pub type SimTypePosition = $position;
        pub type SimTypeForce = $force;
        pub type SimTypeVelocity = $velocity;
        pub type SimTypeVoxel = $voxel;
        pub type SimTypeIndex = $index;
        pub type SimTypeDomain = $domain;
    }
}

#[macro_export]
macro_rules! create_sim_supervisor {
    ($setup:expr) => {
        Result::<
            SimulationSupervisor<
                SimTypePosition,
                SimTypeForce,
                SimTypeVelocity,
                CellAgentType,
                SimTypeIndex,
                SimTypeVoxel,
                SimTypeDomain,
            >,
            Box<dyn std::error::Error>,
        >::from($setup)
        .unwrap()
    };
}
