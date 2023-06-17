use cellular_raza_concepts::{
    cycle::CycleEvent,
    errors::{BoundaryError, CalcError},
};

use std::hash::Hash;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::concepts::SubDomain;

pub struct SubDomainBox<S, C, A>
where
    S: SubDomain<C>,
{
    pub subdomain: S,
    pub voxels: std::collections::BTreeMap<S::VoxelIndex, Voxel<C, A>>,
}

impl<S, C, A> SubDomainBox<S, C, A>
where
    S: SubDomain<C>,
{
    pub fn from_subdomain_and_cells(subdomain: S, cells: Vec<C>) -> Self
    where
        S::VoxelIndex: std::cmp::Eq + Hash + Ord,
        A: Default,
    {
        let voxel_indices = subdomain.get_all_indices();
        // TODO let voxels = subdomain.generate_all_voxels();
        let mut index_to_cells = cells
            .into_iter()
            .map(|cell| (subdomain.get_voxel_index_of(&cell).unwrap(), cell))
            .fold(
                std::collections::HashMap::new(),
                |mut acc, (index, cell)| {
                    let cells_in_voxel = acc.entry(index).or_insert(Vec::new());
                    cells_in_voxel.push((cell, A::default()));
                    acc
                },
            );
        let voxels = voxel_indices
            .into_iter()
            .map(|index| {
                let rng = ChaCha8Rng::seed_from_u64(1);
                let cells = index_to_cells.remove(&index).or(Some(Vec::new())).unwrap();
                (
                    index,
                    Voxel {
                        cells,
                        new_cells: Vec::new(),
                        id_counter: 0,
                        rng,
                    },
                )
            })
            .collect();
        Self { subdomain, voxels }
    }

    pub fn apply_boundary(&mut self) -> Result<(), BoundaryError> {
        self.voxels
            .iter_mut()
            .map(|(_, voxel)| voxel.cells.iter_mut())
            .flatten()
            .map(|(cell, _)| self.subdomain.apply_boundary(cell))
            .collect::<Result<(), BoundaryError>>()
    }

    // TODO this is not a boundary error!
    pub fn insert_cells(&mut self, new_cells: &mut Vec<(C, A)>) -> Result<(), BoundaryError>
    where
        S::VoxelIndex: Ord,
    {
        for cell in new_cells.drain(..) {
            let voxel_index = self.subdomain.get_voxel_index_of(&cell.0)?;
            self.voxels
                .get_mut(&voxel_index)
                .ok_or(BoundaryError {
                    message: "Could not find correct voxel for cell".to_owned(),
                })?
                .cells
                .push(cell);
        }
        Ok(())
    }

    pub fn update_cycle(&mut self) -> Result<(), CalcError>
    where
        C: cellular_raza_concepts::cycle::Cycle<C>,
        A: UpdateCycle,
    {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        self.voxels
            .iter_mut()
            .map(|(_, voxel)| voxel.cells.iter_mut())
            .flatten()
            .for_each(|(cell, aux_storage)| {
                if let Some(event) = C::update_cycle(&mut rng, &0.01, cell) {
                    aux_storage.add_cycle_event(event);
                }
            });
        Ok(())
    }
}

pub trait UpdateMechanics<P, V, const N: usize> {
    fn set_last_position(&mut self, pos: P);
    fn get_previous_positions(&self) -> [P; N];
    fn set_last_velocity(&mut self, vel: V);
    fn get_previous_velocities(&self) -> [V; N];
}

pub trait UpdateCycle {
    fn set_cycle_events(&mut self, events: Vec<CycleEvent>);
    fn get_cycle_events(&self) -> Vec<CycleEvent>;
    fn add_cycle_event(&mut self, event: CycleEvent);
}

pub struct AuxStorage<T> {
    properties: T,
}

impl<T> UpdateCycle for AuxStorage<(AuxStorage<T>, AuxStorageCycle)> {
    fn set_cycle_events(&mut self, events: Vec<CycleEvent>) {
        self.properties.1.set_cycle_events(events)
    }

    fn get_cycle_events(&self) -> Vec<CycleEvent> {
        self.properties.1.get_cycle_events()
    }

    fn add_cycle_event(&mut self, event: CycleEvent) {
        self.properties.1.add_cycle_event(event)
    }
}

impl<T> UpdateCycle for AuxStorage<T>
where
    T: UpdateCycle,
{
    fn add_cycle_event(&mut self, event: CycleEvent) {
        self.properties.add_cycle_event(event)
    }

    fn set_cycle_events(&mut self, events: Vec<CycleEvent>) {
        self.properties.set_cycle_events(events)
    }

    fn get_cycle_events(&self) -> Vec<CycleEvent> {
        self.properties.get_cycle_events()
    }
}

#[derive(Clone)]
pub struct AuxStorageCycle {
    cycle_events: Vec<CycleEvent>,
}

impl Default for AuxStorageCycle {
    fn default() -> Self {
        AuxStorageCycle {
            cycle_events: Vec::new(),
        }
    }
}

impl UpdateCycle for AuxStorageCycle {
    fn set_cycle_events(&mut self, events: Vec<CycleEvent>) {
        self.cycle_events = events;
    }

    fn get_cycle_events(&self) -> Vec<CycleEvent> {
        self.cycle_events.clone()
    }

    fn add_cycle_event(&mut self, event: CycleEvent) {
        self.cycle_events.push(event);
    }
}

#[cfg(test)]
mod test_cycle_implementations {
    use super::{AuxStorage, AuxStorageCycle, UpdateCycle};

    #[test]
    fn test_aux_storage_cycle_stage_1() {
        let aux_storage = AuxStorage {
            properties: AuxStorageCycle {
                cycle_events: Vec::new(),
            },
        };
        let events = aux_storage.get_cycle_events();
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn test_aux_storage_cycle_stage_2() {
        let aux_storage = AuxStorage {
            properties: (
                AuxStorage { properties: 1_f64 },
                AuxStorageCycle {
                    cycle_events: Vec::new(),
                },
            ),
        };
        let events = aux_storage.get_cycle_events();
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn test_aux_storage_cycle_stage_3() {
        let aux_storage = AuxStorage {
            properties: (
                AuxStorage {
                    properties: AuxStorage { properties: 1_f64 },
                },
                AuxStorageCycle {
                    cycle_events: Vec::new(),
                },
            ),
        };
        let events = aux_storage.get_cycle_events();
        assert_eq!(events.len(), 0);
    }
}

pub struct AuxStorageMechanics<P, V, const N: usize> {
    positions: [P; N],
    velocities: [V; N],
}

impl<P, V, const N: usize> UpdateMechanics<P, V, N> for AuxStorageMechanics<P, V, N>
where
    P: Clone,
    V: Clone,
{
    fn get_previous_positions(&self) -> [P; N] {
        self.positions.clone()
    }

    fn get_previous_velocities(&self) -> [V; N] {
        self.velocities.clone()
    }

    fn set_last_position(&mut self, pos: P) {
        todo!()
    }

    fn set_last_velocity(&mut self, vel: V) {
        todo!()
    }
}

pub struct Voxel<C, A> {
    pub cells: Vec<(C, A)>,
    pub new_cells: Vec<C>,
    pub id_counter: u64,
    pub rng: rand_chacha::ChaCha8Rng,
}

/* #[cfg(test)]
pub mod test {

    #[test]
    fn test_subdomains_to_boxes() {
        let min = 0.0;
        let max = 100.0;
        let config = SimulationConfig {
            cells: vec![1.0, 20.0, 26.0, 41.0, 56.0, 84.0, 95.0],
            domain: TestDomain {
                min,
                max,
                n_voxels: 8,
            },
        };

        let n_subdomains = 4;
        let subdomains_and_cells = config.into_subdomains(n_subdomains).unwrap();
        for (_subdomain_index, subdomain, cells) in subdomains_and_cells.into_iter() {
            let mut subdomain_box =
                SubDomainBox::<TestSubDomain, f64, AuxStorageCycle>::from_subdomain_and_cells(
                    subdomain, cells,
                );
            let mut new_cells = (0..10)
                .map(|i| {
                    (
                        subdomain_box.subdomain.min
                            + i as f64
                                * 0.01
                                * (subdomain_box.subdomain.min + subdomain_box.subdomain.max),
                        AuxStorageCycle::default(),
                    )
                })
                .collect::<Vec<_>>();
            println!(
                "Test {} {}",
                subdomain_box.subdomain.min, subdomain_box.subdomain.max
            );
            for (cell, _) in new_cells.iter() {
                println!("{:?}", cell);
            }
            subdomain_box.insert_cells(&mut new_cells).unwrap();
        }
    }
}*/
