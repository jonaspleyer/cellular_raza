use cellular_raza::building_blocks::{CartesianCuboid, NewtonDamped2D};
use cellular_raza::concepts::*;
use cellular_raza_building_blocks::CartesianSubDomain;
use cellular_raza_core::backend::chili::{Settings, SimulationError};
use cellular_raza_core::storage::{StorageBuilder, StorageInterfaceLoad, StorageOption};
use cellular_raza_core::time::FixedStepsize;

use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

#[derive(Domain)]
struct MyDomain {
    #[DomainRngSeed]
    #[SortCells]
    cuboid: CartesianCuboid<f64, 2>,
    spring_strength: f64,
}

impl DomainCreateSubDomains<MySubDomain> for MyDomain {
    type VoxelIndex = [usize; 2];
    type SubDomainIndex = usize;

    fn create_subdomains(
        &self,
        n_subdomains: core::num::NonZeroUsize,
    ) -> Result<
        impl IntoIterator<Item = (Self::SubDomainIndex, MySubDomain, Vec<Self::VoxelIndex>)>,
        DecomposeError,
    > {
        Ok(self
            .cuboid
            .create_subdomains(n_subdomains)?
            .into_iter()
            .map(|(ind, subdomain, voxels)| {
                (
                    ind,
                    MySubDomain {
                        subdomain,
                        spring_strength: self.spring_strength,
                    },
                    voxels,
                )
            }))
    }
}

#[derive(SubDomain, Clone, Debug, Serialize)]
struct MySubDomain {
    #[Base]
    #[SortCells]
    #[Mechanics]
    subdomain: CartesianSubDomain<f64, 2>,
    spring_strength: f64,
}

impl SubDomainForce<Vector2<f64>, Vector2<f64>, Vector2<f64>, ()> for MySubDomain {
    fn calculate_custom_force(
        &self,
        pos: &Vector2<f64>,
        _vel: &Vector2<f64>,
        _: &(),
    ) -> Result<Vector2<f64>, cellular_raza_concepts::CalcError> {
        let force_x = -self.spring_strength * (pos.x - 0.0);
        let force_y = 0.0;
        Ok([force_x, force_y].into())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, CellAgent)]
struct MyAgent(#[Mechanics] NewtonDamped2D);

impl InteractionInformation<()> for MyAgent {
    fn get_interaction_information(&self) {}
}

#[test]
fn spring_single_particle() -> Result<(), SimulationError> {
    let spring_strength = 0.0032;
    let mass = 0.5;
    let dt = 0.001;
    let domain = MyDomain {
        cuboid: CartesianCuboid::from_boundaries_and_n_voxels([-77.0; 2], [77.0; 2], [3; 2])?,
        spring_strength,
    };
    let time = FixedStepsize::from_partial_save_interval(0.0, dt, 10.0, 0.1)?;
    let storage = StorageBuilder::new().priority([StorageOption::Memory]);
    let settings = Settings {
        time,
        storage,
        n_threads: 1.try_into().unwrap(),
        progressbar: None,
    };
    let x0 = 10.0;
    let agents = [MyAgent(NewtonDamped2D {
        pos: [x0, 0.0].into(),
        vel: [0.0, 0.0].into(),
        damping_constant: 0.0,
        mass,
    })];
    let storager = cellular_raza::core::backend::chili::run_simulation!(
        agents: agents,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, DomainForce],
    )?;
    let hists = storager.cells.load_all_element_histories()?;
    let (_, history) = hists.into_iter().next().unwrap();
    let positions = history
        .into_iter()
        .map(|(iter, (cbox, _))| (iter, cbox.cell.pos()));
    let omega = (spring_strength / mass).sqrt();
    for (iter, pos) in positions {
        let exact = x0 * (omega * iter as f64 * dt).cos();
        assert!((pos.x - exact).abs() < 1e-3);
    }
    Ok(())
}
