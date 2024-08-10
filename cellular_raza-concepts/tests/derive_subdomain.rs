use cellular_raza_concepts::*;

struct DomainMechanics;

impl SubDomain for DomainMechanics {
    type VoxelIndex = usize;

    #[allow(unused)]
    fn get_neighbor_voxel_indices(&self, voxel_index: &Self::VoxelIndex) -> Vec<Self::VoxelIndex> {
        Vec::new()
    }

    fn get_all_indices(&self) -> Vec<Self::VoxelIndex> {
        Vec::new()
    }
}

impl SubDomainMechanics<f64, f64> for DomainMechanics {
    fn apply_boundary(&self, pos: &mut f64, vel: &mut f64) -> Result<(), BoundaryError> {
        *pos = 0.0;
        *vel = 1.0;
        Ok(())
    }
}

#[derive(SubDomain)]
struct DeriveDomain {
    #[Base]
    #[Mechanics]
    mechanics: DomainMechanics,
}

#[allow(unused)]
#[derive(SubDomain)]
struct DeriveDomainTuple(
    #[Base]
    #[Mechanics]
    DomainMechanics,
);

#[test]
fn derive_mechanics() {
    let derived_domain = DeriveDomain {
        mechanics: DomainMechanics,
    };
    let mut x = 1000.0;
    let mut y = 1000.0;
    derived_domain.apply_boundary(&mut x, &mut y).unwrap();
    assert_eq!(x, 0.0);
    assert_eq!(y, 1.0);
}

#[test]
fn derive_mechanics_tuple() {
    let derived_domain = DeriveDomainTuple(DomainMechanics);
    let mut x = 1000.0;
    let mut y = 1000.0;
    derived_domain.apply_boundary(&mut x, &mut y).unwrap();
    assert_eq!(x, 0.0);
    assert_eq!(y, 1.0);
}
