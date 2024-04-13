use cellular_raza_concepts::domain_new::*;
use cellular_raza_concepts::DecomposeError;

#[allow(unused)]
struct Agent {
    pos: f32,
}

#[allow(unused)]
struct MyDomain {
    x_min: f32,
    x_max: f32,
}

#[allow(unused)]
struct MySubDomain {
    x_min: f32,
    x_max: f32,
}

impl SubDomain for MySubDomain {
    type VoxelIndex = u8;

    fn get_neighbor_voxel_indices(&self, _voxel_index: &Self::VoxelIndex) -> Vec<Self::VoxelIndex> {
        Vec::new()
    }

    fn get_all_indices(&self) -> Vec<Self::VoxelIndex> {
        vec![1]
    }
}

impl Domain<Agent, MySubDomain> for MyDomain {
    type VoxelIndex = u8;
    type SubDomainIndex = usize;

    fn decompose(
        self,
        _: core::num::NonZeroUsize,
        cells: Vec<Agent>,
    ) -> Result<
        DecomposedDomain<Self::SubDomainIndex, MySubDomain, Agent>,
        cellular_raza_concepts::DecomposeError,
    >
    where
        MySubDomain: SubDomain,
    {
        Ok(DecomposedDomain {
            n_subdomains: 1.try_into().unwrap(),
            index_subdomain_cells: vec![(
                1,
                MySubDomain {
                    x_min: self.x_min,
                    x_max: self.x_max,
                },
                cells,
            )],
            neighbor_map: std::collections::HashMap::from([(1, Vec::new())]),
            rng_seed: 1,
        })
    }
}

#[allow(unused)]
#[derive(Domain)]
struct DerivedDomain {
    #[Base]
    my_domain: MyDomain,
}

#[test]
fn derive_domain() {
    let domain = DerivedDomain {
        my_domain: MyDomain {
            x_min: 3000.0,
            x_max: 3001.0,
        },
    };
    let voxel_indices = domain.get_all_voxel_indices();
    assert_eq!(voxel_indices, vec![1]);
}
