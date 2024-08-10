use cellular_raza_concepts::*;

#[allow(unused)]
struct Agent {
    pos: f32,
}

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

#[allow(unused)]
#[derive(Domain)]
struct TupleDomain(#[Base] MyDomain);

#[test]
fn derive_domain() {
    let domain = DerivedDomain {
        my_domain: MyDomain {
            x_min: 3000.0,
            x_max: 3001.0,
        },
    };
    let decomposed_domain = domain.decompose(2.try_into().unwrap(), vec![]).unwrap();
    assert_eq!(decomposed_domain.rng_seed, 1);
}

impl SortCells<Agent> for MyDomain {
    type VoxelIndex = u8;

    fn get_voxel_index_of(
        &self,
        _cell: &Agent,
    ) -> Result<Self::VoxelIndex, cellular_raza_concepts::BoundaryError> {
        Ok(1)
    }
}

#[derive(Domain)]
#[DomainPartialDerive]
struct DerivedDomain2 {
    #[SortCells]
    my_domain: MyDomain,
}

#[allow(unused)]
#[derive(Domain)]
#[DomainPartialDerive]
struct DeriveDomain2Tuple(#[SortCells] MyDomain);

#[test]
fn derive_sort_cells() {
    let domain = DerivedDomain2 {
        my_domain: MyDomain {
            x_min: 0.0,
            x_max: 3000000.0,
        },
    };
    let agent = Agent { pos: 3.0 };
    let index = domain.get_voxel_index_of(&agent);
    assert!(index.is_ok_and(|x| x == 1));
}

impl DomainRngSeed for MyDomain {
    fn get_rng_seed(&self) -> u64 {
        0
    }
}

#[derive(Domain)]
struct DerivedDomain3 {
    #[DomainRngSeed]
    my_domain: MyDomain,
}

#[allow(unused)]
#[derive(Domain)]
struct DerivedDomain3Tuple(#[DomainRngSeed] MyDomain);

#[test]
fn derive_rng_seed() {
    let domain = DerivedDomain3 {
        my_domain: MyDomain {
            x_min: -3.0,
            x_max: 0.01,
        },
    };
    assert_eq!(0, domain.get_rng_seed());
}

impl DomainCreateSubDomains<MySubDomain> for MyDomain {
    type VoxelIndex = u8;
    type SubDomainIndex = usize;

    fn create_subdomains(
        &self,
        n_subdomains: core::num::NonZeroUsize,
    ) -> Result<
        impl IntoIterator<Item = (Self::SubDomainIndex, MySubDomain, Vec<Self::VoxelIndex>)>,
        DecomposeError,
    > {
        let dx = (self.x_max - self.x_min)
            / <usize as From<core::num::NonZeroUsize>>::from(n_subdomains) as f32;
        Ok((0..n_subdomains.into()).map(move |n| {
            (
                0,
                MySubDomain {
                    x_min: self.x_min.clone() + n as f32 * dx,
                    x_max: self.x_min.clone() + (n + 1) as f32 * dx,
                },
                Vec::new(),
            )
        }))
    }
}

#[derive(Domain)]
#[DomainPartialDerive]
struct DerivedDomain4 {
    #[DomainCreateSubDomains]
    domain: MyDomain,
}

#[allow(unused)]
#[derive(Domain)]
#[DomainPartialDerive]
struct DerivedDomain4Tuple(#[DomainCreateSubDomains] MyDomain);

#[test]
fn derive_create_subdomains() {
    let derived_domain = DerivedDomain4 {
        domain: MyDomain {
            x_min: -1000.0,
            x_max: -1001.0,
        },
    };
    let n_subdomains = 33;
    let new_domains: Vec<_> = derived_domain
        .create_subdomains(n_subdomains.try_into().unwrap())
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(new_domains.len(), n_subdomains);
}
