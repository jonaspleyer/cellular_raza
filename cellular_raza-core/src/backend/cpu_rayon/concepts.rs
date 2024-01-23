use std::collections::BTreeSet;

use cellular_raza_concepts::{errors::*, prelude::CycleEvent};

use serde::{Deserialize, Serialize};

pub trait Domain<P> {
    type SubDomain;

    fn determine_subdomain(&self, position: &P) -> Option<usize>;

    // These subdomains should also be responsible to integrate extracellular mechanics and so on.
    // This is already partly realized by MultivoxelContainers in the domain_decomposition module of the cpu_os_threads backend.
    fn into_subdomains(self, n: usize) -> Result<(usize, Vec<Self::SubDomain>), CalcError>;
}

pub trait SubDomain<P> {
    type Index;

    fn contains_index(&self, index: &Self::Index) -> bool;
    fn voxel_indices(&self) -> Vec<Self::Index>;
    fn get_index(&self, cell: &P) -> Option<Self::Index>;
    fn get_neighbor_subdomain_indices(&self) -> Vec<usize>;
    fn apply_boundary(&self, position: &mut P) -> Result<(), BoundaryError>;
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Rectangle2Domain {
    min: [f64; 2],
    max: [f64; 2],
    n_voxels: [usize; 2],
}

impl Rectangle2Domain {
    pub fn from_interaction_range(min: [f64; 2], max: [f64; 2], range: [f64; 2]) -> Self {
        let n_voxels = [
            ((max[0] - min[0]) / range[0]).floor() as usize,
            ((max[1] - min[1]) / range[1]).floor() as usize,
        ];
        Self { min, max, n_voxels }
    }
}

impl Domain<[f64; 2]> for Rectangle2Domain {
    type SubDomain = Rectangle2SubDomain;

    fn determine_subdomain(&self, position: &[f64; 2]) -> Option<usize> {
        let subdomain_indices = [
            (((self.max[0] - position[0].clamp(self.min[0], self.max[0])) % self.n_voxels[0] as f64)
                as usize),
            (((self.max[0] - position[0].clamp(self.min[1], self.max[1])) % self.n_voxels[1] as f64)
                as usize),
        ];
        Some(0)
    }

    fn into_subdomains(self, n: usize) -> Result<(usize, Vec<Self::SubDomain>), CalcError> {
        // Determine if n was a square
        let n_root = (n as f64).sqrt();
        let nearest_whole_number = n_root.round() as i64;
        let missing_to_square = n as i64 - nearest_whole_number.pow(2);
        println!("{} {} {}", n_root, nearest_whole_number, missing_to_square);

        // Now we create prev_whole_number + next_whole_number
        let dx = (self.max[0] - self.min[0]) / n as f64;
        let subdomains = (0..n)
            .map(|i| Rectangle2SubDomain {
                // TODO this is totally incorrect!
                min: [self.min[0] + (i as f64) / (n as f64) * dx, self.min[1]],
                max: self.max,
                voxels: [].into(),
            })
            .collect();

        Ok((n.min(self.n_voxels.iter().sum()), subdomains))
    }
}

pub struct Rectangle2SubDomain {
    min: [f64; 2],
    max: [f64; 2],
    voxels: BTreeSet<[usize; 2]>,
}

impl SubDomain<[f64; 2]> for Rectangle2SubDomain {
    type Index = [usize; 2];

    fn contains_index(&self, index: &Self::Index) -> bool {
        self.voxels.contains(index)
    }

    fn get_index(&self, position: &[f64; 2]) -> Option<Self::Index> {
        todo!()
    }

    fn get_neighbor_subdomain_indices(&self) -> Vec<usize> {
        todo!()
    }

    fn voxel_indices(&self) -> Vec<Self::Index> {
        todo!()
    }

    fn apply_boundary(&self, position: &mut [f64; 2]) -> Result<(), BoundaryError> {
        for i in 0..2 {
            if position[i] < self.min[i] {
                position[i] = 2.0 * self.min[i] - position[i];
            } else if position[i] > self.max[i] {
                position[i] = 2.0 * self.max[i] - position[i];
            }
        }
        Ok(())
    }
}

macro_rules! run_full_simulation(
    ($cont:expr, Mechanics) => {println!("Mechanics");};
    ($cont:expr, Interaction) => {println!("Interaction");};
    ($cont:expr, Cycle) => {
        $cont.update_cell_cycles()
    };

    ($cont:expr, $s:ident, $($r:ident),+) => {
        run_full_simulation!($cont, $s);
        run_full_simulation!($cont, $($r),+);
    };
);

struct Container {}

impl Container {
    pub fn update_cell_cycles(&self)
    /* where
    Cell: cellular_raza_concepts::Cycle<Cell>,*/
    {
        println!("Update done");
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_domain_hirarchy() {
        let domain = Rectangle2Domain::from_interaction_range([0.0; 2], [100.0; 2], [10.0; 2]);

        let (n, subdomains) = domain.into_subdomains(8).unwrap();
        assert_eq!(subdomains.len(), 8);
        assert_eq!(n, 8);

        // let position = domain.determine_subdomain(&[0.0; 2]);
        // println!("{:?}", position);
    }
}
