// Imports from this crate
use cellular_raza_concepts::*;

// Imports from other crates
use itertools::Itertools;
use nalgebra::SVector;

use serde::{Deserialize, Deserializer, Serialize, ser::SerializeStruct};

/// Helper function to calculate the decomposition of a large number N into n as evenly-sizedchunks
/// chunks as possible
/// Examples:
/// N   n   decomp
/// 10  3    1 *  4  +  3 *  3
/// 13  4    1 *  5  +  3 *  4
/// 100 13   4 * 13  +  4 * 12
/// 225 16   1 * 15  + 15 * 14
/// 225 17   4 * 14  + 13 * 13
pub(super) fn get_decomp_res(n_voxel: usize, n_regions: usize) -> Option<(usize, usize, usize)> {
    // We calculate how many times we need to drain how many voxels
    // Example:
    //      n_voxels    = 59
    //      n_regions   = 6
    //      average_len = (59 / 8).ceil() = (9.833 ...).ceil() = 10
    //
    // try to solve this equation:
    //      n_voxels = average_len * n + (average_len-1) * m
    //      where n,m are whole positive numbers
    //
    // We start with    n = n_regions = 6
    // and with         m = min(0, n_voxel - average_len.pow(2)) = min(0, 59 - 6^2) = 23
    let mut average_len: i64 = (n_voxel as f64 / n_regions as f64).ceil() as i64;

    let residue = |n: i64, m: i64, avg: i64| n_voxel as i64 - avg * n - (avg - 1) * m;

    let mut n = n_regions as i64;
    let mut m = 0;

    for _ in 0..n_regions {
        let r = residue(n, m, average_len);
        if r == 0 {
            return Some((n as usize, m as usize, average_len as usize));
        } else if r > 0 {
            if n == n_regions as i64 {
                // Start from the beginning again but with different value for average length
                average_len += 1;
                n = n_regions as i64;
                m = 0;
            } else {
                n += 1;
                m -= 1;
            }
        // Residue is negative. This means we have subtracted too much and we just decrease n and
        // increase m
        } else {
            n -= 1;
            m += 1;
        }
    }
    None
}

/// A generic Domain with a cuboid layout.
///
/// This struct can be used to define custom domains on top of its behaviour.
#[derive(Clone, Debug)]
pub struct CartesianCuboid<F, const D: usize> {
    min: SVector<F, D>,
    max: SVector<F, D>,
    dx: SVector<F, D>,
    n_voxels: SVector<usize, D>,
    /// Seed from which all random numbers will be initially drawn
    pub rng_seed: u64,
}

impl<F, const D: usize> CartesianCuboid<F, D>
where
    F: Clone,
{
    /// Get the minimum point which defines the simulation domain
    pub fn get_min(&self) -> SVector<F, D> {
        self.min.clone()
    }

    /// Get the maximum point which defines the simulation domain
    pub fn get_max(&self) -> SVector<F, D> {
        self.max.clone()
    }

    /// Get the discretization used to generate voxels
    pub fn get_dx(&self) -> SVector<F, D> {
        self.dx.clone()
    }

    /// Get the number of voxels in each dimension of the domain
    pub fn get_n_voxels(&self) -> SVector<usize, D> {
        self.n_voxels
    }
}

impl<C, Ci, F, const D: usize> Domain<C, CartesianSubDomain<F, D>, Ci> for CartesianCuboid<F, D>
where
    C: Position<nalgebra::SVector<F, D>>,
    F: 'static
        + num::Float
        + Copy
        + core::fmt::Debug
        + num::FromPrimitive
        + num::ToPrimitive
        + core::ops::SubAssign
        + core::ops::Div<Output = F>
        + core::ops::DivAssign,
    Ci: IntoIterator<Item = C>,
{
    type SubDomainIndex = usize;
    type VoxelIndex = [usize; D];

    fn decompose(
        self,
        n_subdomains: core::num::NonZeroUsize,
        cells: Ci,
    ) -> Result<DecomposedDomain<Self::SubDomainIndex, CartesianSubDomain<F, D>, C>, DecomposeError>
    {
        #[derive(Clone, Domain)]
        struct MyIntermdiatedomain<F, const D: usize>
        where
            F: 'static
                + num::Float
                + Copy
                + core::fmt::Debug
                + num::FromPrimitive
                + num::ToPrimitive
                + core::ops::SubAssign
                + core::ops::Div<Output = F>
                + core::ops::DivAssign,
        {
            #[DomainRngSeed]
            #[DomainCreateSubDomains]
            #[SortCells]
            domain: CartesianCuboid<F, D>,
        }
        let my_intermediate_domain = MyIntermdiatedomain { domain: self };
        my_intermediate_domain.decompose(n_subdomains, cells)
    }
}

impl<F, const D: usize> CartesianCuboid<F, D>
where
    F: 'static + num::Float + Copy + core::fmt::Debug + num::FromPrimitive + num::ToPrimitive,
{
    fn check_min_max(min: &[F; D], max: &[F; D]) -> Result<(), BoundaryError>
    where
        F: core::fmt::Debug,
    {
        for i in 0..D {
            if min[i] >= max[i] {
                return Err(BoundaryError(format!(
                    "Min {:?} must be smaller than Max {:?} for domain boundaries!",
                    min, max
                )));
            }
        }
        Ok(())
    }

    /// Builds a new [CartesianCuboid] from given boundaries and maximum interaction ranges of the
    /// containing cells.
    ///
    /// ```
    /// # use cellular_raza_building_blocks::CartesianCuboid;
    /// let min = [2.0, 3.0, 1.0];
    /// let max = [10.0, 10.0, 20.0];
    /// let interaction_range = 2.0;
    /// let domain = CartesianCuboid::from_boundaries_and_interaction_range(
    ///     min,
    ///     max,
    ///     interaction_range
    /// )?;
    ///
    /// assert_eq!(domain.get_n_voxels()[0], 4);
    /// assert_eq!(domain.get_n_voxels()[1], 3);
    /// assert_eq!(domain.get_n_voxels()[2], 9);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_boundaries_and_interaction_range(
        min: impl Into<[F; D]>,
        max: impl Into<[F; D]>,
        interaction_range: F,
    ) -> Result<Self, BoundaryError> {
        // Perform conversions
        let min: [F; D] = min.into();
        let max: [F; D] = max.into();

        // Check that the specified min and max are actually smaller / larger
        Self::check_min_max(&min, &max)?;

        // Calculate the number of voxels from given interaction ranges
        let mut n_voxels = [0; D];
        let mut dx = [F::zero(); D];
        for i in 0..D {
            let n = ((max[i] - min[i]) / interaction_range).floor();
            // This conversion should hopefully never fail.
            n_voxels[i] = n.to_usize().ok_or(BoundaryError(
                cellular_raza_concepts::format_error_message!(
                    format!(
                        "Cannot convert float {:?} of type {} to usize",
                        n,
                        std::any::type_name::<F>()
                    ),
                    "conversion error during domain setup"
                ),
            ))?;
            dx[i] = (max[i] - min[i]) / n;
        }

        Ok(Self {
            min: min.into(),
            max: max.into(),
            dx: dx.into(),
            n_voxels: n_voxels.into(),
            rng_seed: 0,
        })
    }

    /// Builds a new [CartesianCuboid] from given boundaries and the number of voxels per dimension
    /// specified.
    pub fn from_boundaries_and_n_voxels(
        min: impl Into<[F; D]>,
        max: impl Into<[F; D]>,
        n_voxels: impl Into<[usize; D]>,
    ) -> Result<Self, BoundaryError> {
        let min: [F; D] = min.into();
        let max: [F; D] = max.into();
        let n_voxels: [usize; D] = n_voxels.into();
        Self::check_min_max(&min, &max)?;
        let mut dx: SVector<F, D> = [F::zero(); D].into();
        for i in 0..D {
            let n = F::from_usize(n_voxels[i]).ok_or(BoundaryError(
                cellular_raza_concepts::format_error_message!(
                    "conversion error during domain setup",
                    format!(
                        "Cannot convert usize {} to float of type {}",
                        n_voxels[i],
                        std::any::type_name::<F>()
                    )
                ),
            ))?;
            dx[i] = (max[i] - min[i]) / n;
        }
        Ok(Self {
            min: min.into(),
            max: max.into(),
            dx,
            n_voxels: n_voxels.into(),
            rng_seed: 0,
        })
    }
}

impl<F, const D: usize> CartesianCuboid<F, D> {
    fn get_all_voxel_indices(&self) -> impl IntoIterator<Item = [usize; D]> {
        use itertools::*;
        (0..D)
            .map(|i| 0..self.n_voxels[i])
            .multi_cartesian_product()
            .map(|x| {
                let mut index = [0; D];
                index.copy_from_slice(&x);
                index
            })
    }

    /// Get the total amount of indices in this domain
    fn get_n_indices(&self) -> usize {
        let mut res = 1;
        for i in 0..D {
            res *= self.n_voxels[i];
        }
        res
    }
}

mod test_domain_setup {
    #[test]
    fn from_boundaries_and_interaction_range() {
        use crate::CartesianCuboid;
        let min = [0.0; 2];
        let max = [2.0; 2];
        let interaction_range = 1.0;
        let _ = CartesianCuboid::from_boundaries_and_interaction_range(min, max, interaction_range)
            .unwrap();
        // TODO add actual test case here
    }

    #[test]
    fn from_boundaries_and_n_voxels() {
        use crate::CartesianCuboid;
        let min = [-100.0f32; 55];
        let max = [43000.0f32; 55];
        let n_voxels = [22; 55];
        let _ = CartesianCuboid::from_boundaries_and_n_voxels(min, max, n_voxels).unwrap();
        // TODO add actual test case here
    }
}

impl<F, const D: usize> CartesianCuboid<F, D>
where
    F: 'static
        + num::Float
        + Copy
        + core::fmt::Debug
        + num::FromPrimitive
        + num::ToPrimitive
        + core::ops::SubAssign
        + core::ops::Div<Output = F>
        + core::ops::DivAssign,
{
    /// Obtains the voxel index given a regular vector
    ///
    /// This function can be used in derivatives of this type.
    pub fn get_voxel_index_of_raw(&self, pos: &SVector<F, D>) -> Result<[usize; D], BoundaryError> {
        Self::check_min_max(&self.min.into(), &(*pos).into())?;
        let n_vox = (pos - self.min).component_div(&self.dx);
        let mut res = [0usize; D];
        for i in 0..D {
            res[i] = n_vox[i].to_usize().ok_or(BoundaryError(
                cellular_raza_concepts::format_error_message!(
                    "conversion error during domain setup",
                    format!(
                        "Cannot convert float {:?} of type {} to usize",
                        n_vox[i],
                        std::any::type_name::<F>()
                    )
                ),
            ))?;
        }
        Ok(res)
    }
}

impl<C, F, const D: usize> SortCells<C> for CartesianCuboid<F, D>
where
    F: 'static
        + num::Float
        + Copy
        + core::fmt::Debug
        + num::FromPrimitive
        + num::ToPrimitive
        + core::ops::SubAssign
        + core::ops::Div<Output = F>
        + core::ops::DivAssign,
    C: Position<SVector<F, D>>,
{
    type VoxelIndex = [usize; D];

    fn get_voxel_index_of(&self, cell: &C) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.pos();
        self.get_voxel_index_of_raw(&pos)
    }
}

impl<C, F, const D: usize> SortCells<C> for CartesianSubDomain<F, D>
where
    C: Position<nalgebra::SVector<F, D>>,
    F: 'static + num::Float + core::fmt::Debug + core::ops::SubAssign + core::ops::DivAssign,
{
    type VoxelIndex = [usize; D];

    fn get_voxel_index_of(&self, cell: &C) -> Result<Self::VoxelIndex, BoundaryError> {
        let pos = cell.pos();
        self.get_index_of(pos)
    }
}

impl<F, const D: usize> DomainRngSeed for CartesianCuboid<F, D> {
    fn get_rng_seed(&self) -> u64 {
        self.rng_seed
    }
}

#[test]
fn generate_subdomains() {
    use DomainCreateSubDomains;
    let min = [0.0; 3];
    let max = [100.0; 3];
    let interaction_range = 20.0;
    let domain =
        CartesianCuboid::from_boundaries_and_interaction_range(min, max, interaction_range)
            .unwrap();
    let sub_domains = domain
        .create_subdomains(4.try_into().unwrap())
        .unwrap()
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(sub_domains.len(), 4);
    assert_eq!(
        sub_domains
            .iter()
            .map(|(_, _, voxels)| voxels.len())
            .sum::<usize>(),
        5usize.pow(3)
    );
}

/// Subdomain corresponding to the [CartesianCuboid] struct.
#[derive(Clone, Debug, PartialEq)]
pub struct CartesianSubDomain<F, const D: usize> {
    min: SVector<F, D>,
    max: SVector<F, D>,
    dx: SVector<F, D>,
    voxels: Vec<[usize; D]>,
    pub(crate) domain_min: SVector<F, D>,
    pub(crate) domain_max: SVector<F, D>,
    domain_n_voxels: SVector<usize, D>,
}

#[derive(Deserialize)]
#[serde(rename(serialize = "CartesianSubDomain", deserialize = "CartesianSubDomain",))]
struct __CartesianSubDomainSerde<F: 'static + Clone + core::fmt::Debug + PartialEq, const D: usize>
{
    min: SVector<F, D>,
    max: SVector<F, D>,
    dx: SVector<F, D>,
    voxels: Vec<SVector<usize, D>>,
    domain_min: SVector<F, D>,
    domain_max: SVector<F, D>,
    domain_n_voxels: SVector<usize, D>,
}

impl<F, const D: usize> From<__CartesianSubDomainSerde<F, D>> for CartesianSubDomain<F, D>
where
    F: 'static + Clone + core::fmt::Debug + PartialEq,
{
    fn from(s: __CartesianSubDomainSerde<F, D>) -> Self {
        CartesianSubDomain {
            min: s.min,
            max: s.max,
            dx: s.dx,
            voxels: s.voxels.into_iter().map(<[usize; D]>::from).collect(),
            domain_min: s.domain_min,
            domain_max: s.domain_max,
            domain_n_voxels: s.domain_n_voxels,
        }
    }
}

impl<F, const D: usize> Serialize for CartesianSubDomain<F, D>
where
    F: nalgebra::Scalar + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("CartesianSubDomain", 7)?;
        state.serialize_field("min", &self.min)?;
        state.serialize_field("max", &self.max)?;
        state.serialize_field("dx", &self.dx)?;
        let voxels = self
            .voxels
            .iter()
            .map(|ind| ind.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        state.serialize_field("voxels", &voxels)?;
        state.serialize_field("domain_min", &self.domain_min)?;
        state.serialize_field("domain_max", &self.domain_max)?;
        state.serialize_field("domain_n_voxels", &self.domain_n_voxels)?;
        state.end()
    }
}

impl<'de, F, const D: usize> Deserialize<'de> for CartesianSubDomain<F, D>
where
    F: nalgebra::Scalar + for<'a> Deserialize<'a>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let s = __CartesianSubDomainSerde::deserialize(deserializer)?;
        let subdomain = s.into();
        Ok(subdomain)
    }
}

#[test]
fn serialize_cartesiansubdomain() {
    let subdomain = CartesianSubDomain {
        min: [-30.0, 10.0].into(),
        max: [55.33, 11.0].into(),
        dx: [1.0, 0.01].into(),
        voxels: vec![[1, 2], [3, 4], [5, 6]],
        domain_min: [-30.0, 10.0].into(),
        domain_max: [55.33, 22.38].into(),
        domain_n_voxels: [1, 2].into(),
    };
    // TODO finish this test
    use serde_test::{Token, assert_de_tokens, assert_ser_tokens};
    let tokens = [
        Token::Struct {
            name: "CartesianSubDomain",
            len: 7,
        },
        // subdomain.min
        Token::Str("min"),
        Token::Tuple { len: 2 },
        Token::F64(subdomain.min[0]),
        Token::F64(subdomain.min[1]),
        Token::TupleEnd,
        // subdomain.max
        Token::Str("max"),
        Token::Tuple { len: 2 },
        Token::F64(subdomain.max[0]),
        Token::F64(subdomain.max[1]),
        Token::TupleEnd,
        // subdomain.dx
        Token::Str("dx"),
        Token::Tuple { len: 2 },
        Token::F64(subdomain.dx[0]),
        Token::F64(subdomain.dx[1]),
        Token::TupleEnd,
        // subdomain.voxels
        Token::Str("voxels"),
        Token::Seq { len: Some(3) },
        Token::Seq { len: Some(2) },
        Token::U64(subdomain.voxels[0][0] as u64),
        Token::U64(subdomain.voxels[0][1] as u64),
        Token::SeqEnd,
        Token::Seq { len: Some(2) },
        Token::U64(subdomain.voxels[1][0] as u64),
        Token::U64(subdomain.voxels[1][1] as u64),
        Token::SeqEnd,
        Token::Seq { len: Some(2) },
        Token::U64(subdomain.voxels[2][0] as u64),
        Token::U64(subdomain.voxels[2][1] as u64),
        Token::SeqEnd,
        Token::SeqEnd,
        // domain.min
        Token::Str("domain_min"),
        Token::Tuple { len: 2 },
        Token::F64(subdomain.domain_min[0]),
        Token::F64(subdomain.domain_min[1]),
        Token::TupleEnd,
        // domain.max
        Token::Str("domain_max"),
        Token::Tuple { len: 2 },
        Token::F64(subdomain.domain_max[0]),
        Token::F64(subdomain.domain_max[1]),
        Token::TupleEnd,
        // domain.dx
        Token::Str("domain_n_voxels"),
        Token::Tuple { len: 2 },
        Token::U64(subdomain.domain_n_voxels[0] as u64),
        Token::U64(subdomain.domain_n_voxels[1] as u64),
        Token::TupleEnd,
        Token::StructEnd,
    ];
    assert_ser_tokens(&subdomain, &tokens);
    assert_de_tokens(&subdomain, &tokens);
}

impl<F, const D: usize> CartesianSubDomain<F, D>
where
    F: Clone,
{
    /// Get the minimum boundary of the subdomain.
    /// Note that not all voxels which could be in the space of the subdomain need to be in it.
    pub fn get_min(&self) -> SVector<F, D> {
        self.min.clone()
    }

    /// Get the maximum boundary of the subdomain.
    /// Note that not all voxels which could be in the space of the subdomain need to be in it.
    pub fn get_max(&self) -> SVector<F, D> {
        self.max.clone()
    }

    /// Get the discretization used to generate voxels
    pub fn get_dx(&self) -> SVector<F, D> {
        self.dx.clone()
    }

    /// Get all voxel indices which are currently in this subdomain
    pub fn get_voxels(&self) -> Vec<[usize; D]> {
        self.voxels.clone()
    }

    /// See [CartesianCuboid::get_min].
    pub fn get_domain_min(&self) -> SVector<F, D> {
        self.domain_min.clone()
    }

    /// See [CartesianCuboid::get_max].
    pub fn get_domain_max(&self) -> SVector<F, D> {
        self.domain_max.clone()
    }

    /// See [CartesianCuboid::get_n_voxels].
    pub fn get_domain_n_voxels(&self) -> SVector<usize, D> {
        self.domain_n_voxels
    }
}

impl<F, const D: usize> CartesianSubDomain<F, D> {
    /// Generic method to obtain the voxel index of any type that can be casted to an array.
    pub fn get_index_of<P>(&self, pos: P) -> Result<[usize; D], BoundaryError>
    where
        [F; D]: From<P>,
        F: 'static + num::Float + core::fmt::Debug + core::ops::SubAssign + core::ops::DivAssign,
    {
        let pos: [F; D] = pos.into();
        let mut res = [0usize; D];
        for i in 0..D {
            let n_vox = (pos[i] - self.domain_min[i]) / self.dx[i];
            res[i] = n_vox.to_usize().ok_or(BoundaryError(
                cellular_raza_concepts::format_error_message!(
                    "conversion error during domain setup",
                    format!(
                        "Cannot convert float {:?} of type {} to usize",
                        n_vox,
                        std::any::type_name::<F>()
                    )
                ),
            ))?;
        }
        Ok(res)
    }
}

impl<F, const D: usize> DomainCreateSubDomains<CartesianSubDomain<F, D>> for CartesianCuboid<F, D>
where
    F: 'static + num::Float + core::fmt::Debug + num::FromPrimitive,
{
    type SubDomainIndex = usize;
    type VoxelIndex = [usize; D];

    fn create_subdomains(
        &self,
        n_subdomains: core::num::NonZeroUsize,
    ) -> Result<
        impl IntoIterator<
            Item = (
                Self::SubDomainIndex,
                CartesianSubDomain<F, D>,
                Vec<Self::VoxelIndex>,
            ),
        >,
        DecomposeError,
    > {
        let indices = self.get_all_voxel_indices();
        let n_indices = self.get_n_indices();

        let (n, _m, average_len) = get_decomp_res(n_indices, n_subdomains.into()).ok_or(
            DecomposeError::Generic("Could not find a suiting decomposition".to_owned()),
        )?;

        // TODO Currently we are not splitting the voxels apart efficiently
        // These are subdomains which contain n voxels
        let switcher = n * average_len;
        let indices_grouped = indices.into_iter().enumerate().chunk_by(|(i, _)| {
            use num::Integer;
            if *i < switcher {
                i.div_rem(&average_len).0
            } else {
                (i - switcher).div_rem(&(average_len - 1).max(1)).0 + n
            }
        });
        let mut res = Vec::new();
        for (n_subdomain, indices) in indices_grouped.into_iter() {
            let mut min_vox = [usize::MAX; D];
            let mut max_vox = [0; D];
            let voxels = indices
                .into_iter()
                .map(|(_, index)| {
                    for i in 0..D {
                        min_vox[i] = min_vox[i].min(index[i]);
                        max_vox[i] = max_vox[i].max(index[i]);
                    }
                    index
                })
                .collect::<Vec<_>>();
            let mut min = [F::zero(); D];
            let mut max = [F::zero(); D];
            for i in 0..D {
                let n_vox_min = F::from_usize(min_vox[i]).ok_or(DecomposeError::Generic(
                    cellular_raza_concepts::format_error_message!(
                        "conversion error during domain setup",
                        format!(
                            "Cannot convert float {:?} of type {} to usize",
                            min_vox[i],
                            std::any::type_name::<F>()
                        )
                    ),
                ))?;
                let n_vox_max = F::from_usize(max_vox[i]).ok_or(DecomposeError::Generic(
                    cellular_raza_concepts::format_error_message!(
                        "conversion error during domain setup",
                        format!(
                            "Cannot convert float {:?} of type {} to usize",
                            max_vox[i],
                            std::any::type_name::<F>()
                        )
                    ),
                ))?;
                min[i] = self.min[i] + n_vox_min * self.dx[i];
                max[i] = self.min[i] + (n_vox_max + F::one()) * self.dx[i];
            }
            let subdomain = CartesianSubDomain {
                min: min.into(),
                max: max.into(),
                dx: self.dx,
                voxels: voxels.clone(),
                domain_min: self.min,
                domain_max: self.max,
                domain_n_voxels: self.n_voxels,
            };
            res.push((n_subdomain, subdomain, voxels));
        }
        Ok(res)
    }
}

impl<Coord, F, const D: usize> SubDomainMechanics<Coord, Coord> for CartesianSubDomain<F, D>
where
    Coord: Clone,
    [F; D]: From<Coord>,
    Coord: From<[F; D]>,
    Coord: std::fmt::Debug,
    F: num::Float,
{
    fn apply_boundary(&self, pos: &mut Coord, vel: &mut Coord) -> Result<(), BoundaryError> {
        let mut velocity: [F; D] = vel.clone().into();
        let mut position: [F; D] = pos.clone().into();

        // Define constant two
        let two = F::one() + F::one();

        // For each dimension
        for i in 0..D {
            // Check if the particle is below lower edge
            if position[i] < self.domain_min[i] {
                position[i] = two * self.domain_min[i] - position[i];
                velocity[i] = velocity[i].abs();
            }
            // Check if the particle is over the edge
            if position[i] > self.domain_max[i] {
                position[i] = two * self.domain_max[i] - position[i];
                velocity[i] = -velocity[i].abs();
            }
        }

        // If new position is still out of boundary return error
        for i in 0..D {
            if position[i] < self.domain_min[i] || position[i] > self.domain_max[i] {
                return Err(BoundaryError(format!(
                    "Particle is out of domain at position {:?}",
                    pos
                )));
            }
        }

        // Set the position and velocity
        *pos = position.into();
        *vel = velocity.into();
        Ok(())
    }
}

impl<F, const D: usize> SubDomain for CartesianSubDomain<F, D> {
    type VoxelIndex = [usize; D];

    fn get_all_indices(&self) -> Vec<Self::VoxelIndex> {
        self.voxels.clone()
    }

    fn get_neighbor_voxel_indices(&self, voxel_index: &Self::VoxelIndex) -> Vec<Self::VoxelIndex> {
        // Create the bounds for the following creation of all the voxel indices
        let mut bounds = [[0; 2]; D];
        for i in 0..D {
            bounds[i] = [
                (voxel_index[i] as i64 - 1).max(0) as usize,
                (voxel_index[i] + 2).min(self.domain_n_voxels[i]),
            ];
        }

        // Create voxel indices
        (0..D)
            .map(|i| (bounds[i][0]..bounds[i][1]))
            .multi_cartesian_product()
            .map(|ind_v| {
                let mut res = [0; D];
                for i in 0..D {
                    res[i] = ind_v[i];
                }
                res
            })
            .filter(|ind| ind != voxel_index)
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::get_decomp_res;
    use rayon::prelude::*;

    #[test]
    fn test_get_demomp_res() {
        #[cfg(debug_assertions)]
        let max = 500;
        #[cfg(not(debug_assertions))]
        let max = 5_000;

        (1..max)
            .into_par_iter()
            .map(|n_voxel| {
                #[cfg(debug_assertions)]
                let max_regions = 100;
                #[cfg(not(debug_assertions))]
                let max_regions = 1_000;
                for n_regions in 1..max_regions {
                    match get_decomp_res(n_voxel, n_regions) {
                        Some(res) => {
                            let (n, m, average_len) = res;
                            assert_eq!(n + m, n_regions);
                            assert_eq!(n * average_len + m * (average_len - 1), n_voxel);
                        }
                        None => panic!(
                            "No result for inputs n_voxel: {} n_regions: {}",
                            n_voxel, n_regions
                        ),
                    }
                }
            })
            .collect::<Vec<()>>();
    }
}
