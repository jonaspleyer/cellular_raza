type ReactionVector<F> = nalgebra::DVector<F>;

use cellular_raza_concepts::*;
use serde::{Deserialize, Serialize};

use super::{CartesianCuboid, CartesianSubDomain};

/// Domain based on [CartesianCuboid] in 2 dimensions with extracellular diffusion.
///
/// We solve the equations
///
#[derive(Domain, Clone)]
pub struct CartesianDiffusion2D<F> {
    /// See [CartesianCuboid]
    #[DomainRngSeed]
    #[DomainPartialDerive]
    #[SortCells]
    pub domain: CartesianCuboid<F, 2>,
    /// The discretization must be a multiple of the voxel size.
    /// This quantity will be used as an initial estimate and rounded to the nearest candidate.
    pub reactions_dx: nalgebra::Vector2<F>,
    /// Diffusion constant
    pub diffusion_constant: F,
    /// Initial values which are uniform over the simulation domain
    pub initial_value: ReactionVector<F>,
}

/// Subdomain for the [CartesianDiffusion2D] domain.
#[derive(SubDomain, Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F:
    'static
    + PartialEq
    + Clone
    + core::fmt::Debug
    + Serialize
    + for<'a> Deserialize<'a>")]
pub struct CartesianDiffusion2DSubDomain<F> {
    /// See [CartesianSubDomain]
    #[Base]
    #[SortCells]
    #[Mechanics]
    subdomain: CartesianSubDomain<F, 2>,
    reactions_min: nalgebra::Vector2<F>,
    // reactions_max: nalgebra::Vector2<F>,
    reactions_dx: nalgebra::Vector2<F>,
    index_min: nalgebra::Vector2<usize>,
    index_max: nalgebra::Vector2<usize>,
    extracellular: ndarray::Array3<F>,
    ownership_array: ndarray::Array2<bool>,
    /// Diffusion constant
    pub diffusion_constant: F,
    increments: [ndarray::Array3<F>; 3],
    increments_start: usize,
    helper: ndarray::Array3<F>,
}

/// This type is not intendend for direct usage
#[doc(hidden)]
pub struct BorderInfo {
    min_sent: nalgebra::Vector2<usize>,
    max_sent: nalgebra::Vector2<usize>,
}

impl<F> CartesianDiffusion2DSubDomain<F>
where
    F: nalgebra::RealField + Clone,
{
    fn assign_neighbor(&mut self, neighbor: NeighborValue<F>) {
        use ndarray::*;
        let NeighborValue { min, max, values } = neighbor;

        // Cast everything to isize to avoid overflows
        let min = min.cast::<isize>();
        let max = max.cast::<isize>();
        let index_min = self.index_min.cast::<isize>();
        let index_max = self.index_max.cast::<isize>();

        // Legend
        // o = own array
        // x = neighbor value
        //
        // x x x x x x . . . .
        // x x x x x x . . . .
        // x x x o o o o o o o -- shared_min[1]
        // x x x o o o o o o o
        // x x x o o o o o o o -- shared_max[1]
        // . . . o o o o o o o
        // . . . o o o o o o o
        //       |   |
        //       |   shared_max[0]
        //       |
        //       shared_min[0]
        let shared_min = min.sup(&index_min);
        let shared_max = max.inf(&index_max);

        // x x x x x x . . . .                   o o o o o o . . . .
        // x x h h h h . . . . -- helper_min[1]  o o o o o o . . . .
        // x x h o o o o o o o                   o o o o o o h x x x -- helper_min[1]
        // x x h o o o o o o o                   o o o o o o h x x x
        // x x h o o o o o o o -- helper_max[1]  o o o o o o h x x x
        // . . . o o o o o o o                   . . . h h h h x x x -- helper_max[1]
        // . . . o o o o o o o                   . . . x x x x x x x
        //     |     |                                 |     |
        //     |     helper_max[0]                     |     helper_max[0]
        //     |                                       |
        //     heper_min[0]                            helper_min[0]
        let helper_min = shared_min.add_scalar(-1).sup(&min);
        let helper_max = shared_max.add_scalar(1).inf(&max);

        let nmin = helper_min - min;
        let nmax = helper_max - min;
        let hmin = (helper_min - index_min).add_scalar(1);
        // let hmax = (helper_max - index_min).inf(&shared_max).add_scalar(1);
        let hmax = nmax - nmin + hmin;

        Zip::from(
            self.helper
                .slice_mut(s![hmin[0]..hmax[0], hmin[1]..hmax[1], ..])
                .lanes_mut(Axis(2)),
        )
        .and(values.slice(s![nmin[0]..nmax[0], nmin[1]..nmax[1]]))
        .and(
            self.ownership_array
                .slice(s![hmin[0]..hmax[0], hmin[1]..hmax[1]]),
        )
        .for_each(|mut w, v, t| {
            if let (false, Some(vi)) = (*t, v) {
                w.assign(vi);
            }
        });
    }
}

/// This type is not intendend for direct usage
#[doc(hidden)]
pub struct NeighborValue<F> {
    min: nalgebra::Vector2<usize>,
    max: nalgebra::Vector2<usize>,
    values: ndarray::Array2<Option<ndarray::Array1<F>>>,
}

impl<F> SubDomainReactions<nalgebra::SVector<F, 2>, ReactionVector<F>, F>
    for CartesianDiffusion2DSubDomain<F>
where
    F: nalgebra::RealField + Copy + num::traits::AsPrimitive<usize> + ndarray::ScalarOperand,
    usize: num::traits::AsPrimitive<F>,
{
    type BorderInfo = BorderInfo;
    type NeighborValue = NeighborValue<F>;

    fn treat_increments<I, J>(&mut self, neighbors: I, sources: J) -> Result<(), CalcError>
    where
        I: IntoIterator<Item = Self::NeighborValue>,
        J: IntoIterator<Item = (nalgebra::SVector<F, 2>, ReactionVector<F>)>,
    {
        use core::ops::AddAssign;
        use ndarray::*;
        let two = F::one() + F::one();
        let dx2 = self.reactions_dx[0].powf(-two);
        let dy2 = self.reactions_dx[1].powf(-two);
        let dd2 = -two * (dx2 + dy2);

        // Helper variable to store current concentrations
        let co = &self.extracellular;

        // Use helper array which is +2 in every spatial dimension larger than the original array
        // We do this to seamlessly incorporate boundary conditions
        // Fill inner part of the array
        self.helper.fill(F::zero());
        // _ _ _ _ _ _ _
        // _ x x x x x _
        // _ x x x x x _
        // _ _ _ _ _ _ _
        self.helper.slice_mut(s![1..-1, 1..-1, ..]).assign(co);

        // First assume that we obey neumann-boundary conditions and fill outer parts accordingly
        // _ x x x x x _
        // x _ _ _ _ _ x
        // x _ _ _ _ _ x
        // _ x x x x x _
        self.helper
            .slice_mut(s![0, 1..-1, ..])
            .assign(&co.slice(s![0, .., ..]));
        self.helper
            .slice_mut(s![-1, 1..-1, ..])
            .assign(&co.slice(s![-1, .., ..]));
        self.helper
            .slice_mut(s![1..-1, 0, ..])
            .assign(&co.slice(s![.., 0, ..]));
        self.helper
            .slice_mut(s![1..-1, -1, ..])
            .assign(&co.slice(s![.., -1, ..]));

        for neighbor in neighbors {
            self.assign_neighbor(neighbor);
        }

        // Set increment to next time-step to F::zero() everywhere
        let start = self.increments_start;
        self.increments[start].fill(F::zero());

        let dc = self.diffusion_constant;
        // - 2u[i,j] /dx^2 - 2u[i,j]/dy^2
        self.increments[start].add_assign(&(&self.helper.slice(s![1..-1, 1..-1, ..]) * dd2 * dc));
        // + u[i-1,j]/dx^2
        self.increments[start].add_assign(&(&self.helper.slice(s![..-2, 1..-1, ..]) * dx2 * dc));
        // + u[i+1,j]/dx^2
        self.increments[start].add_assign(&(&self.helper.slice(s![2.., 1..-1, ..]) * dx2 * dc));
        // + u[i,j-1]/dy^2
        self.increments[start].add_assign(&(&self.helper.slice(s![1..-1, ..-2, ..]) * dy2 * dc));
        // + u[i,j+1]/dy^2
        self.increments[start].add_assign(&(&self.helper.slice(s![1..-1, 2.., ..]) * dy2 * dc));

        for (pos, dextra) in sources {
            let index = self.get_extracellular_index(&pos)?;
            for (n, &v) in dextra.iter().enumerate() {
                self.increments[start]
                    .slice_mut(ndarray::s![index[0], index[1], n])
                    .add_assign(v);
            }
        }

        Ok(())
    }

    fn update_fluid_dynamics(&mut self, dt: F) -> Result<(), CalcError> {
        use core::ops::AddAssign;
        use num::traits::AsPrimitive;
        let start = self.increments_start;
        let n_incr = self.increments.len();

        // Adams-Bashforth 3rd order
        let k1: F = 5usize.as_() / 12usize.as_();
        let k2: F = 8usize.as_() / 12usize.as_();
        let k3: F = -1usize.as_() / 12usize.as_();
        self.extracellular.add_assign(
            &(&self.increments[start] * k1 * dt
                + &self.increments[(start + 1) % n_incr] * k2 * dt
                + &self.increments[(start + 2) % n_incr] * k3 * dt),
        );
        self.extracellular.map_inplace(|x| *x = x.max(F::zero()));

        // TODO DEBUGGING
        // if start == 0 {
        //     todo!();
        // }

        self.increments_start = (self.increments_start + 1) % n_incr;
        Ok(())
    }

    fn get_extracellular_at_pos(
        &self,
        pos: &nalgebra::SVector<F, 2>,
    ) -> Result<ReactionVector<F>, CalcError> {
        let index = self.get_extracellular_index(pos)?;
        let res = ReactionVector::<F>::from_iterator(
            self.helper.dim().2,
            self.extracellular
                .slice(ndarray::s![index[0], index[1], ..])
                .to_owned(),
        );
        Ok(res)
    }

    fn get_neighbor_value(&self, border_info: Self::BorderInfo) -> Self::NeighborValue {
        use ndarray::*;
        // Calculate shared indices plus padding of one
        let BorderInfo { min_sent, max_sent } = border_info;
        let min = min_sent.map(|x| x.saturating_sub(1)).sup(&self.index_min);
        let max = max_sent.map(|x| x.saturating_add(1)).inf(&self.index_max);

        let omin = min - self.index_min;
        let omax = max - self.index_min;
        let values = Zip::from(
            self.extracellular
                .slice(s![omin[0]..omax[0], omin[1]..omax[1], ..])
                .lanes(Axis(2)),
        )
        .and(
            self.ownership_array
                .slice(s![omin[0] + 1..omax[0] + 1, omin[1] + 1..omax[1] + 1]),
        )
        .map_collect(|v, &o| if o { Some(v.to_owned()) } else { None })
        .to_owned();

        NeighborValue { min, max, values }
    }

    fn get_border_info(&self) -> Self::BorderInfo {
        Self::BorderInfo {
            min_sent: self.index_min,
            max_sent: self.index_max,
        }
    }
}

impl<F> CartesianDiffusion2DSubDomain<F>
where
    F: nalgebra::RealField + Copy + num::traits::AsPrimitive<usize>,
    usize: num::traits::AsPrimitive<F>,
{
    fn get_extracellular_index(
        &self,
        pos: &nalgebra::Vector2<F>,
    ) -> Result<nalgebra::Vector2<usize>, CalcError> {
        use num::traits::AsPrimitive;
        let index = (pos - self.reactions_min).component_div(&self.reactions_dx);
        if index
            .iter()
            .enumerate()
            .any(|(n, &x)| x < F::zero() || x > self.index_max[n].as_())
        {
            return Err(CalcError(format!(
                "Could not find index for position {:?}",
                pos
            )));
        }
        Ok(index.map(|x| x.floor().as_()))
    }
}

impl<F> DomainCreateSubDomains<CartesianDiffusion2DSubDomain<F>> for CartesianDiffusion2D<F>
where
    F: nalgebra::RealField + Copy + num::traits::AsPrimitive<usize>,
    usize: num::traits::AsPrimitive<F>,
    F: 'static + num::Float + core::fmt::Debug + num::FromPrimitive,
{
    type SubDomainIndex = usize;
    type VoxelIndex = [usize; 2];

    fn create_subdomains(
        &self,
        n_subdomains: core::num::NonZeroUsize,
    ) -> Result<
        impl IntoIterator<
            Item = (
                Self::SubDomainIndex,
                CartesianDiffusion2DSubDomain<F>,
                Vec<Self::VoxelIndex>,
            ),
        >,
        DecomposeError,
    > {
        let dx = self.reactions_dx;
        let dx_domain = self.domain.get_dx();
        let n_diffusion = dx_domain
            .component_div(&dx)
            .map(|x| (<F as num::Float>::round(x).as_()).max(1));
        let dx = dx_domain
            .component_div(&n_diffusion.map(|x| <usize as num::traits::AsPrimitive<F>>::as_(x)));

        let diffusion_constant = self.diffusion_constant;

        // Calculate lattice points for subdomains
        // Introduce padding on the outside of the simulated domain.
        let [nrows, ncols] = n_diffusion
            .component_mul(&self.domain.get_n_voxels())
            .into();

        let extracellular_total = ndarray::Array3::from_shape_fn(
            (nrows, ncols, self.initial_value.len()),
            |(_, _, n)| self.initial_value[n],
        );

        Ok(self
            .domain
            .create_subdomains(n_subdomains)?
            .into_iter()
            .map(move |(index, subdomain, voxels)| {
                let max_domain = [nrows, ncols].into();
                let mut min: nalgebra::Vector2<usize> = max_domain;
                let mut max: nalgebra::Vector2<usize> = [0; 2].into();
                for vox in subdomain.get_voxels() {
                    min = min.inf(&vox.into());
                    max = max.sup(&vox.into());
                }

                // Multiply with number of voxels in each dimension
                let min = min.component_mul(&n_diffusion);
                // Here we need to add one more step since this is supposed to be an upper limit
                let max = max.component_mul(&n_diffusion) + n_diffusion;
                let max_domain = max_domain.component_mul(&n_diffusion);

                let extracellular = extracellular_total
                    .slice(ndarray::s![min[0]..max[0], min[1]..max[1], ..])
                    .into_owned();

                let reactions_min = min
                    .map(|x| <usize as num::traits::AsPrimitive<F>>::as_(x))
                    .component_mul(&dx);
                // let reactions_max = max
                //     .map(|x| <usize as num::traits::AsPrimitive<F>>::as_(x))
                //     .component_mul(&dx);

                // Has entry `true` if the given point is owned by this subdomain.
                let d = extracellular.dim();
                let mut ownership_array =
                    ndarray::Array2::<bool>::from_elem((d.0 + 2, d.1 + 2), false);
                for v in subdomain.get_voxels() {
                    let one = nalgebra::Vector2::from([1; 2]);
                    let v: nalgebra::Vector2<usize> = v.into();
                    let vox = v.component_mul(&n_diffusion);
                    let voxp1 = (v + one).component_mul(&n_diffusion);
                    let lower = (vox - min).add_scalar(1);
                    let upper = (voxp1 - min).inf(&max_domain).add_scalar(1);
                    ownership_array
                        .slice_mut(ndarray::s![lower[0]..upper[0], lower[1]..upper[1]])
                        .fill(true);
                }

                let sh = extracellular.shape();
                let increment = ndarray::Array3::zeros((sh[0], sh[1], sh[2]));
                let helper = ndarray::Array3::zeros((sh[0] + 2, sh[1] + 2, sh[2]));
                (
                    index,
                    CartesianDiffusion2DSubDomain {
                        subdomain,
                        reactions_min,
                        // reactions_max,
                        reactions_dx: dx,
                        index_min: min,
                        index_max: max,
                        extracellular,
                        ownership_array,
                        diffusion_constant,
                        increments_start: 0,
                        increments: [increment.clone(), increment.clone(), increment.clone()],
                        helper,
                    },
                    voxels,
                )
            }))
    }
}
