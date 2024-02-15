use std::ops::AddAssign;

use cellular_raza::{
    building_blocks::cell_building_blocks::mechanics::NewtonDamped2D,
    concepts::{CalcError, CellAgent, Mechanics},
};

#[derive(Clone, Debug)]
pub struct SubDomain {
    total_concentration: ndarray::Array3<f64>,
    helper: ndarray::Array3<f64>,
    increment: ndarray::Array3<f64>,
    diffusion_constant: f64,
    min: nalgebra::SVector<f64, 2>,
    max: nalgebra::SVector<f64, 2>,
    dx: nalgebra::SVector<f64, 2>,
}

pub enum Boundary {
    Neumann,
    Dirichlet,
}

trait FluidDynamics<Pos, Conc, Float> {
    type NeighborValues;
    type BorderInfo;

    fn update_fluid_dynamics<'a, I, J>(
        &mut self,
        dt: Float,
        neighbours: I,
        sources: J,
    ) -> Result<(), CalcError>
    where
        Pos: 'static,
        Conc: 'static,
        Self::NeighborValues: 'static,
        I: IntoIterator<Item = &'a Self::NeighborValues>,
        J: IntoIterator<Item = &'a (Pos, Conc)>;

    fn get_concentration_at_pos(&self, pos: &Pos) -> Result<Conc, CalcError>;
    fn get_neighbor_values(border_info: &Self::BorderInfo) -> Self::NeighborValues;
}

struct CartesianBorder {
    min: nalgebra::SVector<f64, 2>,
    max: nalgebra::SVector<f64, 2>,
}

struct CartesianNeighbor {
    border: CartesianBorder,
    concentrations: ndarray::Array3<f64>,
}

impl FluidDynamics<nalgebra::SVector<f64, 2>, ndarray::Array1<f64>, f64> for SubDomain {
    type NeighborValues = CartesianNeighbor;
    type BorderInfo = CartesianBorder;

    fn update_fluid_dynamics<'a, I, J>(
        &mut self,
        dt: f64,
        neighbours: I,
        sources: J,
    ) -> Result<(), CalcError>
    where
        I: IntoIterator<Item = &'a Self::NeighborValues>,
        J: IntoIterator<Item = &'a (nalgebra::SVector<f64, 2>, ndarray::Array1<f64>)>,
    {
        use ndarray::s;

        let s = self.total_concentration.shape();
        let dx = (self.max[0] - self.min[0]) / s[0] as f64;
        let dy = (self.max[1] - self.min[1]) / s[1] as f64;
        let dd2 = dx.powf(-2.0) + dy.powf(-2.0);

        // Helper variable to store current concentrations
        let co = &self.total_concentration;

        // Use helper array which is +2 in every spatial dimension larger than the original array
        // We do this to seamlessly incorporate boundary conditions
        self.helper.fill(0.0);
        // Fill inner part of the array
        // _ _ _ _ _ _ _
        // _ x x x x x _
        // _ x x x x x _
        // _ _ _ _ _ _ _
        self.helper.slice_mut(s![1..-1, 1..-1, ..]).assign(&co);

        // Fill outer parts depending on the type of boundary condition
        // _ x x x x x _
        // x _ _ _ _ _ x
        // x _ _ _ _ _ x
        // _ x x x x x _
        for neighbor in neighbours.into_iter() {
            self.merge_values(neighbor)?;
        }

        // Set increment to next time-step to 0.0 everywhere
        self.increment.fill(0.0);
        self.increment
            .assign(&(-2.0 * dd2 * &self.helper.slice(s![1..-1, 1..-1, ..])));
        self.increment
            .add_assign(&(dx.powf(-2.0) * &self.helper.slice(s![2.., 1..-1, ..])));
        self.increment
            .add_assign(&(dx.powf(-2.0) * &self.helper.slice(s![0..-2, 1..-1, ..])));
        self.increment
            .add_assign(&(dy.powf(-2.0) * &self.helper.slice(s![1..-1, 2.., ..])));
        self.increment
            .add_assign(&(dy.powf(-2.0) * &self.helper.slice(s![1..-1, 0..-2, ..])));

        sources
            .into_iter()
            .map(|(pos, source_increment)| {
                let index = self.get_index_of_position(pos)?;
                self.increment
                    .slice_mut(s![index[0], index[1], ..])
                    .add_assign(source_increment);
                Ok(())
            })
            .collect::<Result<Vec<_>, CalcError>>()?;

        self.total_concentration
            .add_assign(&(self.diffusion_constant * dt * &self.increment));

        Ok(())
    }

    fn get_concentration_at_pos(
        &self,
        pos: &nalgebra::SVector<f64, 2>,
    ) -> Result<ndarray::Array1<f64>, CalcError> {
        let index = self.get_index_of_position(pos)?;
        let s = self.total_concentration.shape();
        use ndarray::s;
        let conc = self.total_concentration.slice(s![index[0], index[1], ..]);
        let mut r = ndarray::Array1::zeros(s[2]);
        r.slice_mut(s![..]).assign(&conc);
        Ok(r)
    }

    fn get_neighbor_values(border_info: &Self::BorderInfo) -> Self::NeighborValues {
        todo!()
    }
}

impl SubDomain {
    /// This code relies on the fact that discretization between Subdomains is exactly identical!
    /// Thus we test this while running in debug mode.
    /// Should an error occur at this point, the implementation needs to be reconsidered!
    fn merge_values(&mut self, neighbor: &CartesianNeighbor) -> Result<(), CalcError> {
        // First calculate the intersection of the rectangles
        // TODO In the future we hope to use the component-wise functions
        // https://github.com/dimforge/nalgebra/pull/665
        let min = (self.min - self.dx).zip_map(&neighbor.border.min, |a, b| a.max(b));
        let max = (self.max + self.dx).zip_map(&neighbor.border.max, |a, b| a.min(b));

        // Check that the discretization is identical
        #[cfg(debug_assertions)]
        {
            let s = neighbor.concentrations.shape();
            let s = nalgebra::SVector::<f64, 2>::from([s[0] as f64, s[1] as f64]);
            let dx_neighbor =
                (neighbor.border.max - neighbor.border.min).zip_map(&s, |diff, si| diff / si);
            assert_eq!(
                dx_neighbor, self.dx,
                "spatial discretization does not match! {:?} != {:?}",
                dx_neighbor, self.dx
            );
        }

        // Now calculate which indices we compare of our own domain
        // end the neighbor domain
        let ind_calculator = |upper: &nalgebra::SVector<f64, 2>,
                              lower: &nalgebra::SVector<f64, 2>|
         -> nalgebra::SVector<usize, 2> {
            (upper - lower).component_div(&self.dx).map(|i| i as usize)
        };
        let ind_min_self = ind_calculator(&min, &(self.min - self.dx));
        let ind_max_self = ind_calculator(&max, &(self.min - self.dx));
        let ind_min_neighbor = ind_calculator(&min, &neighbor.border.min);
        let ind_max_neighbor = ind_calculator(&max, &neighbor.border.min);

        use ndarray::s;
        let neighbor_slice = neighbor.concentrations.slice(s![
            ind_min_neighbor[0]..ind_max_neighbor[0],
            ind_min_neighbor[1]..ind_max_neighbor[1],
            ..
        ]);

        self.helper
            .slice_mut(s![
                ind_min_self[0]..ind_max_self[0],
                ind_min_self[1]..ind_max_self[1],
                ..
            ])
            .assign(&neighbor_slice);
        Ok(())
    }

    pub fn get_index_of_position(
        &self,
        pos: &nalgebra::Vector2<f64>,
    ) -> Result<nalgebra::Vector2<usize>, CalcError> {
        if pos[0] < self.min[0]
            || pos[0] > self.max[0]
            || pos[1] < self.min[1]
            || pos[1] > self.max[1]
        {
            return Err(CalcError(format!(
                "position {:?} is not contained in domain with boundaries {:?} {:?}",
                pos, self.min, self.max
            )));
        }
        let index = (pos - self.min).component_div(&self.dx).map(|i| i as usize);
        Ok(index)
    }

    pub fn plot_result(&self, iteration: usize) {
        let s = self.total_concentration.shape();
        let n_lattice_points_x = s[0];
        let n_lattice_points_y = s[1];

        use plotters::prelude::*;
        let name = format!("output_{:08.0}.png", iteration);
        let root = plotters::backend::BitMapBackend::new(&name, (400, 400)).into_drawing_area();
        root.fill(&plotters::style::colors::WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .margin(0)
            .x_label_area_size(0)
            .y_label_area_size(0)
            .build_cartesian_2d(self.min[0]..self.max[0], self.min[1]..self.max[1])
            .unwrap();

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .draw()
            .unwrap();

        let plotting_area = chart.plotting_area();

        for i in 0..n_lattice_points_x {
            for j in 0..n_lattice_points_y {
                let c = self.total_concentration[[i, j, 0]];
                let color = BlackWhite::get_color_normalized(c, 0.0, 0.5);
                let rect = plotters::element::Rectangle::new(
                    [
                        (
                            self.min[0] + i as f64 * self.dx[0],
                            self.min[1] + j as f64 * self.dx[1],
                        ),
                        (
                            self.min[0] + (i + 1) as f64 * self.dx[0],
                            self.min[1] + (j + 1) as f64 * self.dx[1],
                        ),
                    ],
                    color.filled(),
                );
                plotting_area.draw(&rect).unwrap();
            }
        }
        root.present().unwrap();
    }
}

trait Reactions<I> {
    fn increment_intracellular(&mut self, increment: &I);
    fn calculate_intracellular_increment(&self) -> Result<I, CalcError>;
}

trait ReactionsExtra<I, E>: Reactions<I> {
    fn calculate_combined_increment(&self, extracellular: &E) -> Result<(I, E), CalcError>;
}

use cellular_raza::prelude::RngError;

#[derive(CellAgent)]
struct MyCell {
    #[Mechanics(nalgebra::Vector2<f64>, nalgebra::Vector2<f64>, nalgebra::Vector2<f64>)]
    mechanics: NewtonDamped2D,
    intracellular: ndarray::Array1<f64>,
}

impl Reactions<ndarray::Array1<f64>> for MyCell {
    fn increment_intracellular(&mut self, increment: &ndarray::Array1<f64>) {
        self.intracellular += increment;
    }

    fn calculate_intracellular_increment(&self) -> Result<ndarray::Array1<f64>, CalcError> {
        Ok(0.0 * &self.intracellular)
    }
}

impl ReactionsExtra<ndarray::Array1<f64>, ndarray::Array1<f64>> for MyCell {
    fn calculate_combined_increment(
        &self,
        _extracellular: &ndarray::Array1<f64>,
    ) -> Result<(ndarray::Array1<f64>, ndarray::Array1<f64>), CalcError> {
        Ok((-0.0 * &self.intracellular, -1.0 * &self.intracellular))
    }
}

fn main() {
    // Overall parameters
    let n_lattice_points = nalgebra::SVector::<usize, 2>::from([10, 10]);
    let min = nalgebra::SVector::<f64, 2>::from([-5.0; 2]);
    let max = nalgebra::SVector::<f64, 2>::from([5.0; 2]);
    let n_components = 1;

    // Agent setup
    let positions = vec![
        nalgebra::Vector2::from([-2.5, -2.5]),
        // nalgebra::Vector2::from([-2.5,  2.5]),
        // nalgebra::Vector2::from([ 2.5,  2.5]),
        // nalgebra::Vector2::from([ 2.5, -2.5]),
    ];
    let mut agents = positions
        .into_iter()
        .map(|pos| {
            use num::Zero;
            MyCell {
                mechanics: NewtonDamped2D {
                    pos,
                    vel: nalgebra::Vector2::zero(),
                    damping_constant: 0.1,
                    mass: 1.0,
                },
                intracellular: ndarray::Array1::ones(n_components),
            }
        })
        .collect::<Vec<_>>();

    // Diffusion setup
    let total_concentration =
        ndarray::Array3::zeros((n_lattice_points[0], n_lattice_points[1], n_components));
    let mut subdomain = SubDomain {
        total_concentration,
        helper: ndarray::Array3::zeros((
            n_lattice_points[0] + 2,
            n_lattice_points[1] + 2,
            n_components,
        )),
        increment: ndarray::Array3::zeros((n_lattice_points[0], n_lattice_points[1], n_components)),
        diffusion_constant: 1.0,
        min,
        max,
        dx: (max - min).component_div(&n_lattice_points.cast()),
    };

    let neighbours = vec![CartesianNeighbor {
        border: CartesianBorder {
            min: [-10.0, -7.0].into(),
            max: [-5.0, 9.0].into(),
        },
        concentrations: ndarray::Array3::ones([5, 16, n_components]),
    }];

    let dt = 0.01;
    for n in 0..1_000 {
        println!("{}", subdomain.total_concentration.sum() / n_lattice_points_x as f64 / n_lattice_points_y as f64);
        if n % 20 == 0{
            subdomain.plot_result(n);
        }
        let sources = agents.iter_mut().map(|a| {
            let pos = a.pos();
            let extracellular = subdomain.get_concentration_at_pos(&pos).unwrap();
            let (intra, extra) = a.calculate_combined_increment(&extracellular).unwrap();
            a.increment_intracellular(&(dt * &intra));
            (pos, extra)
        }).collect::<Vec<_>>();
        subdomain.update_fluid_dynamics(dt, &neighbours, &sources).unwrap();
    }
}
