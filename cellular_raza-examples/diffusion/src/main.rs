use std::ops::AddAssign;

use cellular_raza::{building_blocks::cell_building_blocks::mechanics::NewtonDamped2D, concepts::{CalcError, Mechanics}, concepts_derive::CellAgent};

#[derive(Clone, Debug)]
pub struct SubDomain {
    total_concentration: ndarray::Array3<f64>,
    helper: ndarray::Array3<f64>,
    increment: ndarray::Array3<f64>,
    diffusion_constant: f64,
    index: [usize; 2],
    min: [f64; 2],
    max: [f64; 2],
    dx: [f64; 2],
}

pub enum Boundary {
    Neumann,
    Dirichlet,
}

trait FluidDynamics<Pos, Conc, Float> {
    type Boundary;

    fn update_fluid_dynamics<'a, I, J>(&mut self, dt: Float, neighbours: &'a I, sources: &'a J) -> Result<(), CalcError>
    where
        Pos: 'static,
        Conc: 'static,
        Self::Boundary: 'static,
        &'a I: IntoIterator<Item = &'a Self::Boundary>,
        &'a J: IntoIterator<Item = &'a (Pos, Conc)>;

    fn get_concentration_at_pos(&self, pos: &Pos) -> Result<Conc, CalcError>;
}

impl FluidDynamics<nalgebra::SVector<f64, 2>, ndarray::Array1<f64>, f64> for SubDomain {
    type Boundary = ([usize; 2], ndarray::Array3<f64>);

    fn update_fluid_dynamics<'a, I, J>(&mut self, dt: f64, neighbours: &'a I, sources: &'a J) -> Result<(), CalcError>
    where
        &'a I: IntoIterator<Item = &'a Self::Boundary>,
        &'a J: IntoIterator<Item = &'a (nalgebra::SVector<f64, 2>, ndarray::Array1<f64>)>,
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
        self.helper.slice_mut(s![1..-1,1..-1,..]).assign(&co);

        // Fill outer parts depending on the type of boundary condition
        // _ x x x x x _
        // x _ _ _ _ _ x
        // x _ _ _ _ _ x
        // _ x x x x x _
        for (index, value) in neighbours.into_iter() {
            match (
                self.index[0]==index[0]+1,
                self.index[0]+1==index[0],
                self.index[1]==index[1]+1,
                self.index[1]+1==index[1]
            ) {
                // Assign u_i+1 = b
                (true, false, false, false) => {self.helper.slice_mut(s![0,1..-1,..]).assign(&value.slice(s![-1,..,..]))},
                (false, true, false, false) => {self.helper.slice_mut(s![-1,1..-1,..]).assign(&value.slice(s![0,..,..]))},
                (false, false, true, false) => {self.helper.slice_mut(s![1..-1,0,..]).assign(&value.slice(s![..,-1,..]))},
                (false, false, false, true) => {self.helper.slice_mut(s![1..-1,-1,..]).assign(&value.slice(s![..,0,..]))},
                // Assign u_i+1 = b * Δx² + u_i
                /*(Boundary::Neumann, true, false, false, false) => {self.helper.slice_mut(s![0,1..-1,..]).assign(&(- dx * value + co.slice(s![1,..,..])))},
                (Boundary::Neumann, false, true, false, false) => {self.helper.slice_mut(s![-1,1..-1,..]).assign(&(dx * value + co.slice(s![-2,..,..])))},
                (Boundary::Neumann, false, false, true, false) => {self.helper.slice_mut(s![1..-1,0,..]).assign(&(- dy * value + co.slice(s![..,1,..])))},
                (Boundary::Neumann, false, false, false, true) => {self.helper.slice_mut(s![1..-1,-1,..]).assign(&(dy * value + co.slice(s![..,-2,..])))},*/
                _ => println!("Index  {index:?} is not a valid boundary index! Skipping ..."),
            }
        }

        // Set increment to next time-step to 0.0 everywhere
        self.increment.fill(0.0);
        self.increment.assign(&(-2.0 * dd2 * &self.helper.slice(s![1..-1,1..-1,..])));
        self.increment.add_assign(&(dx.powf(-2.0) * &self.helper.slice(s![2.., 1..-1, ..])));
        self.increment.add_assign(&(dx.powf(-2.0) * &self.helper.slice(s![0..-2, 1..-1, ..])));
        self.increment.add_assign(&(dy.powf(-2.0) * &self.helper.slice(s![1..-1, 2.., ..])));
        self.increment.add_assign(&(dy.powf(-2.0) * &self.helper.slice(s![1..-1, 0..-2, ..])));

        sources.into_iter().map(|(pos, source_increment)| {
            let index = self.get_index_of_position(pos)?;
            self.increment.slice_mut(s![index[0], index[1],..]).add_assign(source_increment);
            Ok(())
        }).collect::<Result<Vec<_>, CalcError>>()?;

        self.total_concentration.add_assign(&(self.diffusion_constant * dt * &self.increment));

        println!("{:6.3}", &self.total_concentration.slice(s![..,..,0]));
        Ok(())
    }

    fn get_concentration_at_pos(&self, pos: &nalgebra::SVector<f64, 2>) -> Result<ndarray::Array1<f64>, CalcError> {
        let index = self.get_index_of_position(pos)?;
        let s = self.total_concentration.shape();
        use ndarray::s;
        let conc = self.total_concentration.slice(s![index[0], index[1], ..]);
        let mut r = ndarray::Array1::zeros(s[2]);
        r.slice_mut(s![..]).assign(&conc);
        Ok(r)
    }
}

impl SubDomain {
    pub fn get_index_of_position(&self, pos: &nalgebra::Vector2<f64>) -> Result<[usize; 2], CalcError> {
        if pos[0] < self.min[0] || pos[0] > self.max[0] || pos[1] < self.min[1] || pos[1] > self.max[1] {
            return Err(CalcError(format!("position {:?} is not contained in domain with boundaries {:?} {:?}", pos, self.min, self.max)));
        }
        let index = [
            ((pos[0] - self.min[0]) / self.dx[0] as f64).round() as usize,
            ((pos[1] - self.min[1]) / self.dx[1] as f64).round() as usize
        ];
        println!("{:?}", index);
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
                let color = BlackWhite::get_color_normalized(c, 0.0, 0.2);
                let rect = plotters::element::Rectangle::new(
                    [
                        (self.min[0] + i as f64 * self.dx[0], self.min[1] + j as f64 * self.dx[1]),
                        (self.min[0] + (i + 1) as f64 * self.dx[0], self.min[1] + (j + 1) as f64 * self.dx[1]),
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
    fn increment_intracellular(&mut self, intrement: &I);
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
    fn increment_intracellular(&mut self, intrement: &ndarray::Array1<f64>) {
        self.intracellular += intrement;
    }

    fn calculate_intracellular_increment(&self) -> Result<ndarray::Array1<f64>, CalcError> {
        Ok(0.0*&self.intracellular)
    }
}

impl ReactionsExtra<ndarray::Array1<f64>, ndarray::Array1<f64>> for MyCell {
    fn calculate_combined_increment(&self, _extracellular: &ndarray::Array1<f64>) -> Result<(ndarray::Array1<f64>, ndarray::Array1<f64>), CalcError> {
        Ok((-0.0*&self.intracellular, 1.0*&self.intracellular))
    }
}

fn main() {
    // Overall parameters
    let n_lattice_points_x = 40;
    let n_lattice_points_y = 40;
    let n_components = 1;

    // Agent setup
    let mut agents = (0..1).map(|_| {
        use num::Zero;
        MyCell {
            mechanics: NewtonDamped2D {
                pos: nalgebra::Vector2::from([0.0; 2]),
                vel: nalgebra::Vector2::zero(),
                damping_constant: 0.1,
                mass: 1.0,
            },
            intracellular: ndarray::Array1::ones(n_components)
        }
    }).collect::<Vec<_>>();

    // Diffusion setup
    let total_concentration =
        ndarray::Array3::zeros((n_lattice_points_x, n_lattice_points_y, n_components));
    /* total_concentration
        .slice_mut(ndarray::s![4, 3, ..])
        .add_assign(15.0);*/
    let mut subdomain = SubDomain {
        total_concentration,
        helper: ndarray::Array3::zeros((n_lattice_points_x+2, n_lattice_points_y+2, n_components)),
        increment: ndarray::Array3::zeros((n_lattice_points_x, n_lattice_points_y, n_components)),
        diffusion_constant: 1.0,
        index: [1, 1],
        min: [-20.0; 2],
        max: [20.0; 2],
        dx: [40.0/n_lattice_points_x as f64, 40.0/n_lattice_points_y as f64],
    };

    let neighbours = vec![
        (
            [2, 1],
            10.0*ndarray::Array3::<f64>::zeros([n_lattice_points_x, n_lattice_points_y, n_components]),
        ),
        (
            [0, 1],
            -10.0*ndarray::Array3::<f64>::zeros([n_lattice_points_x, n_lattice_points_y, n_components]),
        ),
        (
            [1, 2],
            ndarray::Array3::<f64>::zeros([n_lattice_points_x, n_lattice_points_y, n_components]),
        ),
        (
            [1, 0],
            ndarray::Array3::<f64>::zeros([n_lattice_points_x, n_lattice_points_y, n_components]),
        ),
    ];

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