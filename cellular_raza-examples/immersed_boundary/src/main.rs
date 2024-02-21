use std::ops::AddAssign;

#[derive(Clone, Debug)]
struct SubDomain {
    total_concentration: ndarray::Array3<f64>,
    helper: ndarray::Array3<f64>,
    increment: ndarray::Array3<f64>,
    diffusion_constant: f64,
    index: [usize; 2],
    min: [f64; 2],
    max: [f64; 2],
}

pub enum Boundary {
    Neumann,
    Dirichlet,
}

trait FluidDynamics<Float> {
    fn do_update<'a, J>(&mut self, dt: Float, boundaries: &'a J)
    where
        &'a J: IntoIterator<Item = &'a Boundary>;
}

impl SubDomain {
    fn do_update<'a, J>(&mut self, dt: f64, boundaries: &'a J)
    where
        &'a J: IntoIterator<Item = &'a ([usize; 2], (Boundary, ndarray::Array2<f64>))>,
    {
        use ndarray::s;

        let s = self.total_concentration.shape();
        let dx = (self.max[0] - self.min[0]) / (s[0] - 1) as f64;
        let dy = (self.max[1] - self.min[1]) / (s[1] - 1) as f64;
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
        for (index, (ty, value)) in boundaries.into_iter() {
            match (
                ty,
                self.index[0] == index[0] + 1,
                self.index[0] + 1 == index[0],
                self.index[1] == index[1] + 1,
                self.index[1] + 1 == index[1],
            ) {
                // Assign u_i+1 = b
                (Boundary::Dirichlet, true, false, false, false) => {
                    self.helper.slice_mut(s![0, 1..-1, ..]).assign(&value)
                }
                (Boundary::Dirichlet, false, true, false, false) => {
                    self.helper.slice_mut(s![-1, 1..-1, ..]).assign(&value)
                }
                (Boundary::Dirichlet, false, false, true, false) => {
                    self.helper.slice_mut(s![1..-1, 0, ..]).assign(&value)
                }
                (Boundary::Dirichlet, false, false, false, true) => {
                    self.helper.slice_mut(s![1..-1, -1, ..]).assign(&value)
                }
                // Assign u_i+1 = b * Δx² + u_i
                (Boundary::Neumann, true, false, false, false) => self
                    .helper
                    .slice_mut(s![0, 1..-1, ..])
                    .assign(&(-dx * value + co.slice(s![1, .., ..]))),
                (Boundary::Neumann, false, true, false, false) => self
                    .helper
                    .slice_mut(s![-1, 1..-1, ..])
                    .assign(&(dx * value + co.slice(s![-2, .., ..]))),
                (Boundary::Neumann, false, false, true, false) => self
                    .helper
                    .slice_mut(s![1..-1, 0, ..])
                    .assign(&(-dy * value + co.slice(s![.., 1, ..]))),
                (Boundary::Neumann, false, false, false, true) => self
                    .helper
                    .slice_mut(s![1..-1, -1, ..])
                    .assign(&(dy * value + co.slice(s![.., -2, ..]))),
                _ => println!("Index  {index:?} is not a valid boundary index! Skipping ..."),
            }
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

        self.total_concentration
            .add_assign(&(self.diffusion_constant * dt * &self.increment));

        println!("{:4.3}", &self.total_concentration.slice(s![.., .., 0]));
    }

    fn plot_result(&self, iteration: usize) {
        let s = self.total_concentration.shape();
        let n_lattice_points_x = s[0];
        let n_lattice_points_y = s[1];
        let dx = (self.max[0] - self.min[0]) / (s[0] - 1) as f64;
        let dy = (self.max[1] - self.min[1]) / (s[1] - 1) as f64;

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
                let color = BlackWhite::get_color_normalized(c, -20.0, 20.0);
                let rect = plotters::element::Rectangle::new(
                    [
                        (self.min[0] + i as f64 * dx, self.min[1] + j as f64 * dy),
                        (
                            self.min[0] + (i + 1) as f64 * dx,
                            self.min[1] + (j + 1) as f64 * dy,
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

fn main() {
    let n_lattice_points_x = 10;
    let n_lattice_points_y = 10;
    let n_components = 1;
    let mut total_concentration =
        ndarray::Array3::zeros((n_lattice_points_x, n_lattice_points_y, n_components));
    total_concentration
        .slice_mut(ndarray::s![4, 3, ..])
        .add_assign(15.0);
    let mut subdomain = SubDomain {
        total_concentration,
        helper: ndarray::Array3::zeros((
            n_lattice_points_x + 2,
            n_lattice_points_y + 2,
            n_components,
        )),
        increment: ndarray::Array3::zeros((n_lattice_points_x, n_lattice_points_y, n_components)),
        diffusion_constant: 1.0,
        index: [1, 1],
        min: [0.0; 2],
        max: [10.0; 2],
    };

    let boundaries = vec![
        (
            [2, 1],
            (
                Boundary::Neumann,
                10.0 * ndarray::Array2::<f64>::ones([n_lattice_points_y, n_components]),
            ),
        ),
        (
            [0, 1],
            (
                Boundary::Neumann,
                10.0 * ndarray::Array2::<f64>::ones([n_lattice_points_y, n_components]),
            ),
        ),
        (
            [1, 2],
            (
                Boundary::Neumann,
                ndarray::Array2::<f64>::zeros([n_lattice_points_x, n_components]),
            ),
        ),
        (
            [1, 0],
            (
                Boundary::Neumann,
                ndarray::Array2::<f64>::zeros([n_lattice_points_x, n_components]),
            ),
        ),
    ];

    let dt = 0.01;
    for n in 0..1_000 {
        println!(
            "{}",
            subdomain.total_concentration.sum()
                / n_lattice_points_x as f64
                / n_lattice_points_y as f64
        );
        if n % 20 == 0 {
            subdomain.plot_result(n);
        }
        subdomain.do_update(dt, &boundaries);
    }
}
