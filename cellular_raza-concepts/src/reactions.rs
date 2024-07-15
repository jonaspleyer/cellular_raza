use crate::CalcError;
use crate::Position;

/// Setter and Getter for intracellular values of a cellagent.
pub trait Intracellular<Ri> {
    /// Sets the current intracellular values.
    fn set_intracellular(&mut self, intracellular: Ri);
    /// Obtains the current intracellular values.
    fn get_intracellular(&self) -> Ri;
}

/// Describes purely intracellular reactions of a cellagent.
///
/// In the most simple case, intracellular values can be assumed to have a homogenous distribution
/// throughout the entire cell.
/// We can then describe them by a list of values $\vec{u}=(u_0,\dots,u_N)$.
pub trait Reactions<Ri>: Intracellular<Ri> {
    /// Calculates the purely intracellular reaction increment.
    /// Users who implement this trait should always use the given argument instead of relying on
    /// values obtained via `self`.
    fn calculate_intracellular_increment(&self, intracellular: &Ri) -> Result<Ri, CalcError>;
}

/// TODO
pub trait ReactionsExtra<Ri, E>: Intracellular<Ri> {
    // TODO
    // type IncrementExtracellular;
    /// TODO
    fn calculate_combined_increment(
        &self,
        intracellular: &Ri,
        extracellular: &E,
    ) -> Result<(Ri, E), CalcError>;
}

/// TODO
pub trait ReactionsContact<Ri, Pos, Float = f64, RInf = ()>: Intracellular<Ri> {
    /// TODO
    fn get_contact_information(&self) -> RInf;

    /// TODO
    fn calculate_contact_increment(
        &self,
        own_intracellular: &Ri,
        ext_intracellular: &Ri,
        own_pos: &Pos,
        ext_pos: &Pos,
        rinf: &RInf,
    ) -> Result<(Ri, Ri), CalcError>;
}

/// Mathematical abstraction similar to the well-known `axpy` method.
///
/// This trait
pub trait Xapy<F> {
    /// TODO
    fn xapy(&self, a: F, y: &Self) -> Self;
}

impl<F, X> Xapy<F> for X
where
    X: for<'a> core::ops::Add<&'a X, Output = X>,
    for<'a> &'a X: core::ops::Mul<F, Output = X>,
{
    fn xapy(&self, a: F, y: &Self) -> Self {
        self * a + y
    }
}

#[allow(unused)]
fn solver_euler_extra<F, C, Ri, E>(
    dt: F,
    cell: &mut C,
    extracellular: &mut E,
) -> Result<(), Box<dyn std::error::Error>>
where
    C: ReactionsExtra<Ri, E>,
    F: num::Zero + num::One + Clone,
    Ri: Xapy<F>,
    E: Xapy<F>,
{
    let intra = cell.get_intracellular();
    let (dintra, dextra) = cell.calculate_combined_increment(&intra, extracellular)?;
    cell.set_intracellular(dintra.xapy(dt.clone(), &intra));
    *extracellular = dextra.xapy(dt, extracellular);
    Ok(())
}

#[allow(unused)]
fn solver_runge_kutta_4th_combined<F, C, Ri, E>(
    dt: F,
    cell: &mut C,
    extracellular: &mut E,
) -> Result<(), Box<dyn std::error::Error>>
where
    C: ReactionsExtra<Ri, E>,
    F: num::Float,
    Ri: Xapy<F> + num::Zero,
    E: Xapy<F> + num::Zero,
{
    let intra = cell.get_intracellular();

    let two = F::one() + F::one();
    let (dintra1, dextra1) = cell.calculate_combined_increment(&intra, extracellular)?;
    let (dintra2, dextra2) = cell.calculate_combined_increment(
        &dintra1.xapy(dt / two, &intra),
        &dextra1.xapy(dt / two, &extracellular),
    )?;
    let (dintra3, dextra3) = cell.calculate_combined_increment(
        &dintra2.xapy(dt / two, &intra),
        &dextra2.xapy(dt / two, &extracellular),
    )?;
    let (dintra4, dextra4) = cell.calculate_combined_increment(
        &dintra3.xapy(dt, &intra),
        &dextra3.xapy(dt, &extracellular),
    )?;
    let six = two + two + two;
    let dintra = dintra1.xapy(
        F::one() / six,
        &dintra2.xapy(
            two / six,
            &dintra3.xapy(two / six, &dintra4.xapy(F::one() / six, &Ri::zero())),
        ),
    );
    let dextra = dextra1.xapy(
        F::one() / six,
        &dextra2.xapy(
            two / six,
            &dextra3.xapy(two / six, &dextra4.xapy(F::one() / six, &E::zero())),
        ),
    );
    cell.set_intracellular(dintra.xapy(dt, &intra));
    *extracellular = dextra.xapy(dt, extracellular);
    Ok(())
}

mod test_plain_float {
    use super::*;

    #[allow(unused)]
    #[derive(Clone)]
    struct MyCell {
        // Intracellular properties
        intracellular: f64,
        production: f64,
        degradation: f64,
        // For contact reactions
        pos: [f64; 2],
        exchange_term: f64,
        reaction_range: f64,
        // Extracellular reactions
        secretion_rate: f64,
    }

    impl Intracellular<f64> for MyCell {
        fn set_intracellular(&mut self, intracellular: f64) {
            self.intracellular = intracellular;
        }

        fn get_intracellular(&self) -> f64 {
            self.intracellular
        }
    }

    impl Reactions<f64> for MyCell {
        fn calculate_intracellular_increment(&self, intracellular: &f64) -> Result<f64, CalcError> {
            Ok(self.production - self.degradation * intracellular)
        }
    }

    impl ReactionsExtra<f64, f64> for MyCell {
        fn calculate_combined_increment(
            &self,
            intracellular: &f64,
            _extracellular: &f64,
        ) -> Result<(f64, f64), CalcError> {
            let secretion = self.secretion_rate * intracellular;
            Ok((-secretion, secretion))
        }
    }

    impl Position<[f64; 2]> for MyCell {
        fn pos(&self) -> [f64; 2] {
            self.pos
        }

        fn set_pos(&mut self, pos: &[f64; 2]) {
            self.pos = *pos;
        }
    }

    impl ReactionsContact<f64, [f64; 2]> for MyCell {
        fn calculate_contact_increment(
            &self,
            own_intracellular: &f64,
            ext_intracellular: &f64,
            own_pos: &[f64; 2],
            ext_pos: &[f64; 2],
            _rinf: &(),
        ) -> Result<(f64, f64), CalcError> {
            let dist =
                ((own_pos[0] - ext_pos[0]).powf(2.0) + (own_pos[1] - ext_pos[1]).powf(2.0)).sqrt();
            if dist < self.reaction_range {
                let exchange = self.exchange_term * (ext_intracellular - own_intracellular);
                Ok((exchange, -exchange))
            } else {
                Ok((0.0, 0.0))
            }
        }

        fn get_contact_information(&self) -> () {}
    }

    #[test]
    fn euler_reactions_contact() -> Result<(), Box<dyn std::error::Error>> {
        // We engineer these cells such that
        let mut c1 = MyCell {
            pos: [0.0; 2],
            intracellular: 1.0,
            production: 0.0,
            degradation: 0.0,
            exchange_term: 0.1,
            reaction_range: 3.0,
            secretion_rate: 0.0,
        };
        let mut c2 = c1.clone();
        c2.intracellular = 0.0;
        c2.pos = [0.25; 2];

        let dt = 0.02;
        for _ in 0..10_000 {
            // Calculate combined increments
            let p1 = c1.pos.clone();
            let r1 = c1.intracellular;
            let p2 = c2.pos.clone();
            let r2 = c2.intracellular;

            // The first index indicates from where the term originated while the second index
            // shows for which cell the value needs to be added.
            // From cell 1 to 2
            let (dr11, dr12) = c1.calculate_contact_increment(&r1, &r2, &p1, &p2, &())?;
            // From cell 2 to 1
            let (dr22, dr21) = c2.calculate_contact_increment(&r2, &r1, &p2, &p1, &())?;

            // Calculate the combined increments
            let dr1 = dt * (dr11 + dr21) / 2.0;
            let dr2 = dt * (dr12 + dr22) / 2.0;

            // Update the intracellular values of c1 and c2
            c1.set_intracellular(r1 + dr1);
            c2.set_intracellular(r2 + dr2);
        }
        // Test that the resulting concentrations are matching in the end.
        assert!((c1.get_intracellular() - c2.get_intracellular()).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_euler_extra() -> Result<(), Box<dyn std::error::Error>> {
        let x0 = 1.0;
        let mut cell = MyCell {
            pos: [0.0; 2],
            intracellular: x0,
            production: 0.0,
            degradation: 0.0,
            exchange_term: 0.0,
            reaction_range: 0.0,
            secretion_rate: 0.1,
        };
        let mut extracellular = 0.0;

        let dt = 0.002;
        let exact_solution_cell =
            |t: f64, x0: f64, degradation: f64| -> f64 { x0 * (-degradation * t).exp() };

        for n_step in 0..10_000 {
            solver_euler_extra(dt, &mut cell, &mut extracellular)?;
            let x_exact = exact_solution_cell((n_step + 1) as f64 * dt, x0, cell.secretion_rate);
            assert!((cell.get_intracellular() - x_exact).abs() < 1e-4);
        }
        Ok(())
    }

    #[test]
    fn runge_kutta_intracellular() -> Result<(), Box<dyn std::error::Error>> {
        let x0 = 2.0;
        let mut cell = MyCell {
            pos: [0.0; 2],
            intracellular: x0,
            production: 1.0,
            degradation: 0.2,
            exchange_term: 0.0,
            reaction_range: 0.0,
            secretion_rate: 0.0,
        };

        let analytical_solution = |t: f64, x0: f64, production: f64, degradation: f64| -> f64 {
            production / degradation
                * (1.0 - (1.0 - x0 * degradation / production) * (-degradation * t).exp())
        };

        let dt = 1e-0;
        for n_step in 0..100 {
            let intra = cell.get_intracellular();
            // Do the runge-kutta numerical integration steps
            let k1 = cell.calculate_intracellular_increment(&(intra))?;
            let k2 = cell.calculate_intracellular_increment(&(intra + dt / 2.0 * k1))?;
            let k3 = cell.calculate_intracellular_increment(&(intra + dt / 2.0 * k2))?;
            let k4 = cell.calculate_intracellular_increment(&(intra + dt * k3))?;

            // Calculate the total increment
            let dintra = dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

            // Update the values
            cell.set_intracellular(intra + dintra);
            let exact_result = analytical_solution(
                dt * (n_step + 1) as f64,
                x0,
                cell.production,
                cell.degradation,
            );
            assert!((cell.get_intracellular() - exact_result).abs() < 1e-4);
        }
        assert!((cell.get_intracellular() - cell.production / cell.degradation).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_generic_solver_runge_kutta_combined() -> Result<(), Box<dyn std::error::Error>> {
        let x0 = 10.0;
        let mut cell = MyCell {
            pos: [0.0; 2],
            intracellular: x0,
            production: 0.0,
            degradation: 0.0,
            exchange_term: 0.0,
            reaction_range: 0.0,
            secretion_rate: 0.15,
        };
        let mut extracellular = 1.0;

        let exact_result =
            |t: f64, x0: f64, degradation: f64| -> f64 { x0 * (-degradation * t).exp() };

        let dt = 0.1;
        for n_step in 0..1_000 {
            let t = (n_step + 1) as f64 * dt;
            solver_runge_kutta_4th_combined(dt, &mut cell, &mut extracellular)?;
            let exact_value = exact_result(t, x0, cell.secretion_rate);
            assert!((exact_value - cell.get_intracellular()).abs() < 1e-6);
        }
        assert!(cell.get_intracellular().abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn adams_bashforth_3rd_extracellular() -> Result<(), Box<dyn std::error::Error>> {
        let x0 = 10.0;
        let mut cell = MyCell {
            pos: [0.0; 2],
            intracellular: x0,
            production: 0.0,
            degradation: 0.0,
            exchange_term: 0.0,
            reaction_range: 0.0,
            secretion_rate: 0.1,
        };
        let mut extracellular = 0.0;

        let mut dcombined1 = None;
        let mut dcombined2 = None;

        let exact_solution =
            |t: f64, x0: f64, degradation: f64| -> f64 { x0 * (-degradation * t).exp() };

        let dt = 0.1;
        for n_step in 0..1_000 {
            let intra = cell.get_intracellular();

            // Calculate the total increment depening on how many previous values we have
            let (dintra, dextra) = cell.calculate_combined_increment(&intra, &extracellular)?;
            match (dcombined1, dcombined2) {
                (Some((dintra1, dextra1)), Some((dintra2, dextra2))) => {
                    let h1 = 23.0 / 12.0;
                    let h2 = -16.0 / 12.0;
                    let h3 = 5.0 / 12.0;
                    cell.set_intracellular(
                        intra + dt * (h1 * dintra + h2 * dintra1 + h3 * dintra2),
                    );
                    extracellular += dt * (h1 * dextra + h2 * dextra1 + h3 * dextra2);
                }
                (Some((dintra1, dextra1)), None) => {
                    let h1 = 3.0 / 2.0;
                    let h2 = -1.0 / 2.0;
                    cell.set_intracellular(intra + dt * (h1 * dintra + h2 * dintra1));
                    extracellular += dt * (h1 * dextra + h2 * dextra1);
                }
                // This is the euler method
                _ => {
                    cell.set_intracellular(intra + dt * dintra);
                    extracellular += dt * dextra;
                }
            }
            // Reset the increments
            dcombined2 = dcombined1;
            dcombined1 = Some((dintra, dextra));
            assert!((cell.get_intracellular() + extracellular - x0).abs() < 1e-6);

            // Calculate the exact value and commpare
            let exact_value = exact_solution((n_step + 1) as f64 * dt, x0, cell.secretion_rate);
            println!("{} {}", cell.get_intracellular(), exact_value);
            assert!((cell.get_intracellular() - exact_value).abs() < 1e-3);
        }
        Ok(())
    }
}

/// ```
/// use cellular_raza_concepts::{CellAgent, Intracellular, Reactions, CalcError};
/// struct MyReactions;
/// impl Intracellular<i16> for MyReactions {
///     fn get_intracellular(&self) -> i16 {
///         42
///     }
///     fn set_intracellular(&mut self, _intracellular: i16) {}
/// }
/// impl Reactions<i16> for MyReactions {
///     fn calculate_intracellular_increment(&self, intracellular: &i16) -> Result<i16,
///     CalcError> {
///         Ok(-1)
///     }
/// }
/// #[derive(CellAgent)]
/// struct MyCell {
///     #[Reactions]
///     reactions: MyReactions,
/// }
/// let mycell = MyCell {
///     reactions: MyReactions,
/// };
/// assert_eq!(mycell.get_intracellular(), 42);
/// assert_eq!(mycell.calculate_intracellular_increment(&7).unwrap(), -1);
/// ```
#[allow(unused)]
fn derive_reactions() {}

/// ```
/// use cellular_raza_concepts::{CellAgent, Intracellular, Reactions, CalcError};
/// struct MyIntracellular(String);
/// impl Intracellular<String> for MyIntracellular {
///     fn get_intracellular(&self) -> String {
///         format!("{}", self.0)
///     }
///     fn set_intracellular(&mut self, intracellular: String) {
///         self.0 = intracellular;
///     }
/// }
/// #[derive(CellAgent)]
/// struct MyAgent {
///     #[Intracellular]
///     intracellular: MyIntracellular,
/// }
/// let mut myagent = MyAgent {
///     intracellular: MyIntracellular("42".to_owned()),
/// };
/// assert_eq!(myagent.get_intracellular(), format!("42"));
/// myagent.set_intracellular(format!("this wind is nice"));
/// assert_eq!(myagent.get_intracellular(), "this wind is nice".to_owned());
/// ```
#[allow(unused)]
fn derive_intracellular() {}

/// TODO
#[allow(unused)]
fn derive_reactions_raw() {}

/// TODO
#[allow(unused)]
fn derive_reactions_contact() {}

/* mod test_particle_sim {
    use super::*;

    struct Particle([f32; 3]);
    struct ParticleCollection(Vec<Particle>);

    struct Cell {
        intracellular: ParticleCollection,
    }
}*/
