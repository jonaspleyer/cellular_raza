use serde::{Deserialize, Serialize};

use cellular_raza_concepts::StepsizeError;

/// A [TimeEvent] describes that a certain action is to be executed after the next iteration step.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub enum TimeEvent {
    /// Saves a partial simulation run which is suitable for data readout but not full recovery of the
    /// simulation for restarting.
    PartialSave,
    /// Performs a complete save from which the simulation should be able to be recovered.
    FullSave,
}

pub struct NextTimePoint<F> {
    pub increment: F,
    pub time: F,
    pub iteration: i64,
    pub event: Option<TimeEvent>,
}

/// Increments time of the simulation
///
/// In the future we hope to add adaptive steppers depending on a specified accuracy function.
pub trait TimeStepper<F> {
    /// Advances the time stepper to the next time point. Also returns if there is an event
    /// scheduled to take place and the next time value and iteration number
    #[must_use]
    fn advance(&mut self) -> Result<Option<NextTimePoint<F>>, StepsizeError>;

    /// Retrieved the last point at which the simulation was fully recovered.
    /// This might be helpful in the future when error handling is more mature and able to recover.
    fn get_last_full_save(&self) -> Option<(F, i64)>;
}

/// Time stepping with a fixed time length
///
/// This time-stepper increments the time variable by the same length.
/// ```
/// # use cellular_raza_core::backend::chili::FixedStepsize;
/// let t0 = 1.0;
/// let dt = 0.2;
/// let partial_save_points = vec![3.0, 5.0, 11.0, 20.0];
/// let time_stepper = FixedStepsize::from_partial_save_points(t0, dt, partial_save_points).unwrap();
/// ```
#[derive(Clone, Deserialize, Serialize)]
pub struct FixedStepsize<F> {
    // The stepsize which was fixed
    dt: F,
    t0: F,
    // An ordered set of time points to store every value at which we should evaluate
    all_events: Vec<(F, i64, TimeEvent)>,
    current_time: F,
    current_iteration: i64,
    maximum_iterations: i64,
    current_event: Option<TimeEvent>,
    past_events: Vec<(F, i64, TimeEvent)>,
}

impl<F> FixedStepsize<F>
where
    F: num::Float + num::ToPrimitive,
{
    /// Simple function to construct the stepper from an initial time point, the time increment and
    /// the time points at which the simulation should be saved. Notice that these saves do not cover
    /// [FullSaves](TimeEvent::FullSave) but only [PartialSaves](TimeEvent::PartialSave).
    pub fn from_partial_save_points(
        t0: F,
        dt: F,
        partial_save_points: Vec<F>,
    ) -> Result<Self, StepsizeError> {
        // Sort the save points
        let mut save_points = partial_save_points;
        save_points.sort_by(|x, y| x.partial_cmp(y).unwrap());
        if save_points.iter().any(|x| t0 > *x) {
            return Err(StepsizeError(
                "Invalid time configuration! Evaluation time point is before starting time point."
                    .to_owned(),
            ));
        }
        let last_save_point = save_points
            .clone()
            .into_iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .ok_or(StepsizeError(
                "No savepoints specified. Simulation will not save any results.".to_owned(),
            ))?;
        let maximum_iterations =
            (((last_save_point - t0) / dt).round())
                .to_i64()
                .ok_or(StepsizeError(
                    "An error in casting of float type to i64 occurred".to_owned(),
                ))?;
        let all_events = save_points
            .clone()
            .into_iter()
            .map(|x| {
                (
                    x,
                    ((x - t0) / dt).round().to_i64().unwrap(),
                    TimeEvent::PartialSave,
                )
            })
            .collect();

        let current_event = if t0
            == save_points
                .into_iter()
                .min_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap()
        {
            Some(TimeEvent::PartialSave)
        } else {
            None
        };

        Ok(Self {
            dt,
            t0,
            all_events,
            current_time: t0,
            current_iteration: 0,
            maximum_iterations,
            // TODO check this again
            current_event,
            past_events: Vec::new(),
        })
    }
}

impl<F> TimeStepper<F> for FixedStepsize<F>
where
    F: num::Float + num::FromPrimitive,
{
    fn advance(&mut self) -> Result<Option<NextTimePoint<F>>, StepsizeError> {
        self.current_iteration += 1;
        self.current_time = F::from_i64(self.current_iteration).ok_or(StepsizeError(
            "Error when casting from i64 to floating point value".to_owned(),
        ))? * self.dt
            + self.t0;
        // TODO Check if a current event should take place
        let event = self
            .all_events
            .iter()
            .filter(|(_, iteration, _)| *iteration == self.current_iteration)
            .map(|(_, _, event)| event.clone())
            .last();

        if self.current_iteration <= self.maximum_iterations {
            Ok(Some(NextTimePoint {
                increment: self.dt,
                time: self.current_time,
                iteration: self.current_iteration,
                event,
            }))
        } else {
            Ok(None)
        }
    }

    fn get_last_full_save(&self) -> Option<(F, i64)> {
        self.past_events
            .clone()
            .into_iter()
            .filter(|(_, _, event)| *event == TimeEvent::FullSave)
            .last()
            .and_then(|x| Some((x.0, x.1)))
    }
}

#[cfg(test)]
pub mod test_time_stepper {
    use rand::Rng;
    use rand::SeedableRng;

    use super::*;

    fn generate_new_fixed_stepper<F>(rng_seed: u64) -> FixedStepsize<F>
    where
        F: num::Float + From<f32>,
    {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(rng_seed);
        let t0 = <F as From<_>>::from(rng.gen_range(0.0..1.0));
        let dt = <F as From<_>>::from(rng.gen_range(0.1..2.0));
        let save_points = vec![
            <F as From<_>>::from(rng.gen_range(0.01..1.8)),
            <F as From<_>>::from(rng.gen_range(2.01..3.8)),
            <F as From<_>>::from(rng.gen_range(4.01..5.8)),
            <F as From<_>>::from(rng.gen_range(6.01..7.8)),
        ];
        FixedStepsize::<F>::from_partial_save_points(t0, dt, save_points).unwrap()
    }

    #[test]
    fn initialization() {
        let t0 = 1.0;
        let dt = 0.2;
        let save_points = vec![3.0, 5.0, 11.0, 20.0];
        let time_stepper = FixedStepsize::from_partial_save_points(t0, dt, save_points).unwrap();
        assert_eq!(t0, time_stepper.current_time);
        assert_eq!(0.2, time_stepper.dt);
        assert_eq!(0, time_stepper.current_iteration);
        assert_eq!(None, time_stepper.current_event);
    }

    #[test]
    #[should_panic]
    fn panic_wrong_save_points() {
        let t0 = 10.0;
        let dt = 0.2;
        let save_points = vec![3.0, 5.0, 11.0, 20.0];
        // This call should fail since t0 is larger than the first two save points
        let _time_stepper = FixedStepsize::from_partial_save_points(t0, dt, save_points).unwrap();
    }

    #[test]
    fn stepping_1() {
        let t0 = 1.0;
        let dt = 0.2;
        let save_points = vec![3.0, 5.0, 11.0, 20.0];
        let mut time_stepper =
            FixedStepsize::from_partial_save_points(t0, dt, save_points).unwrap();

        for i in 1..11 {
            let next = time_stepper.advance().unwrap().unwrap();
            assert_eq!(dt, next.increment);
            assert_eq!(t0 + i as f64 * dt, next.time);
            assert_eq!(i as i64, next.iteration);
            if i == 10 {
                assert_eq!(Some(TimeEvent::PartialSave), next.event);
            } else {
                assert_eq!(None, next.event);
            }
        }
    }

    #[test]
    fn stepping_2() {
        let t0 = 0.0;
        let dt = 0.1;
        let save_points = vec![0.5, 0.7, 0.9, 1.0];
        let mut time_stepper =
            FixedStepsize::from_partial_save_points(t0, dt, save_points.clone()).unwrap();

        assert_eq!(t0, time_stepper.current_time);
        for i in 1..11 {
            let next = time_stepper.advance().unwrap().unwrap();
            assert_eq!(dt, next.increment);
            assert_eq!(t0 + i as f64 * dt, next.time);
            assert_eq!(i as i64, next.iteration);
            if save_points.contains(&next.time) {
                assert_eq!(Some(TimeEvent::PartialSave), next.event);
            }
        }
    }

    fn test_stepping(rng_seed: u64) {
        let mut time_stepper = generate_new_fixed_stepper::<f32>(rng_seed);

        for _ in 0..100 {
            let res = time_stepper.advance().unwrap();
            match res {
                Some(_) => (),
                None => return,
            }
        }
        panic!("The time stepper should have reached the end by now");
    }

    #[test]
    fn stepping_end_0() {
        test_stepping(0);
    }

    #[test]
    fn stepping_end_1() {
        test_stepping(1);
    }

    #[test]
    fn stepping_end_2() {
        test_stepping(2);
    }

    #[test]
    fn stepping_end_3() {
        test_stepping(3);
    }
}
