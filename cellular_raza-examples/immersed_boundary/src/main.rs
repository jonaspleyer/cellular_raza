use cellular_raza::prelude::CalcError;
use rand_chacha::ChaCha8Rng;

trait Reactions<I> {
    fn increment_intracellular(&mut self, increment: &I);
    fn calculate_intracellular_increment(&self, rng: &mut ChaCha8Rng) -> Result<I, CalcError>;
}

trait ReactionsExtra<I, E> {
    fn increment_intracellular(&mut self, increment: &I);
    fn calculate_combined_increment(
        &self,
        extracellular: &E,
        rng: &mut ChaCha8Rng,
    ) -> Result<(I, E), CalcError>;
}

fn main() {}
