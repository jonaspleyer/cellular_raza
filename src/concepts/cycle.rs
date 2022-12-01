
pub trait Cycle<C> {
    fn update(dt: &f64, cell: &mut C);
}
