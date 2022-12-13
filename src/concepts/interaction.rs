use crate::concepts::errors::CalcError;

// TODO Define trait aliases for Position and Force

pub trait Interaction<Pos, Force> {
    fn force(&self, own_pos: &Pos, ext_pos: &Pos) -> Option<Result<Force, CalcError>>;
}
