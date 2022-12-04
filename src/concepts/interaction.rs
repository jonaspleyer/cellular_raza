use crate::concepts::errors::CalcError;

// TODO Define trait aliases for Position and Force

pub trait Interaction<Pos, Force> {
    fn potential(&self, x: &Pos, y: &Pos) -> Result<Force, CalcError>;
}
