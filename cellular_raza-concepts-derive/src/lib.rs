// #![deny(missing_docs)]
// #![warn(clippy::missing_docs_in_private_items)]

#[macro_use]
mod cell_agent;
mod subdomain;

/// Derive cellular concepts
///
/// This macro allows to simply derive already implemented concepts
/// from struct fields.
/// Currently the only allowed notation is by defining macros with curly braces.
/// ```ignore
/// #[derive(CellAgent)]
/// struct MyCell {
///     #[Cycle]
///     cycle: MyCycle,
///     ...
/// }
/// ```
/// Some attributes also require to specify types as well.
/// ```ignore
/// struct MyCell {
///     #[Mechanics]
///     interaction: MyMechanics,
///     ...
/// }
/// ```
/// A summary can be seen in the following table
///
/// | Attribute | Type Arguments |
/// | --- | --- |
/// | `Cycle`                   | `(Float=f64)`                                                         |
/// | `Mechanics`               | `(Pos, Vel, For, Float=f64)`                                          |
/// | `Interaction`             | `(Pos, Vel, For, Inf=())`                                             |
/// | `CellularReactions`       | `(ConcVecIntracellular, ConcVecExtracellular=ConcVecIntracellular)`   |
/// | `ExtracellularGradient`   | `(ConcGradientExtracellular)`                                         |
/// | `Volume`                  | `(Float=f64)`                                                         |
///
/// For a description of these type arguments see `cellular_raza_concepts` crate.
#[proc_macro_derive(
    CellAgent,
    attributes(
        Cycle,
        Mechanics,
        Interaction,
        Reactions,
        ExtracellularGradient,
        Volume,
    )
)]
pub fn derive_cell_agent(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    cell_agent::derive_cell_agent(input)
}

#[proc_macro_derive(SubDomain, attributes(Base, Mechanics, SortCells))]
pub fn derive_subdomain(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    subdomain::derive_subdomain(input)
}
