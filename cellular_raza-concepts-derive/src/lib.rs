// #![deny(missing_docs)]
// #![warn(clippy::missing_docs_in_private_items)]

// Note that this way of importing the macros relies on the order in which the modules get
// imported.
#[macro_use]
mod cell_agent;
#[macro_use]
mod subdomain;
mod domain;

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
#[proc_macro_derive(
    CellAgent,
    attributes(
        Cycle,
        Mechanics,
        MechanicsRaw,
        Position,
        Velocity,
        Interaction,
        Reactions,
        Intracellular,
        ExtracellularGradient,
        Volume,
    )
)]
pub fn derive_cell_agent(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    cell_agent::derive_cell_agent(input)
}

#[proc_macro_derive(SubDomain, attributes(Base, SortCells, Mechanics, Force, Reactions))]
pub fn derive_subdomain(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    subdomain::derive_subdomain(input)
}

#[proc_macro_derive(
    Domain,
    attributes(
        Base,
        DomainPartialDerive,
        DomainRngSeed,
        DomainCreateSubDomains,
        SortCells
    )
)]
pub fn derive_domain(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    domain::derive_domain(input)
}
