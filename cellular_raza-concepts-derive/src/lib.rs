// #![deny(missing_docs)]
// #![warn(clippy::missing_docs_in_private_items)]

mod cell_agent;
mod domain;

use cell_agent::*;

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
///     #[Mechanics([f64; 3], [f64; 3], [f64; 3])]
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
    // Parse the input tokens into a syntax tree
    let agent_parsed = syn::parse_macro_input!(input as AgentParser);
    let agent = AgentImplementer::from(agent_parsed);

    let mut res = proc_macro2::TokenStream::new();
    res.extend(agent.implement_cycle());
    res.extend(agent.implement_mechanics());
    res.extend(agent.implement_reactions());
    res.extend(agent.implement_interaction());
    res.extend(agent.implement_extracellular_gradient());
    res.extend(agent.implement_volume());

    wrap(res).into()
}
