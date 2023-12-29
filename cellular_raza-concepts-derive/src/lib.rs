// #![warn(missing_docs)]
// #![warn(clippy::missing_docs_in_private_items)]

use proc_macro::TokenStream;
use quote::quote;

#[allow(unused)]
struct AgentParser {
    attrs: Vec<syn::Attribute>,
    vis: syn::Visibility,
    struct_token: syn::Token![struct],
    name: syn::Ident,
    generics: syn::Generics,
    aspects: AspectFields,
}

impl syn::parse::Parse for AgentParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            attrs: input.call(syn::Attribute::parse_outer)?,
            vis: input.parse()?,
            struct_token: input.parse()?,
            name: input.parse()?,
            generics: input.parse()?,
            aspects: input.parse()?,
        })
    }
}

#[derive(Clone)]
struct MechanicsParser {
    position: syn::Type,
    _comma_1: syn::Token![,],
    velocity: syn::Type,
    _comma_2: syn::Token![,],
    force: syn::Type,
}

impl syn::parse::Parse for MechanicsParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _mechanics: syn::Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        Ok(Self {
            position: content.parse()?,
            _comma_1: content.parse()?,
            velocity: content.parse()?,
            _comma_2: content.parse()?,
            force: content.parse()?,
        })
    }
}

struct MechanicsImplementer {
    position: syn::Type,
    velocity: syn::Type,
    force: syn::Type,
    field_type: syn::Type,
    field_name: Option<syn::Ident>,
}

#[derive(Clone)]
struct InteractionParser {
    position: syn::Type,
    _comma_1: syn::Token![,],
    velocity: syn::Type,
    _comma_2: syn::Token![,],
    force: syn::Type,
    _comma_3: Option<syn::Token![,]>,
    information: syn::Type,
}

impl syn::parse::Parse for InteractionParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _interaction: syn::Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        Ok(Self {
            position: content.parse()?,
            _comma_1: content.parse()?,
            velocity: content.parse()?,
            _comma_2: content.parse()?,
            force: content.parse()?,
            _comma_3: content.parse().ok(),
            information: if content.is_empty() {
                syn::parse_quote!(())
            } else {
                content.parse()?
            }
        })
    }
}

struct InteractionImplementer {
    position: syn::Type,
    velocity: syn::Type,
    force: syn::Type,
    information: syn::Type,
    field_type: syn::Type,
    field_name: Option<syn::Ident>,
}

#[derive(Clone)]
struct CellularReactionsParser {
    concvecintracellular: syn::Type,
    _comma: Option<syn::Token![,]>,
    concvecextracellular: syn::Type,
}

impl syn::parse::Parse for CellularReactionsParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _cellular_reactions: syn::Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        let concvecintracellular: syn::Type = content.parse()?;
        let _comma = content.parse()?;
        let concvecextracellular = if content.is_empty() {
            concvecintracellular.clone()
        } else {
            content.parse()?
        };
        Ok(Self {
            concvecintracellular,
            _comma,
            concvecextracellular,
        })
    }
}

struct CellularReactionsImplementer {
    concvecintracellular: syn::Type,
    concvecextracellular: syn::Type,
    field_type: syn::Type,
    field_name: Option<syn::Ident>,
}

#[derive(Clone)]
struct ExtracellularGradientParser {
    extracellular_gradient: syn::Type,
}

impl syn::parse::Parse for ExtracellularGradientParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _extracellular_gradients: syn::Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        Ok(Self {
            extracellular_gradient: content.parse()?,
        })
    }
}

struct ExtracellularGradientImplementer {
    extracellular_gradient: syn::Type,
    field_type: syn::Type,
}

#[derive(Clone)]
struct VolumeParser;

impl syn::parse::Parse for VolumeParser {
    #[allow(unused)]
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _volume: syn::Ident = input.parse()?;
        Ok(Self)
    }
}

struct VolumeImplementer {
    field_type: syn::Type,
    field_name: Option<syn::Ident>,
}

#[derive(Clone)]
struct CycleParser;

struct CycleImplementer {
    field_type: syn::Type,
}

impl syn::parse::Parse for CycleParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _cycle: syn::Ident = input.parse()?;
        Ok(Self)
    }
}

#[derive(Clone)]
enum Aspect {
    Mechanics(MechanicsParser),
    Cycle(CycleParser),
    Interaction(InteractionParser),
    CellularReactions(CellularReactionsParser),
    ExtracellularGradient(ExtracellularGradientParser),
    Volume(VolumeParser),
}

impl Aspect {
    fn from_attribute(attr: &syn::Attribute) -> syn::Result<Option<Self>> {
        let path = attr.meta.path().get_ident();
        let cmp = |c: &str| path.is_some_and(|p| p.to_string() == c);

        let s = &attr.meta;
        let stream: TokenStream = quote!(#s).into();

        if cmp("Mechanics") {
            let parsed: MechanicsParser = syn::parse(stream)?;
            return Ok(Some(Aspect::Mechanics(parsed)));
        }

        if cmp("Cycle") {
            let parsed: CycleParser = syn::parse(stream)?;
            return Ok(Some(Aspect::Cycle(parsed)));
        }

        if cmp("Interaction") {
            let parsed: InteractionParser = syn::parse(stream)?;
            return Ok(Some(Aspect::Interaction(parsed)));
        }

        if cmp("CellularReactions") {
            let parsed: CellularReactionsParser = syn::parse(stream)?;
            return Ok(Some(Aspect::CellularReactions(parsed)));
        }

        if cmp("ExtracellularGradient") {
            let parsed: ExtracellularGradientParser = syn::parse(stream)?;
            return Ok(Some(Aspect::ExtracellularGradient(parsed)));
        }

        if cmp("Volume") {
            let parsed: VolumeParser = syn::parse(stream)?;
            return Ok(Some(Aspect::Volume(parsed)));
        }

        Ok(None)
    }
}

#[derive(Clone)]
struct AspectField {
    aspects: Vec<Aspect>,
    field: syn::Field,
}

impl syn::parse::Parse for AspectField {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let field: syn::Field = input.call(syn::Field::parse_named)?;

        let mut errors = vec![];
        let aspects = field
            .attrs
            .iter()
            .map(Aspect::from_attribute)
            .filter_map(|r| r.map_err(|e| errors.push(e)).ok())
            .filter_map(|s| s)
            .collect::<Vec<_>>();
        for e in errors.into_iter() {
            return Err(e);
        }
        Ok(Self { aspects, field })
    }
}

struct AspectFields {
    #[allow(unused)]
    brace_token: syn::token::Brace,
    aspect_fields: syn::punctuated::Punctuated<AspectField, syn::token::Comma>,
}

impl syn::parse::Parse for AspectFields {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        Ok(Self {
            brace_token: syn::braced!(content in input),
            aspect_fields: content.call(
                syn::punctuated::Punctuated::<AspectField, syn::token::Comma>::parse_terminated,
            )?,
        })
    }
}

struct AgentImplementer {
    name: syn::Ident,
    generics: syn::Generics,
    cycle: Option<CycleImplementer>,
    mechanics: Option<MechanicsImplementer>,
    interaction: Option<InteractionImplementer>,
    cellular_reactions: Option<CellularReactionsImplementer>,
    extracellular_gradient: Option<ExtracellularGradientImplementer>,
    volume: Option<VolumeImplementer>,
}

impl From<AgentParser> for AgentImplementer {
    fn from(value: AgentParser) -> Self {
        let mut cycle = None;
        let mut mechanics = None;
        let mut interaction = None;
        let mut cellular_reactions = None;
        let mut extracellular_gradient = None;
        let mut volume = None;

        value
            .aspects
            .aspect_fields
            .into_iter()
            .for_each(|aspect_field| {
                aspect_field
                    .aspects
                    .into_iter()
                    .for_each(|aspect| match aspect {
                        Aspect::Cycle(_) => {
                            cycle = Some(CycleImplementer {
                                field_type: aspect_field.field.ty.clone(),
                            })
                        }
                        Aspect::Mechanics(p) => {
                            mechanics = Some(MechanicsImplementer {
                                position: p.position,
                                velocity: p.velocity,
                                force: p.force,
                                field_type: aspect_field.field.ty.clone(),
                                field_name: aspect_field.field.ident.clone(),
                            })
                        }
                        Aspect::Interaction(p) => {
                            interaction = Some(InteractionImplementer {
                                position: p.position,
                                velocity: p.velocity,
                                force: p.force,
                                information: p.information,
                                field_type: aspect_field.field.ty.clone(),
                                field_name: aspect_field.field.ident.clone(),
                            })
                        }
                        Aspect::CellularReactions(p) => {
                            cellular_reactions = Some(CellularReactionsImplementer {
                                concvecintracellular: p.concvecintracellular,
                                concvecextracellular: p.concvecextracellular,
                                field_type: aspect_field.field.ty.clone(),
                                field_name: aspect_field.field.ident.clone(),
                            })
                        }
                        Aspect::ExtracellularGradient(p) => {
                            extracellular_gradient = Some(ExtracellularGradientImplementer {
                                extracellular_gradient: p.extracellular_gradient,
                                field_type: aspect_field.field.ty.clone(),
                            })
                        }
                        Aspect::Volume(p) => {
                            volume = Some(VolumeImplementer {
                                field_type: aspect_field.field.ty.clone(),
                                field_name: aspect_field.field.ident.clone(),
                            })
                        }
                    })
            });

        Self {
            name: value.name,
            generics: value.generics,
            cycle,
            mechanics,
            interaction,
            cellular_reactions,
            extracellular_gradient,
            volume,
        }
    }
}

fn wrap(input: TokenStream) -> TokenStream {
    // Get the proc_macro2::TokenStream type right here
    #[allow(unused)]
    let mut input_new = quote!();
    input_new = input.into();

    TokenStream::from(quote! {
        #[allow(non_upper_case_globals)]
        const _: () = {
            // TODO consider adding specific import of cellular_raza or cellular_raza_concepts crate
            // extern crate cellular_raza as _cr;
            // or
            // extern crate cellular_raza_concepts as _cr;
            //
            // Also put a _cr::prelude::TRAIT in front of every implemented trait
            // This is currently not possible to do at compile time without any hacks (to my knowledge)
            #input_new
        };
    })
}

impl AgentImplementer {
    fn implement_cycle(&self) -> TokenStream {
        let struct_name = &self.name;
        let struct_generics = &self.generics;

        if let Some(cycle_implementer) = &self.cycle {
            let field_type = &cycle_implementer.field_type;

            let new_stream = quote!(
                #[automatically_derived]
                impl #struct_generics Cycle<#struct_name> for #struct_name #struct_generics {
                    fn update_cycle(
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: &f64,
                        cell: &mut Self,
                    ) -> Option<CycleEvent> {
                        <#field_type as Cycle<#struct_name>>::update_cycle(rng, dt, cell)
                    }

                    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Self) -> Result<Self, DivisionError> {
                        <#field_type as Cycle<#struct_name>>::divide(rng, cell)
                    }

                    fn update_conditional_phased_death(
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: &f64,
                        cell: &mut Self,
                    ) -> Result<bool, DeathError> {
                        <#field_type as Cycle<#struct_name>>::update_conditional_phased_death(rng, dt, cell)
                    }
                }
            );
            return TokenStream::from(new_stream);
        }
        TokenStream::new()
    }

    fn implement_mechanics(&self) -> TokenStream {
        let struct_name = &self.name;
        let struct_generics = &self.generics;

        if let Some(mechanics_implementer) = &self.mechanics {
            let position = &mechanics_implementer.position;
            let velocity = &mechanics_implementer.velocity;
            let force = &mechanics_implementer.force;

            let tokens = quote!(#position, #velocity, #force);
            let field_type = &mechanics_implementer.field_type;
            let field_name = &mechanics_implementer.field_name;

            let res = quote! {
                #[automatically_derived]
                impl #struct_generics Mechanics<#tokens> for #struct_name #struct_generics
                {
                    fn pos(&self) -> #position {
                        <#field_type as Mechanics<#tokens>>::pos(&self.#field_name)
                    }
                    fn velocity(&self) -> #velocity {
                        <#field_type as Mechanics<#tokens>>::velocity(&self.#field_name)
                    }
                    fn set_pos(&mut self, pos: &#position) {
                        <#field_type as Mechanics<#tokens>>::set_pos(&mut self.#field_name, pos)
                    }
                    fn set_velocity(&mut self, velocity: &#velocity) {
                        <#field_type as Mechanics<#tokens>>::set_velocity(&mut self.#field_name, velocity)
                    }
                    fn calculate_increment(&self, force: #force) -> Result<(#position, #velocity), CalcError> {
                        <#field_type as Mechanics<#tokens>>::calculate_increment(&self.#field_name, force)
                    }
                    fn set_random_variable(&mut self,
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: f64,
                    ) -> Result<Option<f64>, RngError> {
                        <#field_type as Mechanics<#tokens>>::set_random_variable(&mut self.#field_name, rng, dt)
                    }
                }
            };
            return TokenStream::from(res);
        }
        TokenStream::new()
    }

    fn implement_interaction(&self) -> TokenStream {
        let struct_name = &self.name;
        let struct_generics = &self.generics;

        if let Some(interaction_implementer) = &self.interaction {
            let field_name = &interaction_implementer.field_name;
            let field_type = &interaction_implementer.field_type;
            let position = &interaction_implementer.position;
            let velocity = &interaction_implementer.velocity;
            let force = &interaction_implementer.force;
            let information = &interaction_implementer.information;

            let res = quote! {
                #[automatically_derived]
                impl #struct_generics Interaction<
                    #position,
                    #velocity,
                    #force,
                    #information
                > for #struct_name #struct_generics {
                    fn get_interaction_information(&self) -> #information {
                        <#field_type as Interaction<
                            #position,
                            #velocity,
                            #force,
                            #information
                        >>::get_interaction_information(
                            &self.#field_name
                        )
                    }

                    fn calculate_force_between(
                        &self,
                        own_pos: &#position,
                        own_vel: &#velocity,
                        ext_pos: &#position,
                        ext_vel: &#velocity,
                        ext_info: &#information,
                    ) -> Option<Result<#force, CalcError>> {
                        <#field_type as Interaction<
                            #position,
                            #velocity,
                            #force,
                            #information
                        >>::calculate_force_between(
                            &self.#field_name,
                            own_pos,
                            own_vel,
                            ext_pos,
                            ext_vel,
                            ext_info
                        )
                    }

                    fn is_neighbour(
                        &self,
                        own_pos: &#position,
                        ext_pos: &#position,
                        ext_inf: &#information
                    ) -> Result<bool, CalcError> {
                        <#field_type as Interaction<
                            #position,
                            #velocity,
                            #force,
                            #information
                        >>::is_neighbour(
                            &self.#field_name,
                            own_pos,
                            ext_pos,
                            ext_inf
                        )
                    }

                    fn react_to_neighbours(
                        &mut self,
                        neighbours: usize
                    ) -> Result<(), CalcError> {
                        <#field_type as Interaction<
                            #position,
                            #velocity,
                            #force,
                            #information
                        >>::react_to_neighbours(
                            &mut self.#field_name,
                            neighbours
                        )
                    }
                }
            };
            return TokenStream::from(res);
        }
        TokenStream::new()
    }

    fn implement_reactions(&self) -> TokenStream {
        let struct_name = &self.name;
        let struct_generics = &self.generics;

        if let Some(cellular_reactions_implemeneter) = &self.cellular_reactions {
            let field_name = &cellular_reactions_implemeneter.field_name;
            let field_type = &cellular_reactions_implemeneter.field_type;
            let concvecintracellular = &cellular_reactions_implemeneter.concvecintracellular;
            let concvecextracellular = &cellular_reactions_implemeneter.concvecextracellular;

            let res = quote! {
                #[automatically_derived]
                impl #struct_generics CellularReactions<
                    #concvecintracellular,
                    #concvecextracellular
                > for #struct_name #struct_generics {
                    fn get_intracellular(&self) -> #concvecintracellular {
                        <#field_type as CellularReactions<
                            #concvecintracellular,
                            #concvecextracellular
                        >>::get_intracellular(&self.#field_name)
                    }

                    fn set_intracellular(
                        &mut self,
                        concentration_vector: #concvecintracellular
                    ) {
                        <#field_type as CellularReactions<
                            #concvecintracellular,
                            #concvecextracellular
                        >>::set_intracellular(
                            &mut self.#field_name,
                            concentration_vector
                        );
                    }

                    fn calculate_intra_and_extracellular_reaction_increment(
                        &self,
                        internal_concentration_vector: &#concvecintracellular,
                        external_concentration_vector: &#concvecextracellular,
                    ) -> Result<(#concvecintracellular, #concvecextracellular), CalcError> {
                        <#field_type as CellularReactions<
                            #concvecintracellular,
                            #concvecextracellular
                        >>::calculate_intra_and_extracellular_reaction_increment(
                            &self.#field_name,
                            internal_concentration_vector,
                            external_concentration_vector
                        )
                    }
                }
            };
            return TokenStream::from(res);
        }
        TokenStream::new()
    }

    fn implement_extracellular_gradient(&self) -> TokenStream {
        let struct_name = &self.name;
        let struct_generics = &self.generics;

        if let Some(extracellular_gradient_implementer) = &self.extracellular_gradient {
            let field_type = &extracellular_gradient_implementer.field_type;

            let extracellular_gradient = &extracellular_gradient_implementer.extracellular_gradient;
            let res = quote! {
                #[automatically_derived]
                impl InteractionExtracellularGradient<
                    #struct_name #struct_generics,
                    #extracellular_gradient
                > for #struct_name #struct_generics {
                    fn sense_gradient(
                        cell: &mut #struct_name #struct_generics,
                        gradient: &#extracellular_gradient,
                    ) -> Result<(), CalcError> {
                        <#field_type as InteractionExtracellularGradient<
                            #struct_name #struct_generics,
                            #extracellular_gradient
                        >>::sense_gradient(cell, gradient)
                    }
                }
            };
            return TokenStream::from(res);
        }
        TokenStream::new()
    }

    fn implement_volume(&self) -> TokenStream {
        let struct_name = &self.name;
        let struct_generics = &self.generics;

        if let Some(volume_implementer) = &self.volume {
            let field_type = &volume_implementer.field_type;
            let field_name = &volume_implementer.field_name;

            let res = quote! {
                #[automatically_derived]
                impl Volume for #struct_name #struct_generics {
                    fn get_volume(&self) -> f64 {
                        <#field_type as Volume>::get_volume(
                            &self.#field_name
                        )
                    }
                }
            };
            return TokenStream::from(res);
        }
        TokenStream::new()
    }
}

/// Derive [concepts](cellular_raza_concepts)
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
/// | `Cycle`                   | -                                                                     |
/// | `Mechanics`               | `(Pos, Vel, For)`                                                     |
/// | `Interaction`             | `(Pos, Vel, For, Inf=())`                                             |
/// | `CellularReactions`       | `(ConcVecIntracellular, ConcVecExtracellular=ConcVecIntracellular)`   |
/// | `ExtracellularGradient`   | `(ConcGradientExtracellular)`                                         |
/// | `Volume`                  | `(F=f64)`                                                         |
///
/// For a description of these type arguments see [cellular_raza_concepts].
#[proc_macro_derive(
    CellAgent,
    attributes(
        Cycle,
        Mechanics,
        Interaction,
        CellularReactions,
        ExtracellularGradient,
        Volume,
    )
)]
pub fn derive_cell_agent(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let agent_parsed = syn::parse_macro_input!(input as AgentParser);
    let agent = AgentImplementer::from(agent_parsed);

    let mut res = TokenStream::new();
    res.extend(agent.implement_cycle());
    res.extend(agent.implement_mechanics());
    res.extend(agent.implement_reactions());
    res.extend(agent.implement_interaction());
    res.extend(agent.implement_extracellular_gradient());
    res.extend(agent.implement_volume());

    wrap(res)
}
