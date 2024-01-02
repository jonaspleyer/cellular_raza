use proc_macro::TokenStream;
use quote::quote;
use syn::parse::ParseStream;

// ##################################### PARSING #####################################
#[allow(unused)]
pub struct AuxStorageParser {
    attrs: Vec<syn::Attribute>,
    vis: syn::Visibility,
    struct_token: syn::Token![struct],
    name: syn::Ident,
    generics: syn::Generics,
    aspects: AspectFields,
}

impl syn::parse::Parse for AuxStorageParser {
    fn parse(input: ParseStream) -> syn::Result<Self> {
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

struct AspectFields {
    #[allow(unused)]
    brace_token: syn::token::Brace,
    aspect_fields: syn::punctuated::Punctuated<AspectField, syn::token::Comma>,
}

impl syn::parse::Parse for AspectFields {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        Ok(Self {
            brace_token: syn::braced!(content in input),
            aspect_fields: content.call(
                syn::punctuated::Punctuated::<AspectField, syn::token::Comma>::parse_terminated,
            )?,
        })
    }
}

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

impl Aspect {
    fn from_attribute(attr: &syn::Attribute) -> syn::Result<Option<Self>> {
        let path = attr.meta.path().get_ident();
        let cmp = |c: &str| path.is_some_and(|p| p.to_string() == c);

        let s = &attr.meta;
        let stream: TokenStream = quote!(#s).into();

        if cmp("UpdateMechanics") {
            let parsed: UpdateMechanicsParser = syn::parse(stream)?;
            return Ok(Some(Aspect::UpdateMechanics(parsed)));
        }

        if cmp("UpdateCycle") {
            let parsed: UpdateCycleParser = syn::parse(stream)?;
            return Ok(Some(Aspect::UpdateCycle(parsed)));
        }

        /* if cmp("UpdateInteraction") {
            let parsed: InteractionParser = syn::parse(stream)?;
            return Ok(Some(Aspect::UpdateInteraction(parsed)));
        }

        if cmp("UpdateReactions") {
            let parsed: ReactionsParser = syn::parse(stream)?;
            return Ok(Some(Aspect::UpdateReactions(parsed)));
        }*/

        Ok(None)
    }
}

enum Aspect {
    UpdateMechanics(UpdateMechanicsParser),
    UpdateCycle(UpdateCycleParser),
    // UpdateInteraction(InteractionParser),
    // UpdateReactions(ReactionsParser),
}

// --------------------------------- UPDATE-MECHANICS --------------------------------
struct UpdateMechanicsParser {
    position: syn::GenericParam,
    _comma_1: syn::token::Comma,
    velocity: syn::GenericParam,
    _comma_2: syn::token::Comma,
    force: syn::GenericParam,
    _comma_3: syn::token::Comma,
    float_type: syn::GenericParam,
    _comma_5: Option<syn::token::Comma>,
    n_saves: syn::GenericParam,
    _comma_4: syn::token::Comma,
}

impl syn::parse::Parse for UpdateMechanicsParser {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let _update_mechanics: syn::Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        Ok(Self {
            position: content.parse()?,
            _comma_1: content.parse()?,
            velocity: content.parse()?,
            _comma_2: content.parse()?,
            force: content.parse()?,
            _comma_3: content.parse()?,
            float_type: content.parse()?,
            _comma_4: content.parse()?,
            n_saves: content.parse()?,
            _comma_5: content.parse()?,
        })
    }
}

// ----------------------------------- UPDATE-CYCLE ----------------------------------
struct UpdateCycleParser;

impl syn::parse::Parse for UpdateCycleParser {
    #[allow(unused)]
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let _update_cycle: syn::Ident = input.parse()?;
        Ok(Self)
    }
}

// -------------------------------- UPDATE-INTERACTION -------------------------------
struct UpdateInteractionParser {}

impl syn::parse::Parse for UpdateInteractionParser {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {})
    }
}

// --------------------------------- UPDATE-REACTIONS --------------------------------
struct UpdateReactionsParser {}

impl syn::parse::Parse for UpdateReactionsParser {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {})
    }
}

// ################################### CONVERSION ####################################
impl From<AuxStorageParser> for AuxStorageImplementer {
    fn from(value: AuxStorageParser) -> Self {
        let mut update_cycle = None;
        let mut update_mechanics = None;
        // let mut update_interaction = None;
        // let mut update_reactions = None;

        value
            .aspects
            .aspect_fields
            .into_iter()
            .for_each(|aspect_field| {
                aspect_field
                    .aspects
                    .into_iter()
                    .for_each(|aspect| match aspect {
                        Aspect::UpdateCycle(_) => {
                            update_cycle = Some(UpdateCycleImplementer {
                                field_name: aspect_field.field.ident.clone(),
                                field_type: aspect_field.field.ty.clone(),
                            })
                        }
                        #[allow(unused)]
                        Aspect::UpdateMechanics(p) => {
                            update_mechanics = Some(UpdateMechanicsImplementer {
                                position: p.position,
                                velocity: p.velocity,
                                force: p.force,
                                n_saves: p.n_saves,
                                field_name: aspect_field.field.ident.clone(),
                                field_type: aspect_field.field.ty.clone(),
                            })
                        } /* Aspect::UpdateInteraction(p) => {
                              update_interaction = Some(InteractionImplementer {
                                  position: p.position,
                                  velocity: p.velocity,
                                  force: p.force,
                                  information: p.information,
                                  field_type: aspect_field.field.ty.clone(),
                                  field_name: aspect_field.field.ident.clone(),
                              })
                          }
                          Aspect::UpdateReactions(p) => {
                              update_reactions = Some(ReactionsImplementer {
                                  concvecintracellular: p.concvecintracellular,
                                  concvecextracellular: p.concvecextracellular,
                                  field_type: aspect_field.field.ty.clone(),
                                  field_name: aspect_field.field.ident.clone(),
                              })
                          }*/
                    })
            });

        Self {
            name: value.name,
            generics: value.generics,
            update_cycle: update_cycle,
            update_mechanics: update_mechanics,
            /*update_interaction: update_interaction,
            update_reactions: update_reactions,*/
        }
    }
}

// ################################## IMPLEMENTING ###################################
#[allow(unused)]
pub struct AuxStorageImplementer {
    name: syn::Ident,
    generics: syn::Generics,
    update_mechanics: Option<UpdateMechanicsImplementer>,
    update_cycle: Option<UpdateCycleImplementer>,
}

fn wrap_pre_flags(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    quote!(
        #[allow(unused)]
        #[doc(hidden)]
        #[automatically_derived]
        #input
    )
}

// --------------------------------- UPDATE-MECHANICS --------------------------------
struct UpdateMechanicsImplementer {
    position: syn::GenericParam,
    velocity: syn::GenericParam,
    force: syn::GenericParam,
    n_saves: syn::GenericParam,
    float_type: syn::GenericParam,
    field_name: Option<syn::Ident>,
    field_type: syn::Type,
}

impl AuxStorageImplementer {
    fn implement_update_mechanics(&self) -> TokenStream {
        if let Some(update_mechanics) = &self.update_mechanics {
            let position = &update_mechanics.position;
            let velocity = &update_mechanics.velocity;
            let force = &update_mechanics.force;
            let n_saves = &update_mechanics.n_saves;
            let float_type = &update_mechanics.float_type;

            let field_generics = quote!(#position, #velocity, #force, #float_type, #n_saves);

            let struct_name = &self.name;
            let (struct_impl_generics, struct_ty_generics, struct_where_clause) =
                &self.generics.split_for_impl();

            let field_name = &update_mechanics.field_name;
            let field_type = &update_mechanics.field_type;

            let new_stream = wrap_pre_flags(quote!(
                impl #struct_impl_generics UpdateMechanics<#field_generics> for #struct_name #struct_ty_generics #struct_where_clause
                where
                    #force: Clone + core::ops::AddAssign<#force> + num::Zero,
                    #float_type: Clone,
                {
                    fn set_last_position(&mut self, pos: #position) {
                        <#field_type as UpdateMechanics<#field_generics>>::set_last_position(&mut self.#field_name, pos)
                    }
                    fn previous_positions(&self) -> std::collections::vec_deque::Iter<#position> {
                        <#field_type as UpdateMechanics<#field_generics>>::previous_positions(&self.#field_name)
                    }
                    fn set_last_velocity(&mut self, vel: #velocity) {
                        <#field_type as UpdateMechanics<#field_generics>>::set_last_velocity(&mut self.#field_name, vel)
                    }
                    fn previous_velocities(&self) -> std::collections::vec_deque::Iter<#velocity> {
                        <#field_type as UpdateMechanics<#field_generics>>::previous_velocities(&self.#field_name)
                    }
                    fn add_force(&mut self, force: #force) {
                        <#field_type as UpdateMechanics<#field_generics>>::add_force(&mut self.#field_name, force);
                    }
                    fn get_current_force(&self) -> #force {
                        <#field_type as UpdateMechanics<#field_generics>>::get_current_force(&self.#field_name)
                    }
                    fn clear_forces(&mut self) {
                        <#field_type as UpdateMechanics<#field_generics>>::clear_forces(&mut self.#field_name)
                    }

                    fn get_next_random_update(&self) -> Option<#float_type> {
                        <#field_type as UpdateMechanics<#field_generics>>::get_next_random_update(&self.#field_name)
                    }

                    fn set_next_random_update(&mut self, next: Option<#float_type>) {
                        <#field_type as UpdateMechanics<#field_generics>>::set_next_random_update(&mut self.#field_name, next)
                    }
                }
            ));
            return TokenStream::from(new_stream);
        }

        TokenStream::new()
    }
}

// ----------------------------------- UPDATE-CYCLE ----------------------------------
struct UpdateCycleImplementer {
    field_name: Option<syn::Ident>,
    field_type: syn::Type,
}

impl AuxStorageImplementer {
    fn implement_update_cycle(&self) -> TokenStream {
        if let Some(update_cycle) = &self.update_cycle {
            let struct_name = &self.name;
            let (impl_generics, ty_generics, where_clause) = &self.generics.split_for_impl();

            let field_name = &update_cycle.field_name;
            let field_type = &update_cycle.field_type;

            let new_stream = wrap_pre_flags(quote!(
                impl #impl_generics UpdateCycle for #struct_name #ty_generics #where_clause {
                    fn set_cycle_events(&mut self, events: Vec<CycleEvent>) {
                        <#field_type as UpdateCycle>::set_cycle_events(&mut self.#field_name, events)
                    }

                    fn get_cycle_events(&self) -> Vec<CycleEvent> {
                        <#field_type as UpdateCycle>::get_cycle_events(&self.#field_name)
                    }

                    fn add_cycle_event(&mut self, event: CycleEvent) {
                        <#field_type as UpdateCycle>::add_cycle_event(&mut self.#field_name, event)
                    }
                }
            ));
            return TokenStream::from(new_stream);
        }
        TokenStream::new()
    }
}

// --------------------------------- UPDATE-REACTIONS --------------------------------
struct ReactionsImplementer {}

impl AuxStorageImplementer {
    fn implement_update_reactions(&self) -> TokenStream {
        TokenStream::new()
    }
}

// -------------------------------- UPDATE-INTERACTION -------------------------------
struct InteractionImplementer {}

impl AuxStorageImplementer {
    fn implement_update_interaction(&self) -> TokenStream {
        TokenStream::new()
    }
}

pub fn derive_aux_storage(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let aux_storage_parsed = syn::parse_macro_input!(input as AuxStorageParser);
    let aux_storage = AuxStorageImplementer::from(aux_storage_parsed);

    let mut res = TokenStream::new();
    res.extend(aux_storage.implement_update_cycle());
    res.extend(aux_storage.implement_update_mechanics());
    res.extend(aux_storage.implement_update_reactions());
    res.extend(aux_storage.implement_update_interaction());

    res
}
