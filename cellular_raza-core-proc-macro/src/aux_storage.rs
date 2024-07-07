//! The main two macros are [derive_aux_storage] and [construct_aux_storage] which allow to
//! derive individual aspects of the the update traits in `cellular_raza_core`

use proc_macro::TokenStream;
use quote::quote;
use syn::spanned::Spanned;

use super::simulation_aspects::*;

// ##################################### PARSING #####################################
struct AuxStorageParser {
    name: syn::Ident,
    generics: syn::Generics,
    aspects: AspectFields,
    core_path: Option<syn::Path>,
}

impl syn::parse::Parse for AuxStorageParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let item_struct: syn::ItemStruct = input.parse()?;
        let core_path = core_path_from_attributes(&item_struct.attrs, item_struct.span())?;

        Ok(Self {
            name: item_struct.ident,
            generics: item_struct.generics,
            aspects: AspectFields::from_fields(item_struct.fields)?,
            core_path,
        })
    }
}

fn core_path_from_attributes(
    attrs: &Vec<syn::Attribute>,
    span: proc_macro2::Span,
) -> syn::Result<Option<syn::Path>> {
    let mut candidates = attrs
        .iter()
        .filter(|attr| {
            attr.meta
                .path()
                .get_ident()
                .is_some_and(|ident| ident == "AuxStorageCorePath")
        })
        .map(|attr| {
            let s = &attr.meta;
            let stream = quote!(#s);
            let parsed: super::from_map::CorePathParser = syn::parse2(stream)?;
            Ok(parsed)
        })
        .collect::<syn::Result<Vec<_>>>()?;
    if candidates.len() > 1 {
        return Err(syn::Error::new(
            span,
            "Expected at most one #[FromMapCorePath(..)] attribute",
        ));
    }
    if candidates.len() == 1 {
        return Ok(Some(candidates.remove(0).core_path));
    }
    Ok(None)
}

struct AspectFields {
    aspect_fields: Vec<AspectField>,
}

impl AspectFields {
    fn from_fields(fields: syn::Fields) -> syn::Result<Self> {
        if let syn::Fields::Named(named_fields) = fields {
            Ok(Self {
                aspect_fields: named_fields
                    .named
                    .into_iter()
                    .map(|field| AspectField::from_field(field))
                    .collect::<Result<Vec<_>, syn::Error>>()?,
            })
        } else {
            Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "Could not parse struct. Use named fields for struct definition syntax!",
            ))
        }
    }
}

struct AspectField {
    aspects: Vec<Aspect>,
    field: syn::Field,
}

impl AspectField {
    fn from_field(field: syn::Field) -> syn::Result<Self> {
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
        let stream: proc_macro::TokenStream = quote!(#s).into();

        if cmp("UpdateMechanics") {
            let parsed: UpdateMechanicsParser = syn::parse(stream)?;
            return Ok(Some(Aspect::UpdateMechanics(parsed)));
        }

        if cmp("UpdateCycle") {
            let parsed: UpdateCycleParser = syn::parse(stream)?;
            return Ok(Some(Aspect::UpdateCycle(parsed)));
        }

        if cmp("UpdateInteraction") {
            let parsed: UpdateInteractionParser = syn::parse(stream)?;
            return Ok(Some(Aspect::UpdateInteraction(parsed)));
        }

        if cmp("UpdateReactions") {
            let parsed: UpdateReactionsParser = syn::parse(stream)?;
            return Ok(Some(Aspect::UpdateReactions(parsed)));
        }

        Ok(None)
    }
}

enum Aspect {
    UpdateMechanics(UpdateMechanicsParser),
    UpdateCycle(UpdateCycleParser),
    UpdateInteraction(UpdateInteractionParser),
    UpdateReactions(UpdateReactionsParser),
}

// --------------------------------- UPDATE-MECHANICS --------------------------------
struct UpdateMechanicsParser {
    position: syn::GenericParam,
    _comma_1: syn::token::Comma,
    velocity: syn::GenericParam,
    _comma_2: syn::token::Comma,
    force: syn::GenericParam,
    _comma_3: syn::token::Comma,
    n_saves: syn::GenericParam,
}

impl syn::parse::Parse for UpdateMechanicsParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
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
            n_saves: content.parse()?,
        })
    }
}

// ----------------------------------- UPDATE-CYCLE ----------------------------------
struct UpdateCycleParser;

impl syn::parse::Parse for UpdateCycleParser {
    #[allow(unused)]
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _update_cycle: syn::Ident = input.parse()?;
        Ok(Self)
    }
}

// -------------------------------- UPDATE-INTERACTION -------------------------------
struct UpdateInteractionParser;

impl syn::parse::Parse for UpdateInteractionParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _update_interaction: syn::Ident = input.parse()?;
        Ok(Self)
    }
}

// --------------------------------- UPDATE-REACTIONS --------------------------------
struct UpdateReactionsParser {
    concentration: syn::GenericParam,
}

impl syn::parse::Parse for UpdateReactionsParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _update_reactions: syn::Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        Ok(Self {
            concentration: content.parse()?,
        })
    }
}

// ################################### CONVERSION ####################################
impl From<AuxStorageParser> for AuxStorageImplementer {
    fn from(value: AuxStorageParser) -> Self {
        let mut update_cycle = None;
        let mut update_mechanics = None;
        let mut update_interaction = None;
        let mut update_reactions = None;

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
                        }
                        Aspect::UpdateInteraction(_) => {
                            update_interaction = Some(UpdateInteractionImplementer {
                                field_type: aspect_field.field.ty.clone(),
                                field_name: aspect_field.field.ident.clone(),
                            })
                        }
                        Aspect::UpdateReactions(p) => {
                            update_reactions = Some(UpdateReactionsImplementer {
                                concentration: p.concentration,
                                field_type: aspect_field.field.ty.clone(),
                                field_name: aspect_field.field.ident.clone(),
                            })
                        }
                    })
            });

        Self {
            name: value.name,
            generics: value.generics,
            update_cycle,
            update_mechanics,
            update_interaction,
            update_reactions,
            core_path: value.core_path,
        }
    }
}

// ################################## IMPLEMENTING ###################################
#[allow(unused)]
struct AuxStorageImplementer {
    name: syn::Ident,
    generics: syn::Generics,
    update_mechanics: Option<UpdateMechanicsImplementer>,
    update_cycle: Option<UpdateCycleImplementer>,
    update_interaction: Option<UpdateInteractionImplementer>,
    update_reactions: Option<UpdateReactionsImplementer>,
    core_path: Option<syn::Path>,
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

            let backend_path = match &self.core_path {
                Some(p) => quote!(#p ::backend::chili::),
                None => quote!(),
            };

            let field_generics = quote!(#position, #velocity, #force, #n_saves);

            let struct_name = &self.name;
            let (struct_impl_generics, struct_ty_generics, struct_where_clause) =
                &self.generics.split_for_impl();
            let where_clause = match struct_where_clause {
                Some(s_where) => {
                    let pred = s_where.predicates.iter();
                    quote!(
                        where
                        #(#pred,)*
                        #force: Clone + core::ops::AddAssign<#force> + num::Zero,
                    )
                }
                None => quote!(
                    where
                    #force: Clone + core::ops::AddAssign<#force> + num::Zero,
                ),
            };

            let field_name = &update_mechanics.field_name;
            let field_type = &update_mechanics.field_type;

            let new_stream = wrap_pre_flags(quote!(
                impl #struct_impl_generics #backend_path UpdateMechanics<#field_generics>
                for #struct_name #struct_ty_generics #where_clause
                {
                    fn set_last_position(&mut self, pos: #position) {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::set_last_position(&mut self.#field_name, pos)
                    }
                    fn previous_positions<'a>(
                        &'a self
                    ) -> #backend_path FixedSizeRingBufferIter<'a, #position, #n_saves> {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::previous_positions(&self.#field_name)
                    }
                    fn set_last_velocity(&mut self, vel: #velocity) {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::set_last_velocity(&mut self.#field_name, vel)
                    }
                    fn previous_velocities<'a>(
                        &'a self
                    ) -> #backend_path FixedSizeRingBufferIter<'a, #velocity, #n_saves> {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::previous_velocities(&self.#field_name)
                    }
                    fn n_previous_values(&self) -> usize {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::n_previous_values(&self.#field_name)
                    }
                    fn add_force(&mut self, force: #force) {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::add_force(&mut self.#field_name, force);
                    }
                    fn get_current_force(&self) -> #force {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::get_current_force(&self.#field_name)
                    }
                    fn clear_forces(&mut self) {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::clear_forces(&mut self.#field_name)
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

            let backend_path = match &self.core_path {
                Some(p) => quote!(#p ::backend::chili::),
                None => quote!(),
            };

            let field_name = &update_cycle.field_name;
            let field_type = &update_cycle.field_type;

            let new_stream = wrap_pre_flags(quote!(
                impl #impl_generics #backend_path UpdateCycle for #struct_name #ty_generics #where_clause {
                    fn set_cycle_events(&mut self, events: Vec<#backend_path CycleEvent>) {
                        <#field_type as #backend_path UpdateCycle>::set_cycle_events(&mut self.#field_name, events)
                    }

                    fn get_cycle_events(&self) -> &Vec<#backend_path CycleEvent> {
                        <#field_type as #backend_path UpdateCycle>::get_cycle_events(&self.#field_name)
                    }

                    fn drain_cycle_events(&mut self) -> std::vec::Drain<#backend_path CycleEvent> {
                        <#field_type as #backend_path UpdateCycle>::drain_cycle_events(&mut self.#field_name)
                    }

                    fn add_cycle_event(&mut self, event: #backend_path CycleEvent) {
                        <#field_type as #backend_path UpdateCycle>::add_cycle_event(&mut self.#field_name, event)
                    }
                }
            ));
            return TokenStream::from(new_stream);
        }
        TokenStream::new()
    }
}

// --------------------------------- UPDATE-REACTIONS --------------------------------
struct UpdateReactionsImplementer {
    concentration: syn::GenericParam,
    field_name: Option<syn::Ident>,
    field_type: syn::Type,
}

impl AuxStorageImplementer {
    fn implement_update_reactions(&self) -> TokenStream {
        if let Some(update_reactions) = &self.update_reactions {
            let concentration = &update_reactions.concentration;

            let struct_name = &self.name;
            let (impl_generics, ty_generics, where_clause) = &self.generics.split_for_impl();

            let backend_path = match &self.core_path {
                Some(p) => quote!(#p ::backend::chili::),
                None => quote!(),
            };

            let field_name = &update_reactions.field_name;
            let field_type = &update_reactions.field_type;

            let where_clause = match where_clause {
                Some(s_where) => {
                    let pred = s_where.predicates.iter();
                    quote!(
                        where
                        #(#pred,)*
                        #concentration: Clone,
                        #concentration: core::ops::Add<#concentration, Output=#concentration>,
                    )
                }
                None => quote!(
                    where
                    #concentration: Clone,
                    #concentration: core::ops::Add<#concentration, Output=#concentration>,
                ),
            };

            let new_stream = wrap_pre_flags(quote!(
                impl #impl_generics #backend_path UpdateReactions<#concentration> for #struct_name #ty_generics #where_clause {
                    fn set_conc(&mut self, conc: #concentration) {
                        <#field_type as #backend_path UpdateReactions<#concentration>>::set_conc(
                            &mut self.#field_name,
                            conc
                        )
                    }

                    fn get_conc(&self) -> #concentration {
                        <#field_type as #backend_path UpdateReactions<#concentration>>::get_conc(
                            &self.#field_name
                        )
                    }

                    fn incr_conc(&mut self, incr: #concentration) {
                        <#field_type as #backend_path UpdateReactions<#concentration>>::incr_conc(
                            &mut self.#field_name,
                            incr
                        )
                    }
                }
            ));
            return TokenStream::from(new_stream);
        }
        TokenStream::new()
    }
}

// -------------------------------- UPDATE-INTERACTION -------------------------------
struct UpdateInteractionImplementer {
    field_name: Option<syn::Ident>,
    field_type: syn::Type,
}

impl AuxStorageImplementer {
    fn implement_update_interaction(&self) -> TokenStream {
        if let Some(update_interaction) = &self.update_interaction {
            let field_name = &update_interaction.field_name;
            let field_type = &update_interaction.field_type;

            let struct_name = &self.name;
            let (impl_generics, ty_generics, where_clause) = &self.generics.split_for_impl();

            let backend_path = match &self.core_path {
                Some(p) => quote!(#p ::backend::chili::),
                None => quote!(),
            };

            let new_stream = wrap_pre_flags(quote!(
                impl #impl_generics #backend_path UpdateInteraction for #struct_name #ty_generics #where_clause {
                    fn get_current_neighbours(&self) -> usize {
                        <#field_type as #backend_path UpdateInteraction>::get_current_neighbours(
                            &self.#field_name
                        )
                    }

                    fn incr_current_neighbours(&mut self, neighbours: usize) {
                        <#field_type as #backend_path UpdateInteraction>::incr_current_neighbours(
                            &mut self.#field_name,
                            neighbours
                        )
                    }

                    fn set_current_neighbours(&mut self, neighbours: usize) {
                        <#field_type as #backend_path UpdateInteraction>::set_current_neighbours(
                            &mut self.#field_name,
                            neighbours
                        )
                    }
                }
            ));
            return TokenStream::from(new_stream);
        }
        TokenStream::new()
    }
}

// ##################################### DERIVE ######################################
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

// #################################### CONSTRUCT ####################################
pub struct Arguments {
    pub name_def: NameDefinition,
    _comma: syn::Token![,],
    pub simulation_aspects: SimulationAspects,
    _comma_2: syn::Token![,],
    pub core_path: CorePath,
}

impl syn::parse::Parse for Arguments {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            name_def: input.parse()?,
            _comma: input.parse()?,
            simulation_aspects: input.parse()?,
            _comma_2: input.parse()?,
            core_path: input.parse()?,
        })
    }
}

pub struct Builder {
    pub struct_name: syn::Ident,
    pub core_path: syn::Path,
    pub aspects: SimulationAspects,
}

impl From<Arguments> for Builder {
    fn from(arguments: Arguments) -> Self {
        Self {
            struct_name: arguments.name_def.struct_name,
            core_path: arguments.core_path.path,
            aspects: arguments.simulation_aspects,
        }
    }
}

impl Builder {
    pub fn build_aux_storage(self) -> proc_macro2::TokenStream {
        let struct_name = self.struct_name;
        let p = self.core_path;
        let (core_path, backend_path) = (quote!(#p), quote!(#p ::backend::chili::));
        let mut generics = vec![];
        let mut fields = vec![];

        use SimulationAspect::*;
        if self.aspects.contains(&Cycle) {
            fields.push(quote!(
                #[UpdateCycle]
                cycle: #backend_path AuxStorageCycle,
            ));
        }

        if self.aspects.contains(&Mechanics) {
            generics.extend(vec![quote!(Pos), quote!(Vel), quote!(For), quote!(const NMec: usize)]);
            fields.push(quote!(
                #[UpdateMechanics(Pos, Vel, For, NMec)]
                mechanics: #backend_path AuxStorageMechanics<Pos, Vel, For, NMec>,
            ));
        }

        /* if self.aspects.contains(&Reactions) {
            generics.extend(vec![quote!(Ri)]);
            fields.push(quote!(
                #[UpdateReactions(Ri)]
                reactions: #backend_path AuxStorageReactions<Ri>,
            ));
        }*/

        if self.aspects.contains(&Interaction) {
            fields.push(quote!(
                #[UpdateInteraction]
                interaction: #backend_path AuxStorageInteraction,
            ));
        }

        let stream = quote!(
            #[allow(non_camel_case_types)]
            #[doc(hidden)]
            #[derive(Clone, Default, Deserialize, Serialize)]
            #[derive(#core_path ::backend::chili::AuxStorage)]
            #[AuxStorageCorePath(#core_path)]
            pub struct #struct_name<#(#generics),*> {
                #(#fields)*
            }
        );
        stream
    }
}

/// Define a AuxStorage struct that
pub fn construct_aux_storage(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let builder: Builder = syn::parse_macro_input!(input as Arguments).into();
    proc_macro::TokenStream::from(builder.build_aux_storage())
}
