//! The main two macros are [derive_aux_storage] and [construct_aux_storage] which allow to
//! derive individual aspects of the the update traits in `cellular_raza_core`

use itertools::Itertools;
use proc_macro::TokenStream;
use quote::quote;
use syn::spanned::Spanned;

use super::run_sim::*;
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

        if cmp("UpdateReactionsContact") {
            let parsed: UpdateReactionsContactParser = syn::parse(stream)?;
            return Ok(Some(Aspect::UpdateReactionsContact(parsed)));
        }

        Ok(None)
    }
}

enum Aspect {
    UpdateMechanics(UpdateMechanicsParser),
    UpdateCycle(UpdateCycleParser),
    UpdateInteraction(UpdateInteractionParser),
    UpdateReactions(UpdateReactionsParser),
    UpdateReactionsContact(UpdateReactionsContactParser),
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
    rintra: syn::GenericParam,
}

impl syn::parse::Parse for UpdateReactionsParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _update_reactions: syn::Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        Ok(Self {
            rintra: content.parse()?,
        })
    }
}

// --------------------------------- UPDATE-REACTIONS --------------------------------
struct UpdateReactionsContactParser {
    rintra: syn::GenericParam,
    _comma_1: syn::token::Comma,
    n_saves: syn::GenericParam,
}

impl syn::parse::Parse for UpdateReactionsContactParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _update_reactions_contact: syn::Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        Ok(Self {
            rintra: content.parse()?,
            _comma_1: content.parse()?,
            n_saves: content.parse()?,
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
        let mut update_reactions_contact = None;

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
                                rintra: p.rintra,
                                field_type: aspect_field.field.ty.clone(),
                                field_name: aspect_field.field.ident.clone(),
                            })
                        }
                        Aspect::UpdateReactionsContact(p) => {
                            update_reactions_contact = Some(UpdateReactionsContactImplementer {
                                rintra: p.rintra,
                                n_saves: p.n_saves,
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
            update_reactions_contact,
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
    update_reactions_contact: Option<UpdateReactionsContactImplementer>,
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
                        #force: Clone + core::ops::AddAssign<#force>,
                    )
                }
                None => quote!(
                    where
                    #force: Clone + core::ops::AddAssign<#force>,
                ),
            };

            let field_name = &update_mechanics.field_name;
            let field_type = &update_mechanics.field_type;

            let new_stream = wrap_pre_flags(quote!(
                impl #struct_impl_generics #backend_path UpdateMechanics<#field_generics>
                for #struct_name #struct_ty_generics #where_clause
                {
                    #[inline]
                    fn set_last_position(&mut self, pos: #position) {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::set_last_position(&mut self.#field_name, pos)
                    }
                    #[inline]
                    fn previous_positions<'a>(
                        &'a self
                    ) -> #backend_path RingBufferIterRef<'a, #position, #n_saves> {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::previous_positions(&self.#field_name)
                    }
                    #[inline]
                    fn set_last_velocity(&mut self, vel: #velocity) {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::set_last_velocity(&mut self.#field_name, vel)
                    }
                    #[inline]
                    fn previous_velocities<'a>(
                        &'a self
                    ) -> #backend_path RingBufferIterRef<'a, #velocity, #n_saves> {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::previous_velocities(&self.#field_name)
                    }
                    #[inline]
                    fn n_previous_values(&self) -> usize {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::n_previous_values(&self.#field_name)
                    }
                    #[inline]
                    fn add_force(&mut self, force: #force) {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::add_force(&mut self.#field_name, force);
                    }
                    #[inline]
                    fn get_current_force_and_reset(&mut self) -> #force {
                        <#field_type as #backend_path UpdateMechanics<#field_generics>>
                            ::get_current_force_and_reset(&mut self.#field_name)
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
                impl #impl_generics #backend_path UpdateCycle
                for #struct_name #ty_generics #where_clause {
                    #[inline]
                    fn set_cycle_events(&mut self, events: Vec<#backend_path CycleEvent>) {
                        <#field_type as #backend_path UpdateCycle>::set_cycle_events(
                            &mut self.#field_name,
                            events
                        )
                    }

                    #[inline]
                    fn get_cycle_events(&self) -> &Vec<#backend_path CycleEvent> {
                        <#field_type as #backend_path UpdateCycle>::get_cycle_events(
                            &self.#field_name
                        )
                    }

                    #[inline]
                    fn drain_cycle_events(&mut self) -> std::vec::Drain<#backend_path CycleEvent> {
                        <#field_type as #backend_path UpdateCycle>::drain_cycle_events(
                            &mut self.#field_name
                        )
                    }

                    #[inline]
                    fn add_cycle_event(&mut self, event: #backend_path CycleEvent) {
                        <#field_type as #backend_path UpdateCycle>::add_cycle_event(
                            &mut self.#field_name,
                            event
                        )
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
    rintra: syn::GenericParam,
    field_name: Option<syn::Ident>,
    field_type: syn::Type,
}

impl AuxStorageImplementer {
    fn implement_update_reactions(&self) -> TokenStream {
        if let Some(update_reactions) = &self.update_reactions {
            let rintra = &update_reactions.rintra;

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
                            #field_type: #backend_path UpdateReactions<#rintra>,
                    )
                }
                None => quote!(
                    where
                        #field_type: #backend_path UpdateReactions<#rintra>,
                ),
            };

            let new_stream = wrap_pre_flags(quote!(
                impl #impl_generics #backend_path UpdateReactions<#rintra>
                for #struct_name #ty_generics #where_clause {
                    #[inline]
                    fn set_conc(&mut self, conc: #rintra) {
                        <#field_type as #backend_path UpdateReactions<#rintra>>::set_conc(
                            &mut self.#field_name,
                            conc
                        )
                    }

                    #[inline]
                    fn get_conc(&self) -> #rintra {
                        <#field_type as #backend_path UpdateReactions<#rintra>>::get_conc(
                            &self.#field_name
                        )
                    }

                    #[inline]
                    fn incr_conc(&mut self, incr: #rintra) {
                        <#field_type as #backend_path UpdateReactions<#rintra>>::incr_conc(
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

// ----------------------------- UPDATE-REACTIONS-CONTACT ----------------------------
struct UpdateReactionsContactImplementer {
    rintra: syn::GenericParam,
    n_saves: syn::GenericParam,
    field_name: Option<syn::Ident>,
    field_type: syn::Type,
}

impl AuxStorageImplementer {
    fn implement_update_reactions_contact(&self) -> TokenStream {
        if let Some(update_reactions_contact) = &self.update_reactions_contact {
            let rintra = &update_reactions_contact.rintra;
            let nsaves = &update_reactions_contact.n_saves;

            let struct_name = &self.name;
            let (impl_generics, ty_generics, where_clause) = &self.generics.split_for_impl();

            let backend_path = match &self.core_path {
                Some(p) => quote!(#p ::backend::chili::),
                None => quote!(),
            };

            let field_name = &update_reactions_contact.field_name;
            let field_type = &update_reactions_contact.field_type;

            let where_clause = match where_clause {
                Some(s_where) => {
                    let pred = s_where.predicates.iter();
                    quote!(
                        where
                            #(#pred,)*
                            #field_type: #backend_path UpdateReactionsContact<#rintra, #nsaves>,
                    )
                }
                None => quote!(
                    where
                        #field_type: #backend_path UpdateReactionsContact<#rintra, #nsaves>,
                ),
            };

            let new_stream = wrap_pre_flags(quote!(
                    impl #impl_generics #backend_path UpdateReactionsContact<#rintra, #nsaves>
                    for #struct_name #ty_generics #where_clause {
                        #[inline]
                        fn get_current_increment(&self) -> Ri {
                            <#field_type as #backend_path UpdateReactionsContact<#rintra, #nsaves>>
                                ::get_current_increment(&self.#field_name)
                        }

                        #[inline]
                        fn incr_current_increment(&mut self, increment: Ri) {
                            <#field_type as #backend_path UpdateReactionsContact<#rintra, #nsaves>>
                                ::incr_current_increment(&mut self.#field_name, increment)

                        }

                        #[inline]
                        fn set_current_increment(&mut self, new_increment: Ri) {
                            <#field_type as #backend_path UpdateReactionsContact<#rintra, #nsaves>>
                                ::set_current_increment(&mut self.#field_name, new_increment)
                        }

                        #[inline]
                        fn previous_increments<'a>(
                            &'a self
                        ) -> #backend_path RingBufferIterRef<'a, #rintra, #nsaves> {
                            <#field_type as #backend_path UpdateReactionsContact<#rintra, #nsaves>>
                                ::previous_increments(&self.#field_name)
                        }

                        #[inline]
                        fn set_last_increment(&mut self, increment: #rintra) {
                            <#field_type as #backend_path UpdateReactionsContact<#rintra, #nsaves>>
                                ::set_last_increment(&mut self.#field_name, increment)
                        }

                        #[inline]
                        fn n_previous_values(&self) -> usize {
                            <#field_type as #backend_path UpdateReactionsContact<#rintra, #nsaves>>
                                ::n_previous_values(&self.#field_name)
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
                    #[inline]
                    fn get_current_neighbors(&self) -> usize {
                        <#field_type as #backend_path UpdateInteraction>::get_current_neighbors(
                            &self.#field_name
                        )
                    }

                    #[inline]
                    fn incr_current_neighbors(&mut self, neighbors: usize) {
                        <#field_type as #backend_path UpdateInteraction>::incr_current_neighbors(
                            &mut self.#field_name,
                            neighbors
                        )
                    }

                    #[inline]
                    fn set_current_neighbors(&mut self, neighbors: usize) {
                        <#field_type as #backend_path UpdateInteraction>::set_current_neighbors(
                            &mut self.#field_name,
                            neighbors
                        )
                    }
                }
            ));
            return TokenStream::from(new_stream);
        }
        TokenStream::new()
    }
}

pub fn generics_placeholders(
    kwargs: impl Into<KwargsAuxStorage>,
    mechanics_solver_order: usize,
    reactions_contact_solver_order: usize,
) -> Vec<proc_macro2::TokenStream> {
    let kwargs: KwargsAuxStorage = kwargs.into();
    let builder = Builder::from(kwargs);
    let (generics, _) = builder.get_generics_and_fields();
    generics
        .into_iter()
        .map(|x| match x {
            syn::GenericParam::Const(c) => {
                if &c.ident == "NMec" {
                    quote::quote!(#mechanics_solver_order)
                } else if &c.ident == "NReact" {
                    quote::quote!(#reactions_contact_solver_order)
                } else {
                    quote::quote!(_)
                }
            }
            _ => quote::quote!(_),
        })
        .collect()
}

// #################################### CONSTRUCT ####################################
pub fn derive_aux_storage(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let aux_storage_parsed = syn::parse_macro_input!(input as AuxStorageParser);
    let aux_storage = AuxStorageImplementer::from(aux_storage_parsed);

    let mut res = TokenStream::new();
    res.extend(aux_storage.implement_update_cycle());
    res.extend(aux_storage.implement_update_mechanics());
    res.extend(aux_storage.implement_update_reactions());
    res.extend(aux_storage.implement_update_reactions_contact());
    res.extend(aux_storage.implement_update_interaction());

    res
}

// #################################### CONSTRUCT ####################################
pub fn default_aux_storage_name() -> syn::Ident {
    syn::Ident::new("_CrAuxStorage", proc_macro2::Span::call_site())
}

define_kwargs!(
    KwargsAuxStorage,
    KwargsAuxStorageParsed,
    aspects: SimulationAspects,
    @optionals
    core_path: syn::Path | crate::kwargs::convert_core_path(None),
    aux_storage_name: syn::Ident | crate::aux_storage::default_aux_storage_name(),
    @from
    KwargsSim,
    KwargsMain
);

pub struct Builder {
    pub struct_name: syn::Ident,
    pub core_path: syn::Path,
    pub aspects: SimulationAspects,
}

impl From<KwargsAuxStorage> for Builder {
    fn from(kwargs: KwargsAuxStorage) -> Self {
        Self {
            struct_name: kwargs.aux_storage_name,
            core_path: kwargs.core_path,
            aspects: kwargs.aspects,
        }
    }
}

pub struct FieldInfo {
    aspects: Vec<SimulationAspect>,
    field_name: syn::Ident,
    field_type: syn::Type,
    generics: syn::Generics,
    fully_formatted_field: proc_macro2::TokenStream,
}

impl Builder {
    pub fn get_field_information(&self) -> Vec<FieldInfo> {
        use SimulationAspect::*;
        let core_path = &self.core_path;
        let backend_path = quote!(#core_path ::backend::chili::);
        // let mut all_generics = vec![];
        let mut fields = vec![];

        if self.aspects.contains(&Cycle) {
            let field_name = syn::parse_quote!(cycle);
            let field_type = syn::parse_quote!(#backend_path AuxStorageCycle);
            let generics = syn::parse_quote!();
            let fully_formatted_field = quote!(
                #[UpdateCycle]
                cycle: #backend_path AuxStorageCycle,
            );
            fields.push(FieldInfo {
                aspects: vec![Cycle],
                field_name,
                field_type,
                generics,
                fully_formatted_field,
            });
        }

        if self.aspects.contains(&Mechanics) {
            let field_name = syn::parse_quote!(mechanics);
            let field_type = syn::parse_quote!(#backend_path AuxStorageMechanics);
            let generics = syn::parse_quote!(<Pos, Vel, For, const NMec: usize>);
            let fully_formatted_field = quote!(
                #[UpdateMechanics(Pos, Vel, For, NMec)]
                #field_name: #backend_path AuxStorageMechanics<Pos, Vel, For, NMec>,
            );
            fields.push(FieldInfo {
                aspects: vec![Mechanics],
                field_name,
                field_type,
                generics,
                fully_formatted_field,
            });
        }

        if self
            .aspects
            .contains_any([&Reactions, &ReactionsContact, &ReactionsExtra])
        {
            let field_name = syn::parse_quote!(reactions);
            let generics = syn::parse_quote!(<Ri>);
            let field_type = syn::parse_quote!(#backend_path AuxStorageReactions);
            let fully_formatted_field = quote!(
                #[UpdateReactions(Ri)]
                #field_name: #backend_path AuxStorageReactions<Ri>,
            );
            fields.push(FieldInfo {
                aspects: vec![Reactions, ReactionsContact, ReactionsExtra],
                field_name,
                field_type,
                generics,
                fully_formatted_field,
            });
        }

        if self.aspects.contains(&ReactionsContact) {
            let field_name = syn::parse_quote!(reactions_extra);
            let field_type = syn::parse_quote!(#backend_path AuxStorageReactionsContact);
            let generics = syn::parse_quote!(<Ri, const NReact: usize>);
            let fully_formatted_field = quote!(
                #[UpdateReactionsContact(Ri, NReact)]
                #field_name: #backend_path AuxStorageReactionsContact<Ri, NReact>,
            );
            fields.push(FieldInfo {
                aspects: vec![ReactionsContact],
                field_name,
                field_type,
                generics,
                fully_formatted_field,
            });
        }

        if self.aspects.contains(&Interaction) {
            let field_name = syn::parse_quote!(interaction);
            let field_type = syn::parse_quote!(AuxStorageInteraction);
            let generics = syn::parse_quote!();
            let fully_formatted_field = quote!(
                #[UpdateInteraction]
                interaction: #backend_path AuxStorageInteraction,
            );
            fields.push(FieldInfo {
                aspects: vec![Interaction],
                field_name,
                field_type,
                generics,
                fully_formatted_field,
            });
        }
        fields
    }

    pub fn build_aux_storage(self) -> proc_macro2::TokenStream {
        let struct_name = &self.struct_name;
        let core_path = &self.core_path;

        let (generics, fields) = self.get_generics_and_fields();

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
pub fn construct_aux_storage(kwargs: KwargsAuxStorage) -> proc_macro::TokenStream {
    let builder: Builder = kwargs.into();
    proc_macro::TokenStream::from(builder.build_aux_storage())
}
