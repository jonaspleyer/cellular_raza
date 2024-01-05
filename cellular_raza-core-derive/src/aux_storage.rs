use proc_macro::TokenStream;
use quote::quote;

// ##################################### PARSING #####################################
#[allow(unused)]
pub struct AuxStorageParser {
    attrs: Vec<syn::Attribute>,
    vis: syn::Visibility,
    struct_token: syn::token::Struct,
    name: syn::Ident,
    generics: syn::Generics,
    aspects: AspectFields,
}

impl syn::parse::Parse for AuxStorageParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let item_struct: syn::ItemStruct = input.parse()?;

        Ok(Self {
            attrs: item_struct.attrs,
            vis: item_struct.vis,
            struct_token: item_struct.struct_token,
            name: item_struct.ident,
            generics: item_struct.generics,
            aspects: AspectFields::from_fields(item_struct.fields)?,
        })
    }
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
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "Could not parse struct. Use named fields for struct definition syntax!",
            ));
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
    float_type: syn::GenericParam,
    _comma_5: Option<syn::token::Comma>,
    n_saves: syn::GenericParam,
    _comma_4: syn::token::Comma,
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
                                float_type: p.float_type,
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
            update_cycle: update_cycle,
            update_mechanics: update_mechanics,
            update_interaction: update_interaction,
            update_reactions: update_reactions,
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
    update_interaction: Option<UpdateInteractionImplementer>,
    update_reactions: Option<UpdateReactionsImplementer>,
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
            let where_clause = match struct_where_clause {
                Some(s_where) => {
                    let pred = s_where.predicates.iter();
                    quote!(
                        where
                        #(#pred,)*
                        #force: Clone + core::ops::AddAssign<#force> + num::Zero,
                        #float_type: Clone,
                    )
                }
                None => quote!(
                    where
                    #force: Clone + core::ops::AddAssign<#force> + num::Zero,
                    #float_type: Clone,
                ),
            };

            let field_name = &update_mechanics.field_name;
            let field_type = &update_mechanics.field_type;

            let new_stream = wrap_pre_flags(quote!(
                impl #struct_impl_generics UpdateMechanics<#field_generics> for #struct_name #struct_ty_generics #where_clause
                {
                    fn set_last_position(&mut self, pos: #position) {
                        <#field_type as UpdateMechanics<#field_generics>>::set_last_position(&mut self.#field_name, pos)
                    }
                    fn previous_positions<'a>(&'a self) -> FixedSizeRingBufferIter<'a, #position, #n_saves> {
                        <#field_type as UpdateMechanics<#field_generics>>::previous_positions(&self.#field_name)
                    }
                    fn set_last_velocity(&mut self, vel: #velocity) {
                        <#field_type as UpdateMechanics<#field_generics>>::set_last_velocity(&mut self.#field_name, vel)
                    }
                    fn previous_velocities<'a>(&'a self) -> FixedSizeRingBufferIter<'a, #velocity, #n_saves> {
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
                impl #impl_generics UpdateReactions<#concentration> for #struct_name #ty_generics #where_clause {
                    fn set_conc(&mut self, conc: #concentration) {
                        <#field_type as UpdateReactions<#concentration>>::set_conc(
                            &mut self.#field_name,
                            conc
                        )
                    }

                    fn get_conc(&self) -> #concentration {
                        <#field_type as UpdateReactions<#concentration>>::get_conc(
                            &self.#field_name
                        )
                    }

                    fn incr_conc(&mut self, incr: #concentration) {
                        <#field_type as UpdateReactions<#concentration>>::incr_conc(
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

            let new_stream = wrap_pre_flags(quote!(
                impl #impl_generics UpdateInteraction for #struct_name #ty_generics #where_clause {
                    fn get_current_neighbours(&self) -> usize {
                        <#field_type as UpdateInteraction>::get_current_neighbours(
                            &self.#field_name
                        )
                    }

                    fn incr_current_neighbours(&mut self, neighbours: usize) {
                        <#field_type as UpdateInteraction>::incr_current_neighbours(
                            &mut self.#field_name,
                            neighbours
                        )
                    }

                    fn set_current_neighbours(&mut self, neighbours: usize) {
                        <#field_type as UpdateInteraction>::set_current_neighbours(
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
struct NameToken;

impl syn::parse::Parse for NameToken {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        match ident=="name" {
            true => return Ok(Self),
            _ => return Err(syn::Error::new(ident.span(), "Expected \"name\" token"))
        }
    }
}

struct AspectsToken;

impl syn::parse::Parse for AspectsToken {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        match ident=="aspects" {
            true => return Ok(Self),
            _ => return Err(syn::Error::new(ident.span(), "Expected \"aspects\" token"))
        }
    }
}

struct AuxStorageBuilder {
    #[allow(unused)]
    name_token: NameToken,
    #[allow(unused)]
    double_colon_1: syn::Token![:],
    struct_name: syn::Ident,
    #[allow(unused)]
    comma: syn::Token![,],
    #[allow(unused)]
    aspects_token: AspectsToken,
    #[allow(unused)]
    double_colon_2: syn::Token![:],
    items: syn::punctuated::Punctuated<AuxStorageAspect, syn::token::Comma>,
}

impl syn::parse::Parse for AuxStorageBuilder {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let name_token = input.parse()?;
        let double_colon_1 = input.parse()?;
        let struct_name = input.parse()?;
        let comma = input.parse()?;
        let aspects_token = input.parse()?;
        let double_colon_2 = input.parse()?;
        let content;
        syn::bracketed!(content in input);
        Ok(Self {
            name_token,
            double_colon_1,
            struct_name,
            comma,
            aspects_token,
            double_colon_2,
            items: syn::punctuated::Punctuated::<AuxStorageAspect, syn::token::Comma>::parse_terminated(
                &content,
            )?,
        })
    }
}

#[derive(Debug)]
enum AuxStorageAspect {
    Mechanics,
    Interaction,
    Cycle,
    Reactions,
}

impl AuxStorageAspect {
    fn get_aspects() -> Vec<AuxStorageAspect> {
        vec![
            AuxStorageAspect::Mechanics,
            AuxStorageAspect::Interaction,
            AuxStorageAspect::Cycle,
            AuxStorageAspect::Reactions,
        ]
    }

    fn build_aux(
        &self,
    ) -> (
        Vec<proc_macro2::TokenStream>,
        proc_macro2::TokenStream,
    ) {
        match &self {
            AuxStorageAspect::Cycle => {
                let generics = vec![];
                let field = quote!(
                    #[UpdateCycle]
                    cycle: AuxStorageCycle,
                );
                (generics, field)
            }
            AuxStorageAspect::Mechanics => {
                let generics = vec![
                    quote!(Pos),
                    quote!(Vel),
                    quote!(For),
                    quote!(Float),
                    quote!(const N: usize),
                ];
                let field = quote!(
                    #[UpdateMechanics(Pos, Vel, For, Float, N)]
                    mechanics: AuxStorageMechanics<Pos, Vel, For, Float, N>,
                );
                (generics, field)
            }
            AuxStorageAspect::Reactions => {
                let generics = vec![quote!(R)];
                let field = quote!(
                    #[UpdateReactions(R)]
                    reactions: AuxStorageReactions<R>,
                );
                (generics, field)
            }
            AuxStorageAspect::Interaction => {
                let generics = vec![];
                let field = quote!(
                    #[UpdateInteraction]
                    interaction: AuxStorageInteraction,
                );
                (generics, field)
            }
        }
    }
}

impl syn::parse::Parse for AuxStorageAspect {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        for aspect in AuxStorageAspect::get_aspects() {
            if ident == format!("{:?}", aspect) {
                return Ok(aspect);
            }
        }
        Err(syn::Error::new(
            ident.span(),
            "Could not find simulation aspect to construct AuxStorage",
        ))
    }
}

impl AuxStorageBuilder {
    fn build_aux_storage(self) -> proc_macro2::TokenStream {
        let struct_name = self.struct_name;
        let mut generics = vec![];
        let mut fields = vec![];
        for item in self.items.into_iter() {
            let (item_generics, item_field) = item.build_aux();
            generics.extend(item_generics);
            fields.extend(item_field);
        }
        let stream = quote!(
            #[allow(non_camel_case_types)]
            #[doc(hidden)]
            #[derive(Clone, Default, Deserialize, Serialize)]
            #[derive(AuxStorage)]
            pub struct #struct_name<#(#generics),*> {
                #(#fields)*
            }
        );
        stream
    }
}

/// Define a AuxStorage struct that
pub fn construct_aux_storage(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let builder = syn::parse_macro_input!(input as AuxStorageBuilder);
    proc_macro::TokenStream::from(builder.build_aux_storage())
}
