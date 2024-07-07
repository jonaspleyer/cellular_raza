use proc_macro2::TokenStream;
use quote::quote;

macro_rules! new_ident(
    ($name:ident, $ident:literal) => {
        let $name = proc_macro2::Ident::new($ident, proc_macro2::Span::call_site());
    }
);

macro_rules! push_ident(
    ($generics:ident, $ident:ident) => {
        if !$generics.params.empty_or_trailing() {
            let punct = syn::Token![,](proc_macro2::Span::call_site());
            $generics.params.push_punct(punct);
        }
        $generics.params.push_value(syn::TypeParam::from($ident.clone()).into());
    }
);

macro_rules! append_where_clause(
    ($struct_where_clause:ident, $field_type:ident, $trait_name:ident, $tokens:ident) => {
        match $struct_where_clause {
            Some(clause) => {
                let punct = if clause.predicates.trailing_punct() {
                    quote::quote!()
                } else {
                    quote::quote!(,)
                };
                quote::quote!(
                    #clause #punct
                    #$field_type: $trait_name<#$tokens>,
                )
            },
            None => quote::quote!(where #$field_type: $trait_name<#$tokens>),
        }
    }
);

// ##################################### PARSING #####################################
#[allow(unused)]
pub struct AgentParser {
    attrs: Vec<syn::Attribute>,
    vis: syn::Visibility,
    struct_token: syn::Token![struct],
    name: syn::Ident,
    generics: syn::Generics,
    aspects: Vec<CellAspectField>,
}

impl syn::parse::Parse for AgentParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let item_struct: syn::ItemStruct = input.parse()?;
        let attrs = item_struct.attrs;
        let vis = item_struct.vis;
        let struct_token = item_struct.struct_token;
        let name = item_struct.ident;
        let generics = item_struct.generics;
        let aspects = CellAspectField::from_fields(name.span(), item_struct.fields)?;

        let res = Self {
            attrs,
            vis,
            struct_token,
            name,
            generics,
            aspects,
        };
        Ok(res)
    }
}

#[derive(Clone)]
enum CellAspect {
    Mechanics,
    Cycle,
    Interaction,
    Reactions,
    ExtracellularGradient,
    Volume,
}

impl CellAspect {
    fn from_attribute(attr: &syn::Attribute) -> Option<Self> {
        let path = attr.meta.path().get_ident();

        if let Some(p) = path {
            let path_str = p.to_string();
            match path_str.as_str() {
                "Mechanics" => Some(CellAspect::Mechanics),
                "Cycle" => Some(CellAspect::Cycle),
                "Interaction" => Some(CellAspect::Interaction),
                "Reactions" => Some(CellAspect::Reactions),
                "ExtracellularGradient" => Some(CellAspect::ExtracellularGradient),
                "Volume" => Some(CellAspect::Volume),
                _ => None,
            }
        } else {
            None
        }
    }
}

// ------------------------------------- ASPECTS -------------------------------------
#[derive(Clone)]
struct CellAspectField {
    aspects: Vec<CellAspect>,
    field: syn::Field,
}

impl CellAspectField {
    fn from_field(field: syn::Field) -> Self {
        let aspects = field
            .attrs
            .iter()
            .map(CellAspect::from_attribute)
            .filter_map(|s| s)
            .collect::<Vec<_>>();
        Self { aspects, field }
    }

    fn from_fields(
        span: proc_macro2::Span,
        fields: syn::Fields,
    ) -> syn::Result<Vec<CellAspectField>> {
        match fields {
            syn::Fields::Named(fields_named) => Ok(fields_named
                .named
                .into_iter()
                .map(|field| CellAspectField::from_field(field))
                .collect::<Vec<_>>()),
            syn::Fields::Unnamed(fields_unnamed) => Ok(fields_unnamed
                .unnamed
                .into_iter()
                .map(|field| CellAspectField::from_field(field))
                .collect::<Vec<_>>()),
            syn::Fields::Unit => Err(syn::Error::new(span, "Cannot derive from unit struct")),
        }
    }
}

pub struct FieldInfo {
    pub field_type: syn::Type,
    pub field_name: Option<syn::Ident>,
}

// ################################### IMPLEMENTING ##################################
pub struct AgentImplementer {
    name: syn::Ident,
    generics: syn::Generics,
    cycle: Option<FieldInfo>,
    mechanics: Option<FieldInfo>,
    interaction: Option<FieldInfo>,
    cellular_reactions: Option<FieldInfo>,
    extracellular_gradient: Option<FieldInfo>,
    volume: Option<FieldInfo>,
}

impl From<AgentParser> for AgentImplementer {
    fn from(value: AgentParser) -> Self {
        let mut cycle = None;
        let mut mechanics = None;
        let mut interaction = None;
        let mut cellular_reactions = None;
        let mut extracellular_gradient = None;
        let mut volume = None;

        value.aspects.into_iter().for_each(|aspect_field| {
            aspect_field.aspects.into_iter().for_each(|aspect| {
                let field_info = FieldInfo {
                    field_type: aspect_field.field.ty.clone(),
                    field_name: aspect_field.field.ident.clone(),
                };
                match aspect {
                    CellAspect::Cycle => {
                        cycle = Some(field_info);
                    }
                    CellAspect::Mechanics => mechanics = Some(field_info),
                    CellAspect::Interaction => {
                        interaction = Some(field_info);
                    }
                    CellAspect::Reactions => {
                        cellular_reactions = Some(field_info);
                    }
                    CellAspect::ExtracellularGradient => {
                        extracellular_gradient = Some(field_info);
                    }
                    CellAspect::Volume => {
                        volume = Some(field_info);
                    }
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

pub fn wrap(input: TokenStream) -> TokenStream {
    quote! {
        #[allow(non_upper_case_globals)]
        const _: () = {
            // TODO consider adding specific import of cellular_raza or
            // cellular_raza_concepts
            //
            // ```
            // crate extern crate cellular_raza as _cr;
            // or
            // extern crate cellular_raza_concepts as _cr;
            // ```
            //
            // Also put a _cr::prelude::TRAIT in front of every implemented trait
            // This is currently not possible to do at compile time without
            // any hacks (to my knowledge)
            #input
        };
    }
}

impl AgentImplementer {
    pub fn implement_cycle(&self) -> TokenStream {
        let struct_name = &self.name;
        let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.cycle {
            let field_type = &field_info.field_type;
            new_ident!(float_type, "__cr_private_Float");

            let tokens = quote!(#struct_name #struct_ty_generics, #float_type);

            let where_clause = append_where_clause!(struct_where_clause, field_type, Cycle, tokens);

            let mut generics = self.generics.clone();
            push_ident!(generics, float_type);
            let impl_generics = generics.split_for_impl().0;

            let new_stream = quote!(
                #[automatically_derived]
                impl #impl_generics Cycle<#tokens>
                for #struct_name #struct_ty_generics #where_clause {
                    fn update_cycle(
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: &#float_type,
                        cell: &mut Self,
                    ) -> Option<CycleEvent> {
                        <#field_type as Cycle<#tokens>>::update_cycle(rng, dt, cell)
                    }

                    fn divide(
                        rng: &mut rand_chacha::ChaCha8Rng,
                        cell: &mut Self
                    ) -> Result<Self, DivisionError> {
                        <#field_type as Cycle<#tokens>>::divide(rng, cell)
                    }

                    fn update_conditional_phased_death(
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: &#float_type,
                        cell: &mut Self,
                    ) -> Result<bool, DeathError> {
                        <#field_type as Cycle<#tokens>>::update_conditional_phased_death(
                            rng,
                            dt,
                            cell
                        )
                    }
                }
            );
            return TokenStream::from(new_stream);
        }
        TokenStream::new()
    }

    pub fn implement_mechanics(&self) -> TokenStream {
        let struct_name = &self.name;
        let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.mechanics {
            new_ident!(position, "__cr_private_Pos");
            new_ident!(velocity, "__cr_private_Vel");
            new_ident!(force, "__cr_private_For");
            new_ident!(float_type, "__cr_private_Float");

            let tokens = quote!(#position, #velocity, #force, #float_type);
            let field_type = &field_info.field_type;
            let field_name = &field_info.field_name;

            let where_clause =
                append_where_clause!(struct_where_clause, field_type, Mechanics, tokens);

            let mut generics = self.generics.clone();
            push_ident!(generics, position);
            push_ident!(generics, velocity);
            push_ident!(generics, force);
            push_ident!(generics, float_type);
            let impl_generics = generics.split_for_impl().0;

            let res = quote! {
                #[automatically_derived]
                impl #impl_generics Mechanics<#tokens> for #struct_name #struct_ty_generics
                    #where_clause
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
                        <#field_type as Mechanics<#tokens>>::set_velocity(
                            &mut self.#field_name,
                            velocity
                        )
                    }
                    fn calculate_increment(&self, force: #force)
                        -> Result<(#position, #velocity), CalcError> {
                        <#field_type as Mechanics<#tokens>>::calculate_increment(
                            &self.#field_name,
                            force
                        )
                    }
                    fn set_random_variable(&mut self,
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: #float_type,
                    ) -> Result<(), RngError> {
                        <#field_type as Mechanics<#tokens>>::set_random_variable(
                            &mut self.#field_name,
                            rng,
                            dt
                        )
                    }
                }
            };
            return TokenStream::from(res);
        }
        TokenStream::new()
    }

    pub fn implement_interaction(&self) -> TokenStream {
        let struct_name = &self.name;
        let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.interaction {
            let field_name = &field_info.field_name;
            let field_type = &field_info.field_type;
            new_ident!(position, "__cr_private_Pos");
            new_ident!(velocity, "__cr_private_Vel");
            new_ident!(force, "__cr_private_For");
            new_ident!(information, "__cr_private_Inf");
            let tokens = quote!(#position, #velocity, #force, #information);

            let where_clause =
                append_where_clause!(struct_where_clause, field_type, Interaction, tokens);

            let mut generics = self.generics.clone();
            push_ident!(generics, position);
            push_ident!(generics, velocity);
            push_ident!(generics, force);
            push_ident!(generics, information);
            let impl_generics = generics.split_for_impl().0;

            let res = quote! {
                #[automatically_derived]
                impl #impl_generics Interaction<#tokens>
                    for #struct_name #struct_ty_generics #where_clause {
                    fn get_interaction_information(&self) -> #information {
                        <#field_type as Interaction<#tokens
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
                    ) -> Result<(#force, #force), CalcError> {
                        <#field_type as Interaction<#tokens>>::calculate_force_between(
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
                        <#field_type as Interaction<#tokens>>::is_neighbour(
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
                        <#field_type as Interaction<#tokens>>::react_to_neighbours(
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

    pub fn implement_reactions(&self) -> TokenStream {
        let struct_name = &self.name;
        let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.cellular_reactions {
            let field_name = &field_info.field_name;
            let field_type = &field_info.field_type;
            new_ident!(cvec_intra, "__cr_private_CVecIntra");
            new_ident!(cvec_extra, "__cr_private_CVecExtra");
            let tokens = quote!(#cvec_intra, #cvec_extra);

            let where_clause =
                append_where_clause!(struct_where_clause, field_type, CellularReactions, tokens);

            let mut generics = self.generics.clone();
            push_ident!(generics, cvec_intra);
            push_ident!(generics, cvec_extra);
            let impl_generics = generics.split_for_impl().0;

            let res = quote! {
                #[automatically_derived]
                impl #impl_generics CellularReactions<
                    #cvec_intra,
                    #cvec_extra
                > for #struct_name #struct_ty_generics #where_clause {
                    fn get_intracellular(&self) -> #cvec_intra {
                        <#field_type as CellularReactions<
                            #cvec_intra,
                            #cvec_extra
                        >>::get_intracellular(&self.#field_name)
                    }

                    fn set_intracellular(
                        &mut self,
                        concentration_vector: #cvec_intra
                    ) {
                        <#field_type as CellularReactions<
                            #cvec_intra,
                            #cvec_extra
                        >>::set_intracellular(
                            &mut self.#field_name,
                            concentration_vector
                        );
                    }

                    fn calculate_intra_and_extracellular_reaction_increment(
                        &self,
                        internal_concentration_vector: &#cvec_intra,
                        external_concentration_vector: &#cvec_extra,
                    ) -> Result<(#cvec_intra, #cvec_extra), CalcError> {
                        <#field_type as CellularReactions<
                            #cvec_intra,
                            #cvec_extra
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

    pub fn implement_extracellular_gradient(&self) -> TokenStream {
        let struct_name = &self.name;
        let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

        if let Some(field_info) = &self.extracellular_gradient {
            let field_type = &field_info.field_type;
            new_ident!(extra_gradient, "__cr_private_ExtraGradient");
            let tokens = quote!(#struct_name #struct_ty_generics, #extra_gradient);

            let where_clause = append_where_clause!(
                struct_where_clause,
                field_type,
                InteractionExtracellularGradient,
                tokens
            );

            let mut generics = self.generics.clone();
            push_ident!(generics, extra_gradient);
            let impl_generics = generics.split_for_impl().0;

            let res = quote! {
                #[automatically_derived]
                impl #impl_generics InteractionExtracellularGradient<#tokens>
                    for #struct_name #struct_ty_generics #where_clause {
                    fn sense_gradient(
                        cell: &mut #struct_name #struct_ty_generics,
                        gradient: &#extra_gradient,
                    ) -> Result<(), CalcError> {
                        <#field_type as InteractionExtracellularGradient<#tokens>>
                            ::sense_gradient(cell, gradient)
                    }
                }
            };
            return TokenStream::from(res);
        }
        TokenStream::new()
    }

    pub fn implement_volume(&self) -> TokenStream {
        let struct_name = &self.name;
        let (_, struct_ty_generics, struct_where_clause) = &self.generics.split_for_impl();

        if let Some(volume_implementer) = &self.volume {
            let field_type = &volume_implementer.field_type;
            let field_name = &volume_implementer.field_name;
            new_ident!(float_type, "__cr_private_Float");

            let where_clause =
                append_where_clause!(struct_where_clause, field_type, Volume, float_type);

            let mut generics = self.generics.clone();
            push_ident!(generics, float_type);
            let impl_generics = generics.split_for_impl().0;

            let res = quote! {
                #[automatically_derived]
                impl #impl_generics Volume<#float_type> for #struct_name #struct_ty_generics
                    #where_clause
                {
                    fn get_volume(&self) -> #float_type {
                        <#field_type as Volume<#float_type>>::get_volume(
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
