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
    fn from_attribute(attr: &syn::Attribute) -> syn::Result<Option<Self>> {
        let path = attr.meta.path().get_ident();

        if let Some(p) = path {
            let path_str = p.to_string();
            match path_str.as_str() {
                "Mechanics" => Ok(Some(CellAspect::Mechanics)),
                "Cycle" => Ok(Some(CellAspect::Cycle)),
                "Interaction" => Ok(Some(CellAspect::Interaction)),
                "Reactions" => Ok(Some(CellAspect::Reactions)),
                "ExtracellularGradient" => Ok(Some(CellAspect::ExtracellularGradient)),
                "Volume" => Ok(Some(CellAspect::Volume)),
                _ => Ok(None),
            }
        } else {
            Ok(None)
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
    fn from_field(field: syn::Field) -> syn::Result<Self> {
        let mut errors = vec![];
        let aspects = field
            .attrs
            .iter()
            .map(CellAspect::from_attribute)
            .filter_map(|r| r.map_err(|e| errors.push(e)).ok())
            .filter_map(|s| s)
            .collect::<Vec<_>>();
        for e in errors.into_iter() {
            return Err(e);
        }
        Ok(Self { aspects, field })
    }

    fn from_fields(span: proc_macro2::Span, fields: syn::Fields) -> syn::Result<Vec<AspectField>> {
        match fields {
            syn::Fields::Named(fields_named) => Ok(fields_named
                .named
                .into_iter()
                .map(|field|CellAspectField::from_field(field))
                .collect::<syn::Result<Vec<_>>>()?),
            syn::Fields::Unnamed(fields_unnamed) => Ok(fields_unnamed
                .unnamed
                .into_iter()
                .map(|field|CellAspectField::from_field(field))
                .collect::<syn::Result<Vec<_>>>()?),
            syn::Fields::Unit => Err(syn::Error::new(span, "Cannot derive from unit struct")),
        }
    }
}

struct FieldInfo {
    field_type: syn::Type,
    field_name: Option<syn::Ident>,
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
            // TODO consider adding specific import of cellular_raza or cellular_raza_concepts crate
            // extern crate cellular_raza as _cr;
            // or
            // extern crate cellular_raza_concepts as _cr;
            //
            // Also put a _cr::prelude::TRAIT in front of every implemented trait
            // This is currently not possible to do at compile time without any hacks (to my knowledge)
            #input
        };
    }
}

impl AgentImplementer {
    pub fn implement_cycle(&self) -> TokenStream {
        let struct_name = &self.name;
        let (struct_impl_generics, struct_ty_generics, struct_where_clause) =
            &self.generics.split_for_impl();

        if let Some(cycle_implementer) = &self.cycle {
            let float_type = match &cycle_implementer.float_type {
                Some(ty) => quote!(#ty),
                None => quote!(f64),
            };
            let field_type = &cycle_implementer.field_type;

            let tokens = quote!(#struct_name #struct_ty_generics, #float_type);

            let new_stream = quote!(
                #[automatically_derived]
                impl #struct_impl_generics Cycle<#tokens> for #struct_name #struct_ty_generics #struct_where_clause {
                    fn update_cycle(
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: &#float_type,
                        cell: &mut Self,
                    ) -> Option<CycleEvent> {
                        <#field_type as Cycle<#tokens>>::update_cycle(rng, dt, cell)
                    }

                    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Self) -> Result<Self, DivisionError> {
                        <#field_type as Cycle<#tokens>>::divide(rng, cell)
                    }

                    fn update_conditional_phased_death(
                        rng: &mut rand_chacha::ChaCha8Rng,
                        dt: &#float_type,
                        cell: &mut Self,
                    ) -> Result<bool, DeathError> {
                        <#field_type as Cycle<#tokens>>::update_conditional_phased_death(rng, dt, cell)
                    }
                }
            );
            return TokenStream::from(new_stream);
        }
        TokenStream::new()
    }

    pub fn implement_mechanics(&self) -> TokenStream {
        let struct_name = &self.name;
        let (struct_impl_generics, struct_ty_generics, struct_where_clause) =
            &self.generics.split_for_impl();

        if let Some(mechanics_implementer) = &self.mechanics {
            let position = &mechanics_implementer.position;
            let velocity = &mechanics_implementer.velocity;
            let force = &mechanics_implementer.force;
            let float_type = match &mechanics_implementer.float_type {
                Some(ty) => quote!(#ty),
                None => quote!(f64),
            };

            let tokens = quote!(#position, #velocity, #force, #float_type);
            let field_type = &mechanics_implementer.field_type;
            let field_name = &mechanics_implementer.field_name;

            let res = quote! {
                #[automatically_derived]
                impl #struct_impl_generics Mechanics<#tokens> for #struct_name #struct_ty_generics #struct_where_clause
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
                        dt: #float_type,
                    ) -> Result<Option<#float_type>, RngError> {
                        <#field_type as Mechanics<#tokens>>::set_random_variable(&mut self.#field_name, rng, dt)
                    }
                }
            };
            return TokenStream::from(res);
        }
        TokenStream::new()
    }

    pub fn implement_interaction(&self) -> TokenStream {
        let struct_name = &self.name;
        let (struct_impl_generics, struct_ty_generics, struct_where_clause) =
            &self.generics.split_for_impl();

        if let Some(interaction_implementer) = &self.interaction {
            let field_name = &interaction_implementer.field_name;
            let field_type = &interaction_implementer.field_type;
            let position = &interaction_implementer.position;
            let velocity = &interaction_implementer.velocity;
            let force = &interaction_implementer.force;
            let information = &interaction_implementer.information;

            let res = quote! {
                #[automatically_derived]
                impl #struct_impl_generics Interaction<
                    #position,
                    #velocity,
                    #force,
                    #information
                > for #struct_name #struct_ty_generics #struct_where_clause {
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
                    ) -> Result<#force, CalcError> {
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

    pub fn implement_reactions(&self) -> TokenStream {
        let struct_name = &self.name;
        let (struct_impl_generics, struct_ty_generics, struct_where_clause) =
            &self.generics.split_for_impl();

        if let Some(cellular_reactions_implementer) = &self.cellular_reactions {
            let field_name = &cellular_reactions_implementer.field_name;
            let field_type = &cellular_reactions_implementer.field_type;
            let concvecintracellular = &cellular_reactions_implementer.concvecintracellular;
            let concvecextracellular = &cellular_reactions_implementer.concvecextracellular;

            let res = quote! {
                #[automatically_derived]
                impl #struct_impl_generics CellularReactions<
                    #concvecintracellular,
                    #concvecextracellular
                > for #struct_name #struct_ty_generics #struct_where_clause {
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

    pub fn implement_extracellular_gradient(&self) -> TokenStream {
        let struct_name = &self.name;
        let (struct_impl_generics, struct_ty_generics, struct_where_clause) =
            &self.generics.split_for_impl();

        if let Some(extracellular_gradient_implementer) = &self.extracellular_gradient {
            let field_type = &extracellular_gradient_implementer.field_type;

            let extracellular_gradient = &extracellular_gradient_implementer.extracellular_gradient;
            let res = quote! {
                #[automatically_derived]
                impl #struct_impl_generics InteractionExtracellularGradient<
                    #struct_name #struct_ty_generics,
                    #extracellular_gradient
                > for #struct_name #struct_ty_generics #struct_where_clause {
                    fn sense_gradient(
                        cell: &mut #struct_name #struct_ty_generics,
                        gradient: &#extracellular_gradient,
                    ) -> Result<(), CalcError> {
                        <#field_type as InteractionExtracellularGradient<
                            #struct_name #struct_ty_generics,
                            #extracellular_gradient
                        >>::sense_gradient(cell, gradient)
                    }
                }
            };
            return TokenStream::from(res);
        }
        TokenStream::new()
    }

    pub fn implement_volume(&self) -> TokenStream {
        let struct_name = &self.name;
        let (struct_impl_generics, struct_ty_generics, struct_where_clause) =
            &self.generics.split_for_impl();

        if let Some(volume_implementer) = &self.volume {
            let field_type = &volume_implementer.field_type;
            let field_name = &volume_implementer.field_name;
            let float_type = &volume_implementer.float_type;

            let res = quote! {
                #[automatically_derived]
                impl #struct_impl_generics Volume<#float_type> for #struct_name #struct_ty_generics #struct_where_clause {
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
