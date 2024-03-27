use proc_macro2::TokenStream;
use quote::quote;

// ##################################### PARSING #####################################
#[allow(unused)]
pub struct AgentParser {
    attrs: Vec<syn::Attribute>,
    vis: syn::Visibility,
    struct_token: syn::Token![struct],
    name: syn::Ident,
    generics: syn::Generics,
    aspects: Vec<AspectField>,
}

impl syn::parse::Parse for AgentParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let item_struct: syn::ItemStruct = input.parse()?;
        let attrs = item_struct.attrs;
        let vis = item_struct.vis;
        let struct_token = item_struct.struct_token;
        let name = item_struct.ident;
        let generics = item_struct.generics;
        let aspects = AspectField::from_fields(name.span(), item_struct.fields)?;

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

// ------------------------------------ MECHANICS ------------------------------------
#[derive(Clone)]
struct MechanicsParser {
    position: syn::Type,
    _comma_1: syn::Token![,],
    velocity: syn::Type,
    _comma_2: syn::Token![,],
    force: syn::Type,
    _comma_3: Option<syn::Token![,]>,
    float_type: Option<syn::Type>,
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
            _comma_3: content.parse()?,
            float_type: if content.is_empty() {
                None
            } else {
                Some(content.parse()?)
            },
        })
    }
}

struct MechanicsImplementer {
    position: syn::Type,
    velocity: syn::Type,
    force: syn::Type,
    float_type: Option<syn::Type>,
    field_type: syn::Type,
    field_name: Option<syn::Ident>,
}

// ----------------------------------- INTERACTION -----------------------------------
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
            },
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

// ------------------------------- CELLULAR-REACTIONS --------------------------------
#[derive(Clone)]
struct ReactionsParser {
    concvecintracellular: syn::Type,
    _comma: Option<syn::Token![,]>,
    concvecextracellular: syn::Type,
}

impl syn::parse::Parse for ReactionsParser {
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

struct ReactionsImplementer {
    concvecintracellular: syn::Type,
    concvecextracellular: syn::Type,
    field_type: syn::Type,
    field_name: Option<syn::Ident>,
}

// ------------------------------ EXTRACELLULAR-GRADIENT -----------------------------
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

// ------------------------------------- VOLUME --------------------------------------
#[derive(Clone)]
struct VolumeParser {
    float_type: syn::Type,
}

impl syn::parse::Parse for VolumeParser {
    #[allow(unused)]
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _volume: syn::Ident = input.parse()?;
        let float_type = if input.is_empty() {
            syn::parse_quote!(f64)
        } else {
            let content;
            syn::parenthesized!(content in input);
            content.parse()?
        };
        Ok(Self { float_type })
    }
}

struct VolumeImplementer {
    float_type: syn::Type,
    field_type: syn::Type,
    field_name: Option<syn::Ident>,
}

// -------------------------------------- CYCLE --------------------------------------
#[derive(Clone)]
struct CycleParser {
    float_type: Option<syn::Type>,
}

struct CycleImplementer {
    float_type: Option<syn::Type>,
    field_type: syn::Type,
}

impl syn::parse::Parse for CycleParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _cycle: syn::Ident = input.parse()?;
        let float_type = if input.is_empty() {
            None
        } else {
            let content;
            syn::parenthesized!(content in input);
            Some(content.parse()?)
        };
        Ok(Self { float_type })
    }
}

#[derive(Clone)]
enum Aspect {
    Mechanics(MechanicsParser),
    Cycle(CycleParser),
    Interaction(InteractionParser),
    Reactions(ReactionsParser),
    ExtracellularGradient(ExtracellularGradientParser),
    Volume(VolumeParser),
}

impl Aspect {
    fn from_attribute(attr: &syn::Attribute) -> syn::Result<Option<Self>> {
        let path = attr.meta.path().get_ident();
        let cmp = |c: &str| path.is_some_and(|p| p.to_string() == c);

        let s = &attr.meta;
        let stream: proc_macro::TokenStream = quote!(#s).into();

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
            let parsed: ReactionsParser = syn::parse(stream)?;
            return Ok(Some(Aspect::Reactions(parsed)));
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

// ------------------------------------- ASPECTS -------------------------------------
#[derive(Clone)]
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

    fn from_fields(span: proc_macro2::Span, fields: syn::Fields) -> syn::Result<Vec<AspectField>> {
        match fields {
            syn::Fields::Named(fields_named) => Ok(fields_named
                .named
                .into_iter()
                .map(|field| AspectField::from_field(field))
                .collect::<syn::Result<Vec<_>>>()?),
            syn::Fields::Unnamed(fields_unnamed) => Ok(fields_unnamed
                .unnamed
                .into_iter()
                .map(|field| AspectField::from_field(field))
                .collect::<syn::Result<Vec<_>>>()?),
            syn::Fields::Unit => Err(syn::Error::new(span, "Cannot derive from unit struct")),
        }
    }
}

// ################################### IMPLEMENTING ##################################
pub struct AgentImplementer {
    name: syn::Ident,
    generics: syn::Generics,
    cycle: Option<CycleImplementer>,
    mechanics: Option<MechanicsImplementer>,
    interaction: Option<InteractionImplementer>,
    cellular_reactions: Option<ReactionsImplementer>,
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

        value.aspects.into_iter().for_each(|aspect_field| {
            aspect_field
                .aspects
                .into_iter()
                .for_each(|aspect| match aspect {
                    Aspect::Cycle(p) => {
                        cycle = Some(CycleImplementer {
                            float_type: p.float_type,
                            field_type: aspect_field.field.ty.clone(),
                        })
                    }
                    Aspect::Mechanics(p) => {
                        mechanics = Some(MechanicsImplementer {
                            position: p.position,
                            velocity: p.velocity,
                            force: p.force,
                            float_type: p.float_type,
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
                    Aspect::Reactions(p) => {
                        cellular_reactions = Some(ReactionsImplementer {
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
                            float_type: p.float_type,
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