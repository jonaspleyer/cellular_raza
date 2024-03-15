use crate::simulation_aspects::SimulationAspects;

macro_rules! implement_token(
    ($token_name:ident, $token_ident:literal) => {
        pub struct $token_name;

        impl syn::parse::Parse for $token_name {
            fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
                let ident: syn::Ident = input.parse()?;
                match ident == $token_ident {
                    true => Ok(Self),
                    _ => Err(syn::Error::new(ident.span(), format!("Expected \"{}\" token", $token_ident))),
                }
            }
        }
    }
);

macro_rules! implement_input_pair(
    ($pair_name:ident, $key_name:ty, $value_type:ty) => {
        pub struct $pair_name {
            #[allow(unused)]
            key: $key_name,
            #[allow(unused)]
            double_colon: syn::Token![:],
            #[allow(unused)]
            value: $value_type,
        }

        impl syn::parse::Parse for $pair_name {
            fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
                Ok(Self {
                    key: input.parse()?,
                    double_colon: input.parse()?,
                    value: input.parse()?,
                })
            }
        }
    }
);

macro_rules! implement_with_trailing_comma(
    ($with_comma_name:ident, $parsed_type:ty) => {
        pub struct $with_comma_name($parsed_type);
        impl syn::parse::Parse for $with_comma_name {
            fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
                let ret = Self(input.parse()?);
                let _comma: syn::Token![,] = input.parse()?;
                Ok(ret)
            }
        }
    }
);

implement_token!(DomainToken, "domain");
implement_input_pair!(DomainInputRaw, DomainToken, syn::Ident);
implement_with_trailing_comma!(DomainInput, DomainInputRaw);
implement_token!(AgentsToken, "agents");
implement_input_pair!(AgentsInputRaw, AgentsToken, syn::Ident);
implement_with_trailing_comma!(AgentsInput, AgentsInputRaw);
implement_token!(SettingsToken, "settings");
implement_input_pair!(SettingsInputRaw, SettingsToken, syn::Ident);
implement_with_trailing_comma!(SettingsInput, SettingsInputRaw);

struct RunSimInputs {
    domain_input: DomainInput,
    agents_input: AgentsInput,
    settings_input: SettingsInput,
    aspects_input: SimulationAspects,
    // TODO more options
    _trailing_comma: Option<syn::Token![,]>,
}

impl syn::parse::Parse for RunSimInputs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(RunSimInputs {
            domain_input: input.parse()?,
            agents_input: input.parse()?,
            settings_input: input.parse()?,
            aspects_input: input.parse()?,
            _trailing_comma: input.parse()?,
        })
    }
}

impl RunSimInputs {
    /// Defines all types which will be used in the simulation
    fn prepare_types(&self) -> proc_macro2::TokenStream {
        proc_macro2::TokenStream::new()
    }

    /// Generate Zero-overhead functions that thest compatibility between
    /// concepts before running the simulation, possibly reducing boilerplate
    /// in compiler errors
    fn test_compatibility(&self) -> proc_macro2::TokenStream {
        proc_macro2::TokenStream::new()
    }

    ///
    fn run_main(&self) -> proc_macro2::TokenStream {
        quote::quote!({Result::<usize, chili::SimulationError>::Ok(1_usize)})
    }
}

pub fn run_simulation(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let run_sim_inputs = syn::parse_macro_input!(input as RunSimInputs);
    let mut output = proc_macro2::TokenStream::new();
    output.extend(run_sim_inputs.prepare_types());
    output.extend(run_sim_inputs.test_compatibility());
    output.extend(run_sim_inputs.run_main());
    output.into()
}
