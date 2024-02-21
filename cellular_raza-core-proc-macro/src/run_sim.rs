struct RunSimInputs {}

impl syn::parse::Parse for RunSimInputs {
    fn parse(_input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(RunSimInputs {})
    }
}

pub fn run_sim(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let _run_sim_inputs = syn::parse_macro_input!(input as RunSimInputs);
    proc_macro::TokenStream::new()
}
