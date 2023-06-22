use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(AuxStorage, attributes(UpdateCycle, UpdateMechanics))]
pub fn aux_storage(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let ast = parse_macro_input!(input as DeriveInput);

    // Build the output, possibly using quasi-quotation
    let struct_name = ast.ident;
    let struct_generics = ast.generics.clone();
    let mut result = TokenStream::new();

    let data: syn::DataStruct = match ast.data {
        syn::Data::Struct(data) => data,
        _ => panic!("Usage of #[UpdateCycle] on a non-struct type"),
    };
    for field in data.fields.iter() {
        // Update Cycle
        if field.attrs.iter().any(|x| match &x.meta {
            syn::Meta::Path(path) => path.is_ident("UpdateCycle"),
            _ => false,
        }) {
            let name = &field.ident;
            let res2 = quote! {
                impl #struct_generics UpdateCycle for #struct_name #struct_generics {
                    fn set_cycle_events(&mut self, events: Vec<CycleEvent>) {
                        self.#name.set_cycle_events(events)
                    }

                    fn get_cycle_events(&self) -> Vec<CycleEvent> {
                        self.#name.get_cycle_events()
                    }

                    fn add_cycle_event(&mut self, event: CycleEvent) {
                        self.#name.add_cycle_event(event)
                    }
                }
            };
            result.extend(TokenStream::from(res2));
        }
        // Update Mechanics
        else if field.attrs.iter().any(|x| match &x.meta {
            syn::Meta::Path(path) => path.is_ident("UpdateMechanics"),
            _ => false,
        }) {
            let name = &field.ident;
            let generic_args_field = match &field.ty {
                syn::Type::Path(path) => {
                    path.path
                        .segments
                        .first()
                        .and_then(|segment| match &segment.arguments {
                            syn::PathArguments::AngleBracketed(arg) => {
                                Some(arg.args.clone().into_token_stream())
                            }
                            _ => None,
                        })
                }
                _ => None,
            }
            .or(Some("".into_token_stream()))
            .unwrap();

            // println!("{}", field_type.into_token_stream().to_string());
            let res2 = quote! {
                // TODO these generic parameters <P, V, N> should be inferred if possible
                // but it does not seem to be possible at this time.
                impl #struct_generics UpdateMechanics <#generic_args_field> for #struct_name #struct_generics {
                    fn set_last_position(&mut self, pos: P) {
                        self.#name.set_last_position(pos)
                    }
                    fn previous_positions(&self) -> std::collections::vec_deque::Iter<P> {
                        self.#name.previous_positions()
                    }
                    fn set_last_velocity(&mut self, vel: V) {
                        self.#name.set_last_velocity(vel)
                    }
                    fn previous_velocities(&self) -> std::collections::vec_deque::Iter<V> {
                        self.#name.previous_velocities()
                    }
                }
            };
            // println!("{}", res2.clone().into_token_stream().to_string());
            result.extend(TokenStream::from(res2));
        }
    }

    // Hand the output tokens back to the compiler
    TokenStream::from(result)
}
