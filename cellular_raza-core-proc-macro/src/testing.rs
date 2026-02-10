use itertools::Itertools;
use quote::quote;

use super::simulation_aspects::{SimulationAspect, SimulationAspects};

#[allow(unused)]
struct Sorted {
    sorted_kw: syn::Ident,
    colon: syn::Token![:],
    sorted: bool,
}

impl syn::parse::Parse for Sorted {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            sorted_kw: input.parse()?,
            colon: input.parse()?,
            sorted: input.parse::<syn::LitBool>()?.value,
        })
    }
}

#[allow(unused)]
struct MacroParser {
    test_token: syn::Ident,
    colon: syn::Token![:],
    macro_name: syn::Ident,
    comma: syn::Token![,],
    aspects: SimulationAspects,
    min_combinations: Option<core::num::NonZeroUsize>,
    sorted: Option<bool>,
}

impl syn::parse::Parse for MacroParser {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut res = Self {
            test_token: input.parse()?,
            colon: input.parse()?,
            macro_name: input.parse()?,
            comma: input.parse()?,
            aspects: input.parse()?,
            min_combinations: None,
            sorted: None,
        };
        while !input.is_empty() {
            let _: syn::Token![,] = input.parse()?;
            if !input.is_empty() {
                let keyword: syn::Ident = input.parse()?;
                let _: syn::Token![:] = input.parse()?;
                match keyword.to_string().as_ref() {
                    "min_combinations" => {
                        res.min_combinations = Some(input.parse::<syn::LitInt>()?.base10_parse()?)
                    }
                    "sorted" => res.sorted = Some(input.parse::<syn::LitBool>()?.value),
                    _ => (),
                }
            }
        }
        Ok(res)
    }
}

impl MacroParser {
    fn spawn_tests(self) -> proc_macro2::TokenStream {
        let macro_name = &self.macro_name;
        let aspects: Vec<_> = self.aspects.to_aspect_list();
        let min_order = self.min_combinations.map(|x| x.get()).unwrap_or(1);
        let sorted = self.sorted.unwrap_or(true);

        let mut stream = quote!();
        for n in min_order..aspects.len() {
            let combinations = get_combinations(n, aspects.clone(), sorted);

            for (name, list) in combinations {
                let list_aspects = list.into_iter().map(|aspect| aspect.to_token_stream());
                let output = quote!(
                    #macro_name !(
                        name:#name,
                        aspects:[#(#list_aspects),*]
                    );
                );
                stream.extend(output);
            }
        }
        stream
    }
}

pub fn run_test_for_aspects(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let macro_parser = syn::parse_macro_input!(input as MacroParser);
    macro_parser.spawn_tests().into()
}

fn idents_overlap(id1: &proc_macro2::TokenStream, id2: &proc_macro2::TokenStream) -> bool {
    let id1_segments = id1.to_string();
    let id1_segments = id1_segments.split("_").collect::<Vec<_>>();
    let id2_segments = id2.to_string();
    let id2_segments = id2_segments.split("_").collect::<Vec<_>>();
    let l1 = id1_segments.len();
    let l2 = id2_segments.len();
    let set = std::collections::HashSet::<&str>::from_iter(
        id1_segments.into_iter().chain(id2_segments.into_iter()),
    );
    set.len() < l1 + l2
}

fn get_combinations(
    n: usize,
    idents: Vec<SimulationAspect>,
    sorted: bool,
) -> Vec<(proc_macro2::TokenStream, Vec<SimulationAspect>)> {
    let idents: Vec<(proc_macro2::TokenStream, Vec<SimulationAspect>)> = idents
        .into_iter()
        .map(|s| (s.to_token_stream_lowercase(), vec![s]))
        .collect();

    if n == 0 {
        return idents;
    }

    fn combine_idents(
        ident1: &(proc_macro2::TokenStream, Vec<SimulationAspect>),
        ident2: &(proc_macro2::TokenStream, Vec<SimulationAspect>),
    ) -> Option<(proc_macro2::TokenStream, Vec<SimulationAspect>)> {
        if idents_overlap(&ident1.0, &ident2.0) {
            return None;
        }
        let i1 = &ident1.0;
        let i2 = &ident2.0;
        let name_ident = quote::format_ident!("{}_{}", i1.to_string(), i2.to_string());
        let name = quote!(#name_ident);
        let mut list = ident1.1.clone();
        let list2 = ident2.1.clone();
        list.extend(list2);
        let list = list.into_iter().map(|s| s.into()).collect();
        Some((name, list))
    }

    if sorted {
        return idents
            .iter()
            .combinations(n)
            .into_iter()
            .map(|ids| {
                ids.into_iter()
                    .fold((quote::quote!(), vec![]), |mut acc, x| {
                        if acc.1.is_empty() {
                            acc = x.clone();
                            return acc;
                        }
                        match combine_idents(&acc, x) {
                            Some(res) => {
                                acc = res;
                                acc
                            }
                            None => acc,
                        }
                    })
            })
            .collect();
    }

    let combinations: Vec<_> = (1..n).fold(
        idents
            .iter()
            .map(|ident1| {
                idents
                    .iter()
                    .filter_map(move |ident2| combine_idents(ident1, ident2))
            })
            .flatten()
            .collect(),
        |acc, _| {
            acc.iter()
                .map(|ident1| {
                    idents
                        .iter()
                        .filter_map(move |ident2| combine_idents(ident1, ident2))
                })
                .flatten()
                .collect()
        },
    );
    combinations
}
