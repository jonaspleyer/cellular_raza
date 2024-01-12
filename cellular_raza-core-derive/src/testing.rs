use super::simulation_aspects::SimulationAspect;

fn idents_overlap(id1: &str, id2: &str) -> bool {
    let id1_segments = id1.split("_").collect::<Vec<_>>();
    let id2_segments = id2.split("_").collect::<Vec<_>>();
    let l1 = id1_segments.len();
    let l2 = id2_segments.len();
    let set = std::collections::HashSet::<&str>::from_iter(
        id1_segments.into_iter().chain(id2_segments.into_iter()),
    );
    set.len() < l1 + l2
}

fn get_communicators(n: usize, idents: Vec<String>) -> String {
    let idents: Vec<(String, Vec<String>)> = idents
        .into_iter()
        .map(|s| (s.to_owned().to_lowercase(), vec![s.to_owned()]))
        .collect();

    fn combine_idents(
        ident1: &(String, Vec<String>),
        ident2: &(String, Vec<String>),
    ) -> Option<(String, Vec<String>)> {
        if idents_overlap(&ident1.0, &ident2.0) {
            return None;
        }
        let name = format!("{}_{}", ident1.0, ident2.0);
        let mut list = ident1.1.clone();
        let list2 = ident2.1.clone();
        list.extend(list2);
        let list = list.into_iter().map(|s| s.into()).collect();
        Some((name, list))
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

    let mut full_output = "".to_owned();

    for (name, list) in combinations {
        let list_formatted = list.into_iter().fold("".to_owned(), |mut acc, entry| {
            let pre = if acc.len() == 0 { "" } else { ", " };
            acc.push_str(&format!("{}{}", pre, entry));
            acc
        });
        let output = format!(
            "
        /// ```
        /// use cellular_raza_core_derive::build_communicator;
        /// build_communicator!(
        ///     name: __MyComm,
        ///     aspects: [{}],
        ///     path: cellular_raza_core::backend::chili::simulation_flow
        /// );
        /// ```
        #[allow(non_snake_case)]
        fn {} () {{}}
        ",
            list_formatted, name
        );
        full_output.push_str(&output);
    }
    full_output
}

pub fn get_all_communicators() -> String {
    let idents = SimulationAspect::get_aspects_strings();
    let mut full_output = "".to_owned();
    for n in 0..idents.len() {
        let output = get_communicators(n + 1, idents.clone());
        full_output.push_str(&output);
    }
    full_output
}
