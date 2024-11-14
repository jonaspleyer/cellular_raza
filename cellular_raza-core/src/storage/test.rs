#[test]
fn store_load_json() -> Result<(), Box<dyn std::error::Error>> {
    use crate::storage::*;
    let builder = StorageBuilder::new()
        .priority([StorageOption::SerdeJson])
        .init();
    let mut storage_manager =
        StorageManager::<usize, BatchSaveFormat<usize, f64>>::open_or_create(builder, 0)?;
    storage_manager.store_single_element(
        0,
        &22,
        &BatchSaveFormat {
            data: vec![CombinedSaveFormat {
                identifier: 22,
                element: 3.0,
            }],
        },
    )?;
    let _res = storage_manager.load_single_element(0, &22)?;
    // TODO
    // match res {
    //     Some(x) => assert_eq!(x.data[0].element, 3.0),
    //     None => assert!(false),
    // }
    Ok(())
}

#[allow(unused)]
macro_rules! open_storage_interface(
    (@internal_path) => {{
        use tempdir::TempDir;
        let dir = TempDir::new("tempdir").unwrap();
        let location = dir.path().join(
            concat!("tempdir_", stringify!($interface_name))
        );
        location
    }};
    ($interface_name:ident) => {{
        let location = open_storage_interface!(@internal_path);
        let interface = StorageWrapper(
            $interface_name::open_or_create(&location, 0).unwrap()
        );
        interface
    }};
    ($interface_name:ident -> $($storage_interface:ident),+) => {{
        let location = open_storage_interface!(@internal_path);
        let mut __index = 0;
        $(
            $storage_interface = StorageWrapper(
                $interface_name::open_or_create(&location, __index).unwrap()
            );
            __index += 1;
        )*
    }};
);

// TODO extend this to test all functions
macro_rules! test_storage_interface(
    ($interface_name:ident, $module_name:ident) => {
        #[cfg(test)]
        mod $module_name {
            use crate::storage::*;

            #[test]
            fn store_multiple_load_multiple_2_interfaces() {
                // Open storage interfaces
                let mut interface_0;
                let mut interface_1;
                open_storage_interface!($interface_name -> interface_0, interface_1);

                // Generate data
                let generate_elements = |low: usize, high: usize| {
                    (low..high).map(|i| (i, i as f64))
                    .collect::<std::collections::HashMap<_, _>>()
                };
                let identifiers_elements_0 = generate_elements(0, 10);
                let identifiers_elements_1 = generate_elements(20, 30);
                let iteration = 100;

                // Save data to storage interface
                interface_0
                    .store_batch_elements(iteration, identifiers_elements_0.iter())
                    .unwrap();
                interface_1
                    .store_batch_elements(iteration, identifiers_elements_1.iter())
                    .unwrap();

                // Load results again
                let loaded_elements_0 = interface_0.load_all_elements_at_iteration(iteration)
                    .unwrap();
                let loaded_elements_1 = interface_1.load_all_elements_at_iteration(iteration)
                    .unwrap();
                let mut identifiers_elements = identifiers_elements_0.clone();
                identifiers_elements.extend(identifiers_elements_1);

                // Check that results match
                assert_eq!(identifiers_elements, loaded_elements_0);
                assert_eq!(identifiers_elements, loaded_elements_1);
                assert_eq!(loaded_elements_0, loaded_elements_1);
            }

            #[test]
            fn store_single_load_single_element() {
                let mut interface = open_storage_interface!($interface_name);

                let identifiers_elements = [
                    (30, format!("This is sparta!")),
                    (40, format!("This is not")),
                    (35, format!("Something else")),
                ];

                let iteration = 350;
                for (identifier, element) in identifiers_elements {
                    interface.store_single_element(iteration, &identifier, &element).unwrap();
                    let element_loaded = interface.load_single_element(
                        iteration,
                        &identifier
                    ).unwrap().unwrap();
                    assert_eq!(element, element_loaded);
                }
            }

            #[test]
            fn store_single_load_multiple_elements() {
                let mut interface = open_storage_interface!($interface_name);
                let identifiers_elements = [
                    (format!("hasta"), (3, 12)),
                    (format!("la"), (3, 11)),
                    (format!("vista"), (45, 33))
                ];
                let iteration = 1001;
                for (identifier, element) in identifiers_elements.iter() {
                    interface.store_single_element(iteration, identifier, element).unwrap();
                }
                let loaded_elements = interface.load_all_elements_at_iteration(iteration).unwrap();
                for (identifier, element) in identifiers_elements {
                    let loaded_element = loaded_elements[&identifier];
                    assert_eq!(loaded_element, element);
                }
            }

            #[test]
            fn store_multiple_load_single_elements() {
            }
        }
    }
);

test_storage_interface!(JsonStorageInterface, json_tests);
test_storage_interface!(RonStorageInterface, ron_tests);
