#[test]
fn store_load_json() -> Result<(), Box<dyn std::error::Error>> {
    use crate::storage::*;
    let builder = StorageBuilder::new()
        .priority([StorageOption::SerdeJson])
        .init();
    let storage_manager =
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
