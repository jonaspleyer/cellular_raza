use nalgebra::Vector2;

use std::env;

mod cell_properties;
mod plotting;
use plotting::*;
use cell_properties::*;

use cellular_raza::storage::sled_database::*;

// pub type MyVoxelType = CartesianCuboidVoxel2Reactions4;
// pub type MyVoxelBox = VoxelBox<[usize; 2], MyVoxelType, MyCellType, Vector2<f64>, Vector2<f64>, Vector2<f64>, ReactionVector, ReactionVector, ReactionVector>;


fn main() {
    let args: Vec<String> = env::args().collect();
    
    let storage_interface = SledStorageInterface::<[usize; 2], i32>::open_or_create("test_db".into()).unwrap();
    storage_interface.store_single_element(0, [1,2], 1234).unwrap();
    storage_interface.store_single_element(1, [1,2], 2345).unwrap();
    storage_interface.store_single_element(2, [1,2], 3456).unwrap();
    storage_interface.store_single_element(3, [1,2], 4567).unwrap();

    storage_interface.store_single_element(2, [0,3], 010).unwrap();
    storage_interface.store_single_element(3, [0,3], 101).unwrap();
    storage_interface.store_single_element(4, [0,3], 110).unwrap();

    storage_interface.store_batch_elements(4, vec![
        ([1,2], 8888),
        ([0,3], 0000),
        ([1,1], 0011)
    ]).unwrap();
    
    let all_elements = storage_interface.load_all_elements().unwrap();
    println!("All elements");
    for (key, element) in all_elements.iter() {
        println!("key: {key:?} element: {element:?}")
    }
    
    let elements_at_iteration = storage_interface.load_all_elements_at_iteration(2).unwrap();
    println!("Elements at iter 2");
    for (key, element) in elements_at_iteration.iter() {
        println!("key: {key:?} element: {element:?}")
    }

    let element_history = storage_interface.load_element_history([1,2]).unwrap();
    println!("Element histories");
    for (iteration, entry) in element_history.iter() {
        println!("{:?} {:?}", iteration, entry)
    }
}
