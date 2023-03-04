use std::env;

mod cell_properties;
mod plotting;

use cell_properties::{MyVoxelBox,MyVoxelType};
use plotting::*;

use cellular_raza::{
    concepts::errors::SimulationError,
    pipelines::cpu_os_threads::prelude::{SimulationSetup,CartesianCuboid2},
    plotting::spatial::CreatePlottingRoot,
};


fn main() {
    let args: Vec<String> = env::args().collect();

    let db = typed_sled::open(&args[1]).unwrap();
    let tree_voxels = typed_sled::Tree::<String, Vec<u8>>::open(&db, "voxel_storage");
    let tree_setups = typed_sled::Tree::<String, Vec<u8>>::open(&db, "meta_infos");

    use rayon::prelude::*;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(14).build().unwrap();
    pool.install(|| -> Result<(), SimulationError> {
        // Create progress bar for tree deserialization
        // Deserialize the database tree
        let style = indicatif::ProgressStyle::with_template(cellular_raza::pipelines::cpu_os_threads::config::PROGRESS_BAR_STYLE)?;

        let voxels_at_iter = cellular_raza::storage::sled_database::io::voxels_deserialize_tree::<MyVoxelBox>(&tree_voxels, Some(style.clone())).unwrap();
        let setups_at_iter = cellular_raza::storage::sled_database::io::setup_deserialize_tree::<SimulationSetup<CartesianCuboid2, MyVoxelType>>(&tree_setups, Some(style.clone())).unwrap();

        // Create progress bar for image generation
        println!("Generating Images");
        let bar = indicatif::ProgressBar::new(voxels_at_iter.len() as u64);
        bar.set_style(style);

        voxels_at_iter.into_par_iter()
            .map(|(iteration, voxel_boxes)| -> Result<(), SimulationError> {
            // Create a plotting root
            let filename = format!("out/cells_at_iter_{:010.0}.png", iteration);

            let mut chart = match setups_at_iter.get(&iteration) {
                Some(setup) => setup,
                None => setups_at_iter.values().next().unwrap(),
            }.domain.create_bitmap_root(1500, &filename)?;

            voxel_boxes
                .iter()
                .map(|voxelbox| plot_voxel(&voxelbox.voxel, &mut chart))
                .collect::<Result<(), SimulationError>>()?;

            voxel_boxes
                .iter()
                .map(|voxelbox| voxelbox.cells.iter())
                .flatten()
                .map(|(cellbox, _)| plot_modular_cell(&cellbox.cell, &mut chart))
                .collect::<Result<(), SimulationError>>()?;

            chart.present()?;
            bar.inc(1);
            Ok(())
            
        }).collect::<Result<(), SimulationError>>()?;
        bar.finish();
        Ok(())
    }).unwrap();
}
