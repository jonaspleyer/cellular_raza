use cellular_raza::backend::cpu_os_threads::prelude::*;

trait SimulationSetup {
    type Domain;
    type Cell;
}

struct MySetup {
    domain: CartesianCuboid3,
    cell: ModularCell<
        MechanicsModel3D,
        NoInteraction,
        NoCycle,
        NoExtracellularGradientSensing,
        NoCellularreactions,
    >,
}

impl SimulationSetup for MySetup {
    type Domain = CartesianCuboid3;
    type Cell = ModularCell<
        MechanicsModel3D,
        NoInteraction,
        NoCycle,
        NoExtracellularGradientSensing,
        NoCellularreactions,
    >;
}

fn main() {}
