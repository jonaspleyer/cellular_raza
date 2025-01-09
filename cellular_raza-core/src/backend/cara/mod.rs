//! 🐯 (Placeholder) GPU-centered backend using [CUDA](https://developer.nvidia.com/cuda-toolkit)
//!
//! This backend is currently not developed and only here to serve as a placeholder.

#[repr(C)]
struct Runner {
    n_agents: u32,
    agents: *mut Agent,
}

type UID = u32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Position {
    p: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Velocity {
    v: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Force {
    f: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Agent {
    id: UID,
    position: Position,
    velocity: Velocity,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AgentContainer {
    agent: Agent,
    current_force: Force,
}

extern "C" {
    fn do_compute();
    fn update_positions(runner: &mut Runner);
    fn print_positions(runner: &Runner);
}

/// Empty main routine to showcase how to include cuda files
pub fn run() {
    let n_agents = 10;
    let mut agents: Vec<_> = (0..n_agents)
        .map(|n| Agent {
            id: n,
            position: Position { p: [n as f32; 3] },
            velocity: Velocity { v: [0.; 3] },
        })
        .collect();

    let mut runner = Runner {
        n_agents,
        // TODO this needs to be changed. What if there are not agents present to index?
        agents: &mut (agents[0]),
    };
    println!("{:p}", runner.agents);

    unsafe {
        update_positions(&mut runner);
        print_positions(&runner);
        do_compute();
    }
    assert!(false);
}

#[test]
fn test_main() {
    run();
}
