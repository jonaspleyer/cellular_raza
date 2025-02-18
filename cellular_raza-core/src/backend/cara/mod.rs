//! 🐯 (Placeholder) GPU-centered backend using [CUDA](https://developer.nvidia.com/cuda-toolkit)
//!
//! This backend is currently in a very early development state and thus not usably at this time.

#[repr(C)]
struct Runner {
    uid_counter: u64,
    n_agents: u64,
    agents_up_to_date: bool,
    agents: *mut AgentContainer,
    positions: *mut f32,
    velocities: *mut f32,
    forces: *mut f32,
}

type UID = u64;

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
        .map(|n| AgentContainer {
            agent: Agent {
                id: n,
                position: Position { p: [n as f32; 3] },
                velocity: Velocity { v: [0.; 3] },
            },
        })
        .collect();

    let mut runner = Runner {
        uid_counter: 0,
        n_agents,
        agents_up_to_date: false,
        // TODO this needs to be changed. What if there are not agents present to index?
        agents: &mut (agents[0]),
        positions: &mut 0.0,
        velocities: &mut 0.0,
        forces: &mut 0.0,
    };

    unsafe {
        update_positions(&mut runner);
        print_positions(&runner);
        do_compute();
    }
}

#[test]
fn test_main() {
    run();
}

#[test]
fn test_cudarc() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::LaunchAsync;
    const SIN_KERNEL: &str = include_str!("kernel.cu");
    let dev = cudarc::driver::CudaDevice::new(0)?;

    // allocate buffers
    let inp = dev.htod_copy(vec![1.0f32; 100])?;
    let mut out = dev.alloc_zeros::<f32>(100)?;
    let ptx = cudarc::nvrtc::compile_ptx(SIN_KERNEL)?;
    dev.load_ptx(ptx, "my_module", &["sin_kernel"])?;
    let sin_kernel = dev.get_func("my_module", "sin_kernel").unwrap();
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(100);
    unsafe { sin_kernel.launch(cfg, (&mut out, &inp, 100usize)) }?;
    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out)?;
    for i in 0..100 {
        assert_eq!(out_host[i], 1f32.sin());
    }
    // assert_eq!(out_host, [1.0; 100].map(f32::sin));
    Ok(())
}
