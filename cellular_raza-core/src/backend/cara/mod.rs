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

impl Drop for Runner {
    fn drop(&mut self) {
        unsafe {
            drop_runner(self);
        }
    }
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
struct Agent {
    id: UID,
    position: Position,
    velocity: Velocity,
}

#[repr(C)]
struct AgentContainer {
    agent: Agent,
}

unsafe extern "C" {
    fn new_runner(agents: *const Agent, n_agents: usize) -> *mut Runner;
    fn drop_runner(runner: *mut Runner);
    fn do_compute();
    fn update_positions(runner: &mut Runner);
    fn print_positions(runner: &Runner);
}

/// Empty main routine to showcase how to include cuda files
pub fn run() {
    let n_agents = 10;
    let agents: Vec<_> = (0..n_agents)
        .map(|n| Agent {
            id: n,
            position: Position { p: [n as f32; 3] },
            velocity: Velocity { v: [0.; 3] },
        })
        .collect();

    let runner = unsafe { new_runner(agents.as_ptr(), agents.len()) };
    /* let mut runner = Runner {
        uid_counter: 0,
        n_agents,
        agents_up_to_date: false,
        // TODO this needs to be changed. What if there are not agents present to index?
        agents: &mut (agents[0]),
        positions: &mut 0.0,
        velocities: &mut 0.0,
        forces: &mut 0.0,
    };*/

    unsafe {
        update_positions(&mut *runner);
        print_positions(&*runner);
        do_compute();
    }
}

#[test]
fn test_main() {
    run();
}

#[test]
fn test_cudarc() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::*;
    const SIN_KERNEL: &str = include_str!("kernel.cu");
    // Get a stream for GPU 0
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // copy a rust slice to the device
    let inp = stream.memcpy_stod(&[1.0f32; 100])?;

    // or allocate directly
    let mut out = stream.alloc_zeros::<f32>(100)?;

    let ptx = cudarc::nvrtc::compile_ptx(SIN_KERNEL)?;
    // Dynamically load it into the device
    let module = ctx.load_module(ptx)?;
    let sin_kernel = module.load_function("sin_kernel")?;

    // let sin_kernel = dev.get_func("my_module", "sin_kernel").unwrap();
    let mut builder = stream.launch_builder(&sin_kernel);
    builder.arg(&mut out);
    builder.arg(&inp);
    builder.arg(&100usize);
    unsafe { builder.launch(cudarc::driver::LaunchConfig::for_num_elems(100)) }?;

    let out_host: Vec<f32> = stream.memcpy_dtov(&out)?;
    for q in out_host.into_iter() {
        assert!((q - f32::sin(1.).abs() < 0.001));
    }
    Ok(())
}
