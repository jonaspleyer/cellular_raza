//! 🐺 (Placeholder) Cross-platform GPU-centered backend using
//! [opencl](https://docs.rs/opencl3/latest/opencl3/)

struct OpenCLComm {
    device: opencl3::device::Device,
    context: opencl3::context::Context,
    queue: opencl3::command_queue::CommandQueue,
}

impl OpenCLComm {
    fn open_connection() -> opencl3::Result<Self> {
        // TODO we should not be selecting the first device by default
        let device_id = *opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU)?
            .first()
            .ok_or(opencl3::error_codes::CL_DEVICE_NOT_AVAILABLE)?;
        let device = opencl3::device::Device::new(device_id);

        // Create a Context on an OpenCL device
        let context = opencl3::context::Context::from_device(&device)?;

        // Create a command_queue on the Context's device
        let queue = opencl3::command_queue::CommandQueue::create_default(
            &context,
            opencl3::command_queue::CL_QUEUE_PROFILING_ENABLE,
        )?;

        Ok(Self {
            device,
            context,
            queue,
        })
    }

    fn build_program(
        &self,
        source: &str,
        options: Option<&str>,
    ) -> opencl3::Result<opencl3::program::Program> {
        let options = options.unwrap_or("");

        // Build the OpenCL program source and create the kernel.
        Ok(
            opencl3::program::Program::create_and_build_from_source(&self.context, source, options)
                .map_err(|_| opencl3::error_codes::CL_BUILD_PROGRAM_FAILURE)?,
        )
    }

    fn create_kernel(
        &self,
        program: &opencl3::program::Program,
        kernel_name: &str,
    ) -> opencl3::Result<opencl3::kernel::Kernel> {
        opencl3::kernel::Kernel::create(&program, kernel_name)
    }
}

fn my_kernel_saxpy_float(z: &mut [f32], x: &[f32], y: &[f32], a: f32) {
    z.iter_mut()
        .zip(x.iter().zip(y.iter()))
        .for_each(|(z, (x, y))| {
            *z = a * x + y;
        })
}

fn run_main(n_agents: usize) -> opencl3::Result<()> {
    let comm = OpenCLComm::open_connection()?;
    let program = comm.build_program(include_str!("saxpy.cl"), None)?;
    let kernel = comm.create_kernel(&program, "saxpy_float")?;

    /////////////////////////////////////////////////////////////////////
    // Compute data
    use opencl3::kernel::ExecuteKernel;
    use opencl3::memory::Buffer;
    use opencl3::memory::{CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
    use opencl3::types::cl_float;
    use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING};

    let array_size = n_agents * 8 * 3;
    // The input data
    let val1s = vec![0.0; array_size];
    let val2s = vec![1.0; array_size];

    let t4 = std::time::Instant::now();
    // Create OpenCL device buffers
    let mut x = unsafe {
        Buffer::<cl_float>::create(
            &comm.context,
            CL_MEM_READ_ONLY,
            array_size,
            core::ptr::null_mut(),
        )?
    };
    let mut y = unsafe {
        Buffer::<cl_float>::create(
            &comm.context,
            CL_MEM_READ_ONLY,
            array_size,
            core::ptr::null_mut(),
        )?
    };
    let z = unsafe {
        Buffer::<cl_float>::create(
            &comm.context,
            CL_MEM_WRITE_ONLY,
            array_size,
            core::ptr::null_mut(),
        )?
    };
    let t4 = t4.elapsed().as_micros();

    let t = std::time::Instant::now();
    // Blocking write
    let _x_write_event = unsafe {
        comm.queue
            .enqueue_write_buffer(&mut x, CL_BLOCKING, 0, &val1s, &[])?
    };

    // Non-blocking write, wait for y_write_event
    let y_write_event = unsafe {
        comm.queue
            .enqueue_write_buffer(&mut y, CL_NON_BLOCKING, 0, &val2s, &[])?
    };
    let t0 = t.elapsed().as_micros();

    // a value for the kernel function
    let a: cl_float = 300.0;

    let mut z_my = vec![1.0; array_size];

    // Use the ExecuteKernel builder to set the kernel buffer and
    // cl_float value arguments, before setting the one dimensional
    // global_work_size for the call to enqueue_nd_range.
    // Unwraps the Result to get the kernel execution event.
    let t1 = std::time::Instant::now();
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&z)
            .set_arg(&x)
            .set_arg(&y)
            .set_arg(&a)
            .set_global_work_size(array_size)
            .set_wait_event(&y_write_event)
            .enqueue_nd_range(&comm.queue)?
    };
    let t1 = t1.elapsed().as_micros();

    let t2 = std::time::Instant::now();
    my_kernel_saxpy_float(&mut z_my, &val1s, &val2s, a);
    let t2 = t2.elapsed().as_micros();

    println!("Alloc: {t4}µs Copy: {t0}µs OpenCL: {t1}µs Rust: {t2}µs");
    let mut z_out = vec![0.0; array_size];
    let z_read = unsafe {
        comm.queue
            .enqueue_read_buffer(&z, CL_BLOCKING, 0, &mut z_out, &[])
    }?;
    assert_eq!(z_my.iter().sum::<f32>(), z_out.iter().sum::<f32>());
    Ok(())
}

#[test]
fn test_run_main() {
    for n in [100, 200, 400, 800, 1200, 1600, 2000] {
        run_main(n).unwrap();
    }
}
