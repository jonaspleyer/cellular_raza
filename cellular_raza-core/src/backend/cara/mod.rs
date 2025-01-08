//! 🐯 (Placeholder) GPU-centered backend using [CUDA](https://developer.nvidia.com/cuda-toolkit)
//!
//! This backend is currently not developed and only here to serve as a placeholder.

extern "C" {
    fn do_compute();
}

/// Empty main routine to showcase how to include cuda files
pub fn run() {
    unsafe {
        do_compute();
    }
}

#[test]
fn test_main() {
    run();
}
