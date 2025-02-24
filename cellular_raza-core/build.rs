fn main() {
    // Exit early without doing anything if we are building for docsrs
    if !std::env::var("DOCS_RS").is_ok() {
        return;
    }

    #[cfg(feature = "cara")] {
    println!("cargo:rerun-if-changed=src/backend/cara");
    cc::Build::new()
        // Switch to CUDA C++ library compilation using NVCC.
        .cuda(true)
        .ccbin(false)
        .cudart("static")
        // Generate code for Maxwell (GTX 970, 980, 980 Ti, Titan X).
        .flag("-gencode")
        .flag("arch=compute_52,code=sm_52")
        // Generate code for Maxwell (Jetson TX1).
        .flag("-gencode")
        .flag("arch=compute_53,code=sm_53")
        // Generate code for Pascal (GTX 1070, 1080, 1080 Ti, Titan Xp).
        .flag("-gencode")
        .flag("arch=compute_61,code=sm_61")
        // Generate code for Pascal (Tesla P100).
        .flag("-gencode")
        .flag("arch=compute_60,code=sm_60")
        // Generate code for Pascal (Jetson TX2).
        .flag("-gencode")
        .flag("arch=compute_62,code=sm_62")
        // Generate code in parallel
        .flag("-t0")
        .file("src/backend/cara/cara.cu")
        .compile("bar");
    }
}
