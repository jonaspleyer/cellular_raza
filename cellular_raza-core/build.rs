fn main() {
    // Exit early without doing anything if we are building for docsrs
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    #[cfg(feature = "cara")]
    {
        println!("cargo:rerun-if-changed=src/backend/cara");
        cc::Build::new()
        // Switch to CUDA C++ library compilation using NVCC.
        .cuda(true)
        .ccbin(false)
        .cudart("static")
        // Generate code in parallel
        .flag("-t0")
        .file("src/backend/cara/cara.cu")
        .compile("bar");
    }
}
