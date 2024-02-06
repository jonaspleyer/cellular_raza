generate_docs() {
    TARGETS=(
        cellular_raza
        cellular_raza-benchmarks
        cellular_raza-building-blocks
        cellular_raza-concepts
        cellular_raza-concepts-derive
        cellular_raza-core
        cellular_raza-core-proc-macro
    )

    cargo clean --doc

    for target in ${TARGETS[@]}; do
        cd $target
        cargo +nightly-2024-01-01 rustdoc --all-features -- --cfg doc_cfg
        cd ..
    done
}

build_website() {
    # Copy generated code to website folder
    cp -r target/doc cellular_raza-homepage/cr_doc

    # Swap old for new folder
        cd cellular_raza-homepage
    rm -rf content/docs
    mv cr_doc content/docs

    hugo -d public_html
    cd ..
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo " -h --help        Display this help message"
    echo " -d --doc         Generate rust documentation"
    echo " -w --website     Build website"
    echo " -u --upload      Upload to server"
    echo " -a --all         Do a full build with upload. Same as -d -w -u"
}

upload() {
    cd cellular_raza-homepage
    cwd="$PWD"
    # This command can also be used to simply recursively copy all files
    # id does not filter duplicates and thus takes a long time
    # scp -r public_html celluld@www139.your-server.de:/

    sftp celluld@www139.your-server.de<<EOF
    put -R $cwd/public_html
    exit
EOF
    cd ..
}

handle_options() {
    while [ $# -gt 0 ]; do
        case $1 in
            -d | --doc)
                generate_docs
                exit 0
                ;;
            -w | --website)
                build_website
                exit 0
                ;;
            -u | --upload)
                upload
                exit 0
                ;;
            -a | --all)
                generate_docs
                build_website
                upload
                exit 0
                ;;
            -h | --help)
                usage
                exit 0
                ;;
            \?)
                usage
                exit 1
                ;;
        esac
    done
}

handle_options "$@"

