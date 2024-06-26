generate_docs() {
    TARGETS=(
        cellular_raza
        cellular_raza-building-blocks
        cellular_raza-concepts
        cellular_raza-core
    )

    cargo clean --doc

    for target in ${TARGETS[@]}; do
        cargo +nightly-2024-06-01 rustdoc \
            -p $target \
            --all-features --\
            --cfg docsrs\
            --theme cellular_raza-homepage/static/hextra.css \
            --default-theme hextra \
            --html-in-header cellular_raza-homepage/custom_navbar.html \
            --html-in-header .docs-header.html
    done

    # Swap old for new folder
    rm -rf cellular_raza-homepage/static/docs
    cp -r target/doc cellular_raza-homepage/static/docs
}

build_website() {
    cd cellular_raza-homepage
    hugo -d public_html
    cd ..
}

movie() {
    # Generate the movie
    gource \
    -s .15 \
    --default-user-image cellular_raza-homepage/static/JonasPleyer-circle.png \
    -1280x720 \
    --auto-skip-seconds .05 \
    --multi-sampling \
    --stop-at-end \
    --key \
    --highlight-users \
    --date-format "%d/%m/%y" \
    --hide mouse,filenames \
    --file-idle-time 0 \
    --max-files 0  \
    --background-colour 000000 \
    --font-size 25 \
    --output-framerate 60 \
    --disable-auto-rotate \
    --max-user-speed 30 \
    --dir-name-depth 1 \
    --highlight-dirs \
    --dir-name-position 1.0 \
    --file-filter "(cellular_raza-homepage|cellular_raza_book|cellular_raza-book)" \
    --output-ppm-stream - \
    | ffmpeg -y -r 60 -f image2pipe -pix_fmt yuv420p -c:v libx264 -y -vcodec ppm -i - -b 65536K output.mp4
    # Compress the movie
    ffmpeg -i output.mp4 -pix_fmt yuv420p -y cellular_raza-development-gource.mp4
    # Move it to the hompage folder
    mkdir -p cellular_raza-homepage/static/internals/
    mv cellular_raza-development-gource.mp4 cellular_raza-homepage/static/internals/
}

scc_table() {
    scc \
        --exclude-dir cellular_raza-homepage \
        --exclude-dir target \
        --no-size \
        -f html-table \
        -s lines \
        -o cellular_raza-homepage/static/internals/scc-table.html
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo " -h --help        Display this help message"
    echo " -d --doc         Generate rust documentation"
    echo " -w --website     Build website"
    echo " -m --movie       Generate Movie of cellular_raza development"
    echo "    --scc         Create scc table"
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
            -m | --movie)
                movie
                exit 0
                ;;
            --scc)
                scc_table
                exit 0
                ;;
            -a | --all)
                generate_docs
                build_website
                scc_table
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

