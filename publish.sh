crates=(
    cellular_raza-concepts-derive
    cellular_raza-concepts
    cellular_raza-building-blocks
    cellular_raza-core-proc-macro
    cellular_raza-core
    cellular_raza
)

CWD=$(pwd)
for crate in ${crates[@]}; do
    cd $crate && cargo publish -n && cargo publish
    cd $CWD
done
