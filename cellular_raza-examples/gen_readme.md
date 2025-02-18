TARGETS=(
    bacteria_population
    bacterial_branching
    bacterial_rods
    cell_sorting
    getting-started
    organoid_turing_growth
    puzzle
    semi_vertex
    ureter_signalling
)

for f in ${TARGETS[@]};
do
    cp ../README2.md $f/README.md
    sed -i 's/placeholderxx/'$f'/g' $f/README.md
done

exit
