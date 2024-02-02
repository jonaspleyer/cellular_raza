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
	cargo cr_doc
	cd ..
done

cp -r target/doc cellular_raza-homepage/cr_doc
cd cellular_raza-homepage

rm -rf content/docs
mv cr_doc content/docs

hugo -d public_html

# This command can also be used to simply recursively copy all files
# id does not filter duplicates and thus takes a long time
# scp -r public_html celluld@www139.your-server.de:/

cwd="$PWD"

sftp celluld@www139.your-server.de <<EOF
put -a $cwd/public_html
exit
EOF

