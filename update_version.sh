OLD_VERSION=0\.2\.1
NEW_VERSION=0\.2\.2

for file in $(grep -lr $OLD_VERSION --exclude-dir target *); do
    sed -i "s/$OLD_VERSION/$NEW_VERSION/" $file
done
