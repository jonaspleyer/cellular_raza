OLD_VERSION='0\.4\.0'
NEW_VERSION='0\.4\.0-rc1'

for file in $(grep -lr $OLD_VERSION --exclude-dir target *); do
    sed -i "s/$OLD_VERSION/$NEW_VERSION/" $file
done
