OLD_VERSION='0\.3\.1-rc1'
NEW_VERSION='0\.3\.1'

for file in $(grep -lr $OLD_VERSION --exclude-dir target *); do
    sed -i "s/$OLD_VERSION/$NEW_VERSION/" $file
done
