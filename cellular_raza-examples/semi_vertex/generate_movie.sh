# Generated with ffmpeg 6.1.1

FOLDERS=($(ls "out/semi_vertex"))
LAST_FOLDER=${FOLDERS[-1]}

python plot.py

ffmpeg \
    -y \
    -pattern_type glob \
    -i "out/semi_vertex/"$LAST_FOLDER'/images/*.png' \
    -c:v libx264 \
    -pix_fmt yuv420p "out/semi_vertex/"$LAST_FOLDER'/output.mp4'

firefox "out/semi_vertex/"$LAST_FOLDER'/output.mp4'
