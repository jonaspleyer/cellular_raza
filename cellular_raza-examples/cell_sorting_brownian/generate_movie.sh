# Generated with ffmpeg 6.1.1

FOLDERS=($(ls "out/cell_sorting"))
LAST_FOLDER=${FOLDERS[-1]}

# python plot.py

ffmpeg \
    -y \
    -pattern_type glob \
    -i "out/cell_sorting/"$LAST_FOLDER'/images/*.png' \
    -c:v libx264 \
    -r 60\
    -pix_fmt yuv420p output.mp4
