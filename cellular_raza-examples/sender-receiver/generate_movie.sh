# Generated with ffmpeg 6.1.1

OUT_FOLDER="out/sender_receiver"
FOLDERS=($(ls $OUT_FOLDER))
LAST_FOLDER=${FOLDERS[-1]}

python plot.py

ffmpeg \
    -y \
    -pattern_type glob \
    -i $OUT_FOLDER/$LAST_FOLDER'/images/*.png' \
    -c:v libx264 \
    -pix_fmt yuv420p output.mp4

