# Generated with ffmpeg 6.1.1

ffmpeg \
    -y \
    -pattern_type glob \
    -i 'out/kidney_organoid_model/images/*.png' \
    -c:v libx264 \
    -pix_fmt yuv420p output.mp4
