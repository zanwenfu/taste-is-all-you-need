#!/bin/sh
# Renders the README banner and the social card from banner.html into docs/img:
#   banner.avif, banner-dark.avif  the banner, moving: a 16-second loop at 12 fps
#   banner.png                     its still, for visitors who turn motion off
#   social-card.png                GitHub's social preview, 1280 x 640
# Needs Node (run `npm install` here once), Google Chrome, and ffmpeg built with
# SVT-AV1 (Homebrew's ffmpeg has it). A full build takes a few minutes. The fonts
# are Geist, under the SIL Open Font License (LICENSES/Geist-OFL.txt).
set -e
cd "$(dirname "$0")"
img=../img
rm -rf out && mkdir -p out/light out/dark
node render.cjs frames light out/light 192 12
node render.cjs frames dark out/dark 192 12
# AV1 in an AVIF file, full-range sRGB colour, looping forever.
encode() {
  SVT_LOG=1 ffmpeg -hide_banner -loglevel error -y -framerate 12 -i "$1/f_%04d.png" \
    -vf "scale=out_color_matrix=bt709:out_range=full,format=yuv420p,setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709:range=pc" \
    -c:v libsvtav1 -preset 4 -crf 22 -g 192 -svtav1-params "tune=0:scm=1" -loop 0 "$2"
}
encode out/light "$img/banner.avif"
encode out/dark "$img/banner-dark.avif"
node render.cjs still light out/still.png
ffmpeg -hide_banner -loglevel error -y -i out/still.png -compression_level 9 -pred mixed "$img/banner.png"
node render.cjs still og "$img/social-card.png"
echo "wrote docs/img/banner.avif, banner-dark.avif, banner.png and social-card.png"
