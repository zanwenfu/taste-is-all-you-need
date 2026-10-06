#!/bin/sh
# Render the README banner (light and dark) and the social card from banner.html
# with Chrome's own headless screenshot; no other dependency.
#   docs/banner/render.sh            # writes docs/img/banner-{light,dark}.png, docs/img/social-card.png
# CHROME may name another Chrome or Chromium binary. The fonts are Geist, under
# the SIL Open Font License (LICENSES/Geist-OFL.txt).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out="$here/../img"
CHROME=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
[ -x "$CHROME" ] || CHROME=$(command -v chromium || command -v google-chrome || true)
[ -n "$CHROME" ] || { echo "no Chrome found; set CHROME" >&2; exit 1; }

shot() { # variant width height scale file
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --allow-file-access-from-files \
    --force-device-scale-factor="$4" --window-size="$2,$3" --virtual-time-budget=6000 \
    --screenshot="$out/$5" "file://$here/banner.html?variant=$1" 2>/dev/null
  echo "wrote docs/img/$5"
}
# The banner at twice its layout size, for sharp text on high-density screens.
shot light 1280 400 2 banner-light.png
shot dark 1280 400 2 banner-dark.png
# GitHub's social preview: 1280 x 640, under 1 MB.
shot og 1280 640 1 social-card.png
