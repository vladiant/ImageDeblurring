#!/usr/bin/env bash
#
# Download public-domain test images and verify the blind deconvolution
# implementation across several blur scenarios. Prints a summary table and
# exits non-zero if any scenario fails to improve the image.
#
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP=./build/deconvblind_app
DATA=data
RES=results
mkdir -p "$DATA" "$RES"

if [ ! -x "$APP" ]; then
    echo "Building..."
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build build -j"$(nproc)" >/dev/null
fi

# Public, freely redistributable sample images (OpenCV sample data).
BASE="https://raw.githubusercontent.com/opencv/opencv/master/samples/data"
declare -A IMAGES=(
    [baboon]="baboon.jpg"
    [fruits]="fruits.jpg"
    [butterfly]="butterfly.jpg"
)
for name in "${!IMAGES[@]}"; do
    f="$DATA/${IMAGES[$name]}"
    if [ ! -s "$f" ]; then
        echo "Downloading ${IMAGES[$name]}..."
        curl -fsSL -m 60 -o "$f" "$BASE/${IMAGES[$name]}" ||
            echo "  (download failed; skipping $name)"
    fi
done

# Scenario: label | image | extra args. The flat-init size is matched to the
# blur support, as recommended for deconvblind.
SCENARIOS=(
    "gauss_broad|fruits|--blur=gauss --psf=11 --sigma=5 --iter=20"
    "gauss_broad|baboon|--blur=gauss --psf=11 --sigma=5 --iter=20"
    "gauss_broad|butterfly|--blur=gauss --psf=11 --sigma=5 --iter=20"
    "gauss_flat|fruits|--blur=gauss --psf=9 --sigma=8 --iter=20"
    "gauss_noisy|fruits|--blur=gauss --psf=11 --sigma=5 --iter=20 --noise=0.01 --dampar=0.02"
)

printf '\n%-14s %-10s %10s %10s %10s %6s\n' \
    "SCENARIO" "IMAGE" "PSNR_blur" "PSNR_best" "dPSNR" "PASS"
printf -- '---------------------------------------------------------------------\n'

fail=0
for s in "${SCENARIOS[@]}"; do
    IFS='|' read -r label image args <<<"$s"
    img="$DATA/${IMAGES[$image]}"
    [ -s "$img" ] || continue
    out="$RES/${label}_${image}"
    log=$($APP "$img" $args --out="$out" 2>&1)
    pb=$(echo "$log" | awk '/Blurred input/{print $3}')
    pbest=$(echo "$log" | awk '/Deblurred \(best\)/{print $3}')
    dp=$(awk "BEGIN{printf \"%.2f\", $pbest-$pb}")
    if echo "$log" | grep -q "RESULT: PASS"; then
        pass="YES"
    else
        pass="NO"
        fail=1
    fi
    printf '%-14s %-10s %10s %10s %10s %6s\n' \
        "$label" "$image" "$pb" "$pbest" "$dp" "$pass"
done

printf -- '---------------------------------------------------------------------\n'
if [ "$fail" -eq 0 ]; then
    echo "ALL SCENARIOS PASSED. Output images are in $RES/."
else
    echo "SOME SCENARIOS DID NOT IMPROVE (see above)."
fi
exit "$fail"
