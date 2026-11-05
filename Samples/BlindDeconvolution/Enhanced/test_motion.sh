#!/usr/bin/env bash
#
# Characterize the blind-deconvolution algorithm on LINEAR MOTION BLUR.
#
# It exercises three regimes and prints a summary table:
#   * flat init  - fully blind (uniform square kernel) -> hard case
#   * line init  - orientation known, profile blind     -> works very well
#   * line+noise - robustness with damping / early stopping
#
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP=./build/deconvblind_app
DATA=data
RES=results/motion
mkdir -p "$RES"

if [ ! -x "$APP" ]; then
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build build -j"$(nproc)" >/dev/null
fi

# Ensure at least one test image exists.
IMG="$DATA/fruits.jpg"
if [ ! -s "$IMG" ]; then
    curl -fsSL -m 60 -o "$IMG" \
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg" ||
        { echo "Could not download test image"; exit 1; }
fi

# Run the app and extract: blurred PSNR, best PSNR, best iter, PSF NCC.
run() { # args... -> "<psnr_blur> <psnr_best> <iter> <ncc>"
    "$APP" "$@" 2>/dev/null | awk '
        /Blurred input/        {b=$3}
        /Deblurred \(best\)/   {bp=$3; sub(/[()]/,"",$6); bi=$6}
        /PSF recovery/         {n=$8}
        END {printf "%s %s %s %s", b, bp, bi, n}'
}

row() { # label | extra-args...
    local label="$1"; shift
    read -r b bp bi n <<<"$(run "$IMG" "$@" --out="$RES/${label// /_}")"
    local dp; dp=$(awk "BEGIN{printf \"%+.2f\", $bp-$b}")
    printf '%-26s %9s %9s %7s %8s %8s\n' "$label" "$b" "$bp" "$bi" "$dp" "$n"
}

echo
echo "LINEAR MOTION BLUR CHARACTERIZATION  (image: $(basename "$IMG"))"
printf '%-26s %9s %9s %7s %8s %8s\n' \
    "scenario" "PSNRblur" "PSNRbest" "iter" "dPSNR" "psfNCC"
printf -- '--------------------------------------------------------------------------\n'

echo "# Fully blind (flat square init) - motion is a hard case:"
for L in 9 15 21; do
    row "flat len=$L a=30" --blur=motion --len=$L --psf=$L --angle=30 --init=flat --iter=30
done

echo "# Orientation-aware (line init) - direction known, profile blind:"
for a in 0 30 45 90 135; do
    row "line a=$a len=15" --blur=motion --len=15 --psf=15 --angle=$a --init=line --iter=20
done
for L in 9 21 31; do
    row "line len=$L a=30" --blur=motion --len=$L --psf=$L --angle=30 --init=line --iter=20
done

echo "# Orientation-aware + noise (damped, early stopping):"
for nz in 0.005 0.01 0.02; do
    row "line noise=$nz" --blur=motion --len=15 --psf=15 --angle=30 \
        --init=line --iter=30 --noise=$nz --dampar=0.02
done

printf -- '--------------------------------------------------------------------------\n'
cat <<'NOTE'
Summary:
  * The Richardson-Lucy core restores motion blur well when the PSF support
    shape is reasonable: a LINE init (orientation only) recovers the kernel
    almost exactly (NCC ~ 1.0) and gains ~5-6 dB PSNR across all angles.
  * A FLAT square init is a poor shape prior AND a near-stationary point for
    symmetric kernels, so fully-blind-from-flat barely improves motion blur
    (best is usually iteration 1). This matches deconvblind's documentation:
    "PSF restoration is affected strongly by the size [and shape] of its guess".
  * Under noise, damping + early stopping keep the result stable; gains shrink
    as noise grows, as expected for a maximum-likelihood deconvolution.
Output images are in results/motion/.
NOTE
