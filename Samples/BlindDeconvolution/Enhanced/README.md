# Blind Deconvolution (C++ / OpenCV)

A faithful C++/OpenCV port of MATLAB's Image Processing Toolbox function
[`deconvblind.m`](../deconvblind.m) (and its internal helper `corelucy.m`).

Given a **blurred grayscale image** and an **initial PSF guess**, the algorithm
jointly estimates:

* the **deblurred image**, and
* the **blur kernel (point-spread function, PSF)**

using the *accelerated, damped Richardson–Lucy* maximum-likelihood algorithm
(Biggs & Andrews, *Applied Optics* 36(8), 1997).

## Files

| File | Purpose |
|------|---------|
| [`deconvblind.h`](deconvblind.h) / [`deconvblind.cpp`](deconvblind.cpp) | The reusable blind-deconvolution library |
| [`main.cpp`](main.cpp) | Test/demo harness: simulates a known blur, runs deconvolution, reports metrics and saves images |
| [`CMakeLists.txt`](CMakeLists.txt) | Build configuration |
| [`run_tests.sh`](run_tests.sh) | Downloads public test images and verifies the algorithm over several scenarios |
| [`test_motion.sh`](test_motion.sh) | Characterizes the algorithm's behaviour on linear motion blur |

## Building

Requires OpenCV 4 and CMake ≥ 3.16.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Running

```bash
# Simulate a blur on a sharp image, deblur it and report the quality gain:
./build/deconvblind_app data/fruits.jpg --blur=gauss --psf=11 --sigma=5 --iter=20 --out=results/demo

# Deblur an image that is already blurred (no ground truth / metrics):
./build/deconvblind_app my_blurred.png --already-blurred --psf=11 --iter=20 --out=results/mine
```

Key options (see `--help` / no-args for the full list):

| Option | Meaning |
|--------|---------|
| `--blur=gauss|motion` | blur type to simulate |
| `--psf=<N>` | PSF / flat-init support size (odd) |
| `--sigma=<f>` | Gaussian blur sigma |
| `--len=<N>`, `--angle=<f>` | motion-blur length and angle |
| `--iter=<N>` | number of iterations |
| `--noise=<f>` | additive Gaussian noise sigma, in `[0,1]` |
| `--dampar=<f>` | damping parameter (suppresses noise amplification) |
| `--init=flat\|true\|line` | initial PSF: `flat` uniform kernel (fully blind, default); `line` uniform line at `--angle` (orientation known, profile blind); `true` ground-truth PSF (validation upper bound) |
| `--already-blurred` | treat the input as already blurred; skip simulation & metrics |

Each run writes `*_original.png`, `*_blurred.png`, `*_deblurred.png`,
`*_deblurred_best.png`, `*_true_psf.png` and `*_recovered_psf.png`.

## Verification on public images

`run_tests.sh` downloads freely redistributable sample images from the
[OpenCV sample-data set](https://github.com/opencv/opencv/tree/master/samples/data)
(baboon, fruits, butterfly), blurs each with a known kernel, runs the blind
deconvolution and checks that image quality (PSNR **and** SSIM) improves.

```bash
./run_tests.sh
```

Representative output:

```
SCENARIO       IMAGE       PSNR_blur  PSNR_best      dPSNR   PASS
---------------------------------------------------------------------
gauss_broad    fruits         27.243     28.936       1.69    YES
gauss_broad    baboon         19.966     21.164       1.20    YES
gauss_broad    butterfly      17.787     19.640       1.85    YES
gauss_flat     fruits         27.545     31.109       3.56    YES
gauss_noisy    fruits         27.020     28.014       0.99    YES
```

Two independent correctness checks are reported for every run:

1. **Image quality** – PSNR and SSIM of the restored image against the
   ground-truth sharp image both increase.
2. **Data fidelity** – the restored image, re-blurred by the recovered PSF,
   reproduces the observed blurred image with a *lower* RMSE than before. This
   confirms the iterations genuinely minimise the maximum-likelihood objective.

## Linear motion blur

Motion blur is a stress test for blind deconvolution, so it has its own
characterization script:

```bash
./test_motion.sh
```

Representative results on `fruits.jpg` (15-pixel motion blur, `psfNCC` = shape
correlation of the recovered kernel with the true kernel):

```
scenario                    PSNRblur  PSNRbest    iter    dPSNR   psfNCC
# Fully blind (flat square init):
flat len=15 a=30              27.512    27.595       1    +0.08    0.406
# Orientation-aware (line init), angle sweep:
line a=0   len=15             27.596    32.703      11    +5.11    1.000
line a=30  len=15             27.512    32.357       9    +4.84    1.000
line a=45  len=15             27.775    33.982      10    +6.21    1.000
line a=90  len=15             27.406    32.623      11    +5.22    1.000
line a=135 len=15             27.043    33.124      10    +6.08    1.000
# Orientation-aware + noise (damped):
line noise=0.005              27.452    30.480       8    +3.03    1.000
line noise=0.02               26.638    27.547       4    +0.91    1.000
```

**Behavioural findings:**

* The Richardson–Lucy **core handles motion blur well**. With a reasonable
  kernel-shape prior — a uniform **`line` init that knows only the motion
  orientation** — it recovers the kernel almost exactly (`NCC ≈ 1.0`) and gains
  **~5–6 dB PSNR across all angles and lengths**. Seeding with the true PSF
  (`--init=true`) gives the same result, confirming the port is correct.
* A **flat square init fails** for motion: it is both a poor shape prior for a
  thin line and a *near-stationary point* of the PSF update for axis-aligned
  (0°/90°) kernels, where the uniform kernel does not move at all. Fully
  blind-from-flat therefore barely improves motion blur (the best iterate is
  usually iteration 1) and over-sharpens thereafter. This matches the
  `deconvblind` documentation: *"PSF restoration is affected strongly by the
  size of its initial guess."* For a flat init the recovered kernel still picks
  up the correct **orientation** of diagonal motion (`NCC ≈ 0.4`) but cannot
  concentrate the energy onto the line.
* **Under noise**, damping (`--dampar`) plus best-iterate early stopping keep the
  result stable; the gain shrinks smoothly as noise grows and the optimal
  iteration count decreases — the expected maximum-likelihood behaviour.

**Practical guidance for motion blur:** provide a kernel-shaped initial guess
(a thin line at the estimated motion direction) rather than a flat square. Even
a rough orientation estimate is enough for the algorithm to recover the full
kernel and deblur the image.

## Algorithm notes

The port mirrors `deconvblind.m` step for step:

* `psf2otf` / `otf2psf` – PSF↔OTF conversion with the exact MATLAB
  circular-shift centring convention (implemented via OpenCV's `cv::dft`).
* `corelucy` – the accelerated, damped Richardson–Lucy core operator, including
  the `DAMPAR` damping branch (`gm = 10`).
* Per iteration: Biggs–Andrews acceleration (the `lambda` prediction step),
  positivity constraints on both image and PSF, and sum-to-one PSF
  normalisation.

### Choosing the initial PSF — important

As the MATLAB documentation states, *"the PSF restoration is affected strongly
by the size of its initial guess."* A flat (all-ones) initial PSF of size `N`
behaves like a uniform blur of radius `N/2`, so the init size should
**approximate the true blur extent**. An init that is much larger than the real
blur gives the PSF too many degrees of freedom and the algorithm tends to
over-sharpen the image into a flat PSF instead of recovering the true one.

### Early stopping

Like all Richardson–Lucy methods, the algorithm sharpens up to an optimum and
then amplifies noise and ringing. The harness therefore monitors every
iteration (via the `DeconvParams::onIteration` callback) and also reports the
**best** iterate; `--dampar` can be used to suppress noise amplification and
extend the usable iteration range.

## Library API

```cpp
#include "deconvblind.h"

cv::Mat blurred = cv::imread("blurred.png", cv::IMREAD_GRAYSCALE);
cv::Mat initPSF = cv::Mat::ones(11, 11, CV_64F) / (11.0 * 11.0);

enhanced::DeconvParams params;
params.numIterations = 20;
params.dampar        = 0.0;   // optional damping

enhanced::DeconvResult r = enhanced::deconvblind(blurred, initPSF, params);
// r.image : restored image  (CV_64F, [0,1])
// r.psf   : recovered PSF    (CV_64F, sums to 1)
```
