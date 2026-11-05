#pragma once

#include <opencv2/core.hpp>

#include <functional>

// Blind deconvolution (accelerated, damped Richardson-Lucy) for grayscale
// images. This is a C++/OpenCV port of MATLAB's Image Processing Toolbox
// function deconvblind.m (and its helper corelucy.m).
//
// References:
//  - D.S.C. Biggs and M. Andrews, "Acceleration of iterative image
//    restoration algorithms", Applied Optics, Vol. 36, No. 8, 1997.
//  - R.J. Hanisch, R.L. White, R.L. Gilliland, "Deconvolutions of Hubble
//    Space Telescope Images and Spectra", 1997.
namespace enhanced {

// Parameters controlling the blind deconvolution, mirroring the optional
// arguments of MATLAB's deconvblind.
struct DeconvParams {
    int numIterations = 10;   // NUMIT  - number of iterations.
    double dampar = 0.0;      // DAMPAR - damping threshold (image units, [0,1]).
    double readout = 0.0;     // READOUT- additive/read-out noise (image units).
    cv::Mat weight;           // WEIGHT - per-pixel quality mask (empty => all ones).

    // Optional callback invoked at the end of every iteration with the
    // 1-based iteration index, the current image estimate and PSF estimate
    // (both CV_64F). Useful for monitoring convergence / early stopping.
    std::function<void(int, const cv::Mat&, const cv::Mat&)> onIteration;
};

// Result of a blind deconvolution run.
struct DeconvResult {
    cv::Mat image;  // Restored (deblurred) image, CV_64F in [0, 1].
    cv::Mat psf;    // Recovered point-spread function, CV_64F, sums to 1.
};

// Run blind deconvolution.
//
//   blurred  : input blurred grayscale image. Accepts CV_8U, CV_16U or
//              CV_64F. It is converted internally to CV_64F in [0, 1].
//   initPSF  : initial point-spread function guess. Its *size* strongly
//              influences the recovered PSF; an array of ones is a safe guess.
//              Accepts CV_8U, CV_16U or CV_64F.
//   params   : optional iteration parameters.
//
// Returns the restored image and the recovered PSF.
DeconvResult deconvblind(const cv::Mat& blurred,
                         const cv::Mat& initPSF,
                         const DeconvParams& params = DeconvParams());

// --- Lower level helpers (exposed for testing / reuse) ---

// Convert an image to CV_64F scaled to [0, 1] (matching MATLAB's im2double).
cv::Mat im2double(const cv::Mat& src);

// Compute the OTF (complex, CV_64FC2) of a PSF padded to outSize, matching
// MATLAB's psf2otf (the PSF centre is moved to the array origin).
cv::Mat psf2otf(const cv::Mat& psf, cv::Size outSize);

// Inverse of psf2otf: convert an OTF (complex, CV_64FC2) back to a real PSF
// of size psfSize, matching MATLAB's otf2psf.
cv::Mat otf2psf(const cv::Mat& otf, cv::Size psfSize);

}  // namespace enhanced

