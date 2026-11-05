#include "deconvblind.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace enhanced {

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();  // ~2.22e-16

// Circularly shift a single-channel matrix by (dy, dx). Positive values shift
// content down / right; negative values shift up / left (MATLAB circshift).
cv::Mat circShift(const cv::Mat& src, int dy, int dx) {
    const int rows = src.rows;
    const int cols = src.cols;
    dy = ((dy % rows) + rows) % rows;
    dx = ((dx % cols) + cols) % cols;

    cv::Mat dst(src.size(), src.type());
    for (int quadrant = 0; quadrant < 4; ++quadrant) {
        const int srcY = (quadrant & 1) ? 0 : rows - dy;
        const int srcX = (quadrant & 2) ? 0 : cols - dx;
        const int h = (quadrant & 1) ? rows - dy : dy;
        const int w = (quadrant & 2) ? cols - dx : dx;
        if (h <= 0 || w <= 0) {
            continue;
        }
        const int dstY = (srcY + dy) % rows;
        const int dstX = (srcX + dx) % cols;
        src(cv::Rect(srcX, srcY, w, h)).copyTo(dst(cv::Rect(dstX, dstY, w, h)));
    }
    return dst;
}

// Forward 2D FFT of a real image, returning a complex (CV_64FC2) spectrum.
cv::Mat fft2(const cv::Mat& real) {
    cv::Mat planes[] = {real.clone(), cv::Mat::zeros(real.size(), CV_64F)};
    cv::Mat complex;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex, cv::DFT_COMPLEX_OUTPUT);
    return complex;
}

// Inverse 2D FFT, returning the real part (matches real(ifftn(...))).
cv::Mat ifft2Real(const cv::Mat& complex) {
    cv::Mat real;
    cv::dft(complex, real, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    return real;
}

// Element-wise complex product a .* b, optionally conjugating b.
cv::Mat mulComplex(const cv::Mat& a, const cv::Mat& b, bool conjB) {
    cv::Mat out;
    cv::mulSpectrums(a, b, out, 0, conjB);
    return out;
}

double matSum(const cv::Mat& m) { return cv::sum(m)[0]; }

// Accelerated, damped Richardson-Lucy core operator (port of corelucy.m).
// Returns the FFT of the image ratio used by both the image and PSF updates.
cv::Mat coreLucy(const cv::Mat& Y, const cv::Mat& otf, double dampar22,
                 const cv::Mat& wI, double readout) {
    // 1. Reblur the current estimate: ReBlurred = real(ifftn(H .* fftn(Y))).
    cv::Mat reBlurred = ifft2Real(mulComplex(otf, fft2(Y), false));

    // 2. Estimate for the next step.
    reBlurred += readout;
    cv::Mat imRatio(Y.size(), CV_64F);
    const int rows = Y.rows;
    const int cols = Y.cols;

    if (dampar22 == 0.0) {
        // No damping: ImRatio = wI ./ ReBlurred + eps.
        for (int y = 0; y < rows; ++y) {
            const double* rb = reBlurred.ptr<double>(y);
            const double* w = wI.ptr<double>(y);
            double* out = imRatio.ptr<double>(y);
            for (int x = 0; x < cols; ++x) {
                double denom = rb[x] == 0.0 ? kEps : rb[x];
                out[x] = w[x] / denom + kEps;
            }
        }
    } else {
        // Damping of the image relative to dampar22 = (DAMPAR/sqrt(N))^2 / ...
        const double gm = 10.0;
        for (int y = 0; y < rows; ++y) {
            const double* rb = reBlurred.ptr<double>(y);
            const double* w = wI.ptr<double>(y);
            double* out = imRatio.ptr<double>(y);
            for (int x = 0; x < cols; ++x) {
                double denom = rb[x] == 0.0 ? kEps : rb[x];
                double anEstim = w[x] / denom + kEps;
                double g = (w[x] * std::log(anEstim) + readout - w[x] + rb[x]) /
                           dampar22;
                g = std::min(g, 1.0);
                double G = std::pow(g, gm - 1.0) * (gm - (gm - 1.0) * g);
                out[x] = 1.0 + G * (anEstim - 1.0);
            }
        }
    }

    return fft2(imRatio);
}

// Positivity constraint: max(m, 0) in place.
void clampNonNegative(cv::Mat& m) { cv::max(m, 0.0, m); }

// Normalize a PSF so its elements sum to one (guarding against all-zero).
void normalizePSF(cv::Mat& psf) {
    double s = matSum(psf);
    psf /= (s + (s == 0.0 ? kEps : 0.0));
}

}  // namespace

cv::Mat im2double(const cv::Mat& src) {
    cv::Mat dst;
    switch (src.depth()) {
        case CV_64F:
            src.convertTo(dst, CV_64F);
            break;
        case CV_32F:
            src.convertTo(dst, CV_64F);
            break;
        case CV_8U:
            src.convertTo(dst, CV_64F, 1.0 / 255.0);
            break;
        case CV_16U:
            src.convertTo(dst, CV_64F, 1.0 / 65535.0);
            break;
        default:
            src.convertTo(dst, CV_64F);
            break;
    }
    return dst;
}

cv::Mat psf2otf(const cv::Mat& psf, cv::Size outSize) {
    CV_Assert(psf.rows <= outSize.height && psf.cols <= outSize.width);
    cv::Mat padded = cv::Mat::zeros(outSize, CV_64F);
    psf.copyTo(padded(cv::Rect(0, 0, psf.cols, psf.rows)));
    // Move the PSF centre to the array origin: shift by -floor(psfSize/2).
    padded = circShift(padded, -(psf.rows / 2), -(psf.cols / 2));
    return fft2(padded);
}

cv::Mat otf2psf(const cv::Mat& otf, cv::Size psfSize) {
    cv::Mat full = ifft2Real(otf);
    // Undo the psf2otf shift, then crop to the PSF support.
    full = circShift(full, psfSize.height / 2, psfSize.width / 2);
    return full(cv::Rect(0, 0, psfSize.width, psfSize.height)).clone();
}

DeconvResult deconvblind(const cv::Mat& blurred, const cv::Mat& initPSF,
                         const DeconvParams& params) {
    if (blurred.empty()) {
        throw std::invalid_argument("deconvblind: input image is empty");
    }
    if (initPSF.empty()) {
        throw std::invalid_argument("deconvblind: initial PSF is empty");
    }
    if (blurred.channels() != 1) {
        throw std::invalid_argument("deconvblind: image must be grayscale");
    }
    if (params.numIterations < 1) {
        throw std::invalid_argument("deconvblind: numIterations must be >= 1");
    }

    // 1. Prepare parameters.
    const cv::Mat I = im2double(blurred);
    const cv::Size sizeI = I.size();
    const cv::Size sizePSF = initPSF.size();
    if (sizePSF.width > sizeI.width || sizePSF.height > sizeI.height) {
        throw std::invalid_argument("deconvblind: PSF must not exceed image size");
    }

    cv::Mat P2;
    initPSF.convertTo(P2, CV_64F);

    cv::Mat weight;
    if (params.weight.empty()) {
        weight = cv::Mat::ones(sizeI, CV_64F);
    } else {
        CV_Assert(params.weight.size() == sizeI);
        params.weight.convertTo(weight, CV_64F);
    }

    const double readout = params.readout;
    const double dampar22 = (params.dampar * params.dampar) / 2.0;

    cv::Mat wI;
    cv::multiply(weight, I + readout, wI);
    clampNonNegative(wI);  // positivity constraint
    const cv::Mat fw = fft2(weight);

    // Iterative state (MATLAB J{} and P{} cells).
    cv::Mat J2 = I.clone();                       // current image estimate
    cv::Mat J3 = cv::Mat::zeros(sizeI, CV_64F);   // previous image estimate
    cv::Mat J4a = cv::Mat::zeros(sizeI, CV_64F);  // most recent image gradient
    cv::Mat J4b = cv::Mat::zeros(sizeI, CV_64F);  // previous image gradient

    cv::Mat P3 = cv::Mat::zeros(sizePSF, CV_64F);   // previous PSF estimate
    cv::Mat P4a = cv::Mat::zeros(sizePSF, CV_64F);  // most recent PSF gradient
    cv::Mat P4b = cv::Mat::zeros(sizePSF, CV_64F);  // previous PSF gradient

    // 2. Richardson-Lucy iterations.
    for (int k = 1; k <= params.numIterations; ++k) {
        // 2.a Image prediction (acceleration) with positivity constraint.
        double lambda = 0.0;
        if (k > 2) {
            double num = matSum(J4a.mul(J4b));
            double den = matSum(J4b.mul(J4b)) + kEps;
            lambda = std::max(std::min(num / den, 1.0), 0.0);
        }
        cv::Mat Y = J2 + lambda * (J2 - J3);
        clampNonNegative(Y);

        // PSF prediction with positivity + normalization constraints.
        lambda = 0.0;
        if (k > 2) {
            double num = matSum(P4a.mul(P4b));
            double den = matSum(P4b.mul(P4b)) + kEps;
            lambda = std::max(std::min(num / den, 1.0), 0.0);
        }
        cv::Mat B = P2 + lambda * (P2 - P3);
        clampNonNegative(B);
        normalizePSF(B);

        // 2.b Core Richardson-Lucy estimation.
        cv::Mat CC = coreLucy(Y, psf2otf(B, sizeI), dampar22, wI, readout);

        // 2.c Next image estimate (positivity constraint).
        J3 = J2.clone();
        cv::Mat H = psf2otf(P2, sizeI);
        cv::Mat scale = ifft2Real(mulComplex(fw, H, true)) +
                        std::sqrt(kEps);
        cv::Mat numerator = ifft2Real(mulComplex(CC, H, true));
        cv::Mat newJ2;
        cv::multiply(Y, numerator, newJ2);
        cv::divide(newJ2, scale, newJ2);
        clampNonNegative(newJ2);
        J4b = J4a;
        J4a = newJ2 - Y;
        J2 = newJ2;

        // 2.d Next PSF estimate (positivity + normalization constraints).
        P3 = P2.clone();
        cv::Mat Hj = fft2(J3);
        cv::Mat scaleP = otf2psf(mulComplex(fw, Hj, true), sizePSF) +
                         std::sqrt(kEps);
        cv::Mat numP = otf2psf(mulComplex(CC, Hj, true), sizePSF);
        cv::Mat newP2;
        cv::multiply(B, numP, newP2);
        cv::divide(newP2, scaleP, newP2);
        clampNonNegative(newP2);
        normalizePSF(newP2);
        P4b = P4a;
        P4a = newP2 - B;
        P2 = newP2;

        if (params.onIteration) {
            params.onIteration(k, J2, P2);
        }
    }

    DeconvResult result;
    result.image = J2;
    result.psf = P2;
    return result;
}

}  // namespace enhanced
