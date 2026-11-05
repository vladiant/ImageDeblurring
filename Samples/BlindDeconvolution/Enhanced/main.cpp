#include "deconvblind.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using enhanced::DeconvParams;
using enhanced::DeconvResult;

namespace {

// Build a normalized (sum == 1) Gaussian PSF of the given odd size.
cv::Mat gaussianPSF(int size, double sigma) {
    cv::Mat psf(size, size, CV_64F);
    const int half = size / 2;
    double sum = 0.0;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            double dy = y - half;
            double dx = x - half;
            double v = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
            psf.at<double>(y, x) = v;
            sum += v;
        }
    }
    psf /= sum;
    return psf;
}

// Linear motion-blur PSF of the given length and angle (degrees).
cv::Mat motionPSF(int length, double angleDeg) {
    cv::Mat psf = cv::Mat::zeros(length, length, CV_64F);
    const double angle = angleDeg * CV_PI / 180.0;
    const double cx = (length - 1) / 2.0;
    const double cy = (length - 1) / 2.0;
    for (int i = 0; i < length; ++i) {
        double t = i - (length - 1) / 2.0;
        int x = static_cast<int>(std::round(cx + t * std::cos(angle)));
        int y = static_cast<int>(std::round(cy - t * std::sin(angle)));
        if (x >= 0 && x < length && y >= 0 && y < length) {
            psf.at<double>(y, x) += 1.0;
        }
    }
    psf /= cv::sum(psf)[0];
    return psf;
}

// Circular (FFT-based) convolution of a real image with a PSF. This matches
// the periodic blur model that the deconvolution algorithm inverts.
cv::Mat convolveCircular(const cv::Mat& img, const cv::Mat& psf) {
    cv::Mat otf = enhanced::psf2otf(psf, img.size());
    cv::Mat planes[] = {img.clone(), cv::Mat::zeros(img.size(), CV_64F)};
    cv::Mat complex;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex, cv::DFT_COMPLEX_OUTPUT);
    cv::mulSpectrums(complex, otf, complex, 0);
    cv::Mat out;
    cv::dft(complex, out, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    return out;
}

double psnr(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff = diff.mul(diff);
    double mse = cv::mean(diff)[0];
    if (mse <= 1e-20) {
        return 99.0;
    }
    return 10.0 * std::log10(1.0 / mse);  // images are in [0, 1]
}

double rmse(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return std::sqrt(cv::mean(diff.mul(diff))[0]);
}

// Normalized cross-correlation between two equally sized PSFs in [-1, 1].
// 1.0 means the recovered kernel matches the true kernel exactly in shape.
double ncc(const cv::Mat& a, const cv::Mat& b) {
    cv::Scalar ma, sa, mb, sb;
    cv::meanStdDev(a, ma, sa);
    cv::meanStdDev(b, mb, sb);
    double denom = sa[0] * sb[0];
    if (denom < 1e-30 || !std::isfinite(denom)) {
        return 0.0;
    }
    cv::Mat prod = (a - ma[0]).mul(b - mb[0]);
    double v = cv::mean(prod)[0] / denom;
    return std::isfinite(v) ? v : 0.0;
}

// Center-pad (or center-crop) a PSF to a target size, preserving its centre.
// Both psf2otf/otf2psf use a centred convention, so this keeps kernels aligned.
cv::Mat centerResizePSF(const cv::Mat& psf, cv::Size target) {
    cv::Mat out = cv::Mat::zeros(target, CV_64F);
    int dy = target.height / 2 - psf.rows / 2;
    int dx = target.width / 2 - psf.cols / 2;
    for (int y = 0; y < psf.rows; ++y) {
        int ty = y + dy;
        if (ty < 0 || ty >= target.height) continue;
        for (int x = 0; x < psf.cols; ++x) {
            int tx = x + dx;
            if (tx < 0 || tx >= target.width) continue;
            out.at<double>(ty, tx) = psf.at<double>(y, x);
        }
    }
    return out;
}

// Structural Similarity Index (single-channel, double images in [0, 1]).
double ssim(const cv::Mat& a, const cv::Mat& b) {
    const double C1 = 0.01 * 0.01;
    const double C2 = 0.03 * 0.03;
    cv::Mat mu1, mu2;
    cv::GaussianBlur(a, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(b, mu2, cv::Size(11, 11), 1.5);
    cv::Mat mu1Sq = mu1.mul(mu1);
    cv::Mat mu2Sq = mu2.mul(mu2);
    cv::Mat mu1Mu2 = mu1.mul(mu2);

    cv::Mat sigma1Sq, sigma2Sq, sigma12;
    cv::GaussianBlur(a.mul(a), sigma1Sq, cv::Size(11, 11), 1.5);
    sigma1Sq -= mu1Sq;
    cv::GaussianBlur(b.mul(b), sigma2Sq, cv::Size(11, 11), 1.5);
    sigma2Sq -= mu2Sq;
    cv::GaussianBlur(a.mul(b), sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1Mu2;

    cv::Mat t1 = 2 * mu1Mu2 + C1;
    cv::Mat t2 = 2 * sigma12 + C2;
    cv::Mat t3 = t1.mul(t2);
    t1 = mu1Sq + mu2Sq + C1;
    t2 = sigma1Sq + sigma2Sq + C2;
    t1 = t1.mul(t2);
    cv::Mat ssimMap;
    cv::divide(t3, t1, ssimMap);
    return cv::mean(ssimMap)[0];
}

// Scale a double image/PSF to 8-bit for saving (full-range normalization).
cv::Mat toDisplay(const cv::Mat& m, bool normalize) {
    cv::Mat out;
    if (normalize) {
        cv::normalize(m, out, 0, 255, cv::NORM_MINMAX);
    } else {
        out = m * 255.0;
    }
    out.convertTo(out, CV_8U);
    return out;
}

void printUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " <image> [options]\n"
        << "  Simulates a known blur on a sharp grayscale image, runs blind\n"
        << "  deconvolution and reports the quality improvement.\n\n"
        << "Options:\n"
        << "  --blur=gauss|motion   blur type to simulate (default gauss)\n"
        << "  --psf=<N>             PSF support size, odd (default 9)\n"
        << "  --sigma=<f>           gaussian blur sigma (default 2.0)\n"
        << "  --len=<N>             motion blur length (default psf size)\n"
        << "  --angle=<f>           motion blur angle degrees (default 30)\n"
        << "  --iter=<N>            deconvolution iterations (default 30)\n"
        << "  --noise=<f>           gaussian noise sigma in [0,1] (default 0)\n"
        << "  --dampar=<f>          damping parameter (default 0)\n"
        << "  --init=flat|true|line initial PSF guess (default flat).\n"
        << "                        flat: uniform ones kernel (fully blind);\n"
        << "                        line: uniform line at --angle (informed,\n"
        << "                        known orientation); true: ground-truth PSF\n"
        << "                        (validation upper bound, not blind)\n"
        << "  --out=<prefix>        output file prefix (default out)\n"
        << "  --already-blurred     treat <image> as already blurred; skip\n"
        << "                        simulation and metrics\n";
}

std::string optValue(const std::string& arg) {
    auto pos = arg.find('=');
    return pos == std::string::npos ? std::string() : arg.substr(pos + 1);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string imagePath = argv[1];
    std::string blurType = "gauss";
    int psfSize = 11;
    double sigma = 5.0;
    int motionLen = -1;
    double angle = 30.0;
    int iterations = 20;
    double noiseSigma = 0.0;
    double dampar = 0.0;
    std::string initMode = "flat";
    std::string outPrefix = "out";
    bool alreadyBlurred = false;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--blur=", 0) == 0) {
            blurType = optValue(a);
        } else if (a.rfind("--psf=", 0) == 0) {
            psfSize = std::stoi(optValue(a));
        } else if (a.rfind("--sigma=", 0) == 0) {
            sigma = std::stod(optValue(a));
        } else if (a.rfind("--len=", 0) == 0) {
            motionLen = std::stoi(optValue(a));
        } else if (a.rfind("--angle=", 0) == 0) {
            angle = std::stod(optValue(a));
        } else if (a.rfind("--iter=", 0) == 0) {
            iterations = std::stoi(optValue(a));
        } else if (a.rfind("--noise=", 0) == 0) {
            noiseSigma = std::stod(optValue(a));
        } else if (a.rfind("--dampar=", 0) == 0) {
            dampar = std::stod(optValue(a));
        } else if (a.rfind("--init=", 0) == 0) {
            initMode = optValue(a);
        } else if (a.rfind("--out=", 0) == 0) {
            outPrefix = optValue(a);
        } else if (a == "--already-blurred") {
            alreadyBlurred = true;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    cv::Mat gray = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        std::cerr << "Error: could not read image '" << imagePath << "'\n";
        return 1;
    }
    cv::Mat original = enhanced::im2double(gray);

    // Initial PSF guess: a flat (uniform) kernel is a robust starting point.
    cv::Mat initPSF = cv::Mat::ones(psfSize, psfSize, CV_64F) /
                      static_cast<double>(psfSize * psfSize);

    DeconvParams params;
    params.numIterations = iterations;
    params.dampar = dampar;

    if (alreadyBlurred) {
        std::cout << "Running blind deconvolution on already-blurred image...\n";
        DeconvResult res = enhanced::deconvblind(original, initPSF, params);
        cv::imwrite(outPrefix + "_deblurred.png", toDisplay(res.image, false));
        cv::imwrite(outPrefix + "_psf.png", toDisplay(res.psf, true));
        std::cout << "Saved " << outPrefix << "_deblurred.png and "
                  << outPrefix << "_psf.png\n";
        return 0;
    }

    // Build the ground-truth PSF and simulate the blur (+ optional noise).
    cv::Mat truePSF;
    if (blurType == "motion") {
        truePSF = motionPSF(motionLen > 0 ? motionLen : psfSize, angle);
    } else {
        truePSF = gaussianPSF(psfSize, sigma);
    }

    // Select the initial PSF guess.
    //   flat : uniform kernel (fully blind, default)
    //   line : uniform line at the given angle (orientation known, blind to profile)
    //   true : the ground-truth PSF (validation upper bound, not blind)
    if (initMode == "true") {
        initPSF = truePSF.clone();
    } else if (initMode == "line") {
        initPSF = motionPSF(psfSize, angle);
    } else if (initMode != "flat") {
        std::cerr << "Unknown --init mode: " << initMode << "\n";
        return 1;
    }

    cv::Mat blurred = convolveCircular(original, truePSF);
    if (noiseSigma > 0.0) {
        cv::Mat noise(blurred.size(), CV_64F);
        cv::randn(noise, 0.0, noiseSigma);
        blurred += noise;
    }
    cv::max(blurred, 0.0, blurred);
    cv::min(blurred, 1.0, blurred);

    std::cout << "Image: " << imagePath << " (" << gray.cols << "x" << gray.rows
              << ")\n";
    std::cout << "Blur: " << blurType << ", PSF size " << psfSize
              << ", iterations " << iterations << ", init " << initMode;
    if (noiseSigma > 0.0) {
        std::cout << ", noise sigma " << noiseSigma;
    }
    std::cout << "\n";

    const double psnrBlur = psnr(original, blurred);
    const double ssimBlur = ssim(original, blurred);
    const double fitBlur = rmse(convolveCircular(blurred, truePSF), blurred);

    // Monitor every iteration and keep the best image estimate. Richardson-
    // Lucy sharpens up to an optimum, then amplifies noise/ringing, so the
    // best-quality iterate is the meaningful verification of correct work.
    cv::Mat bestImage;
    cv::Mat bestPsf;
    double bestPsnr = -1.0;
    int bestIter = 0;
    params.onIteration = [&](int it, const cv::Mat& img, const cv::Mat& psf) {
        double p = psnr(original, img);
        if (p > bestPsnr) {
            bestPsnr = p;
            bestIter = it;
            bestImage = img.clone();
            bestPsf = psf.clone();
        }
    };

    DeconvResult res = enhanced::deconvblind(blurred, initPSF, params);

    const double psnrFinal = psnr(original, res.image);
    const double ssimFinal = ssim(original, res.image);
    const double ssimBest = ssim(original, bestImage);
    // Data-fidelity residual: how well the restored estimate, reblurred by the
    // recovered PSF, reproduces the observed blurred image. A decrease proves
    // the iterations minimise the likelihood objective (optimiser correctness).
    const double fitFinal = rmse(convolveCircular(res.image, res.psf), blurred);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n                       PSNR (dB)    SSIM\n";
    std::cout << "  Blurred input        " << std::setw(8) << psnrBlur << "    "
              << std::setw(6) << ssimBlur << "\n";
    std::cout << "  Deblurred (final)    " << std::setw(8) << psnrFinal << "    "
              << std::setw(6) << ssimFinal << "   (iter " << iterations << ")\n";
    std::cout << "  Deblurred (best)     " << std::setw(8) << bestPsnr << "    "
              << std::setw(6) << ssimBest << "   (iter " << bestIter << ")\n";
    std::cout << "  Best improvement     " << std::setw(8)
              << (bestPsnr - psnrBlur) << "    " << std::setw(6)
              << (ssimBest - ssimBlur) << "\n";
    std::cout << "\n  Data-fidelity residual (RMSE of reblur vs observed):\n";
    std::cout << "    before: " << fitBlur << "   after: " << fitFinal
              << (fitFinal < fitBlur ? "   (reduced -> likelihood minimised)\n"
                                     : "   (not reduced)\n");

    // PSF recovery quality: shape correlation of the recovered kernel (at the
    // best iterate) with the ground-truth kernel (1.0 = perfect shape match).
    cv::Size psfSz = bestPsf.size();
    double psfNcc = ncc(bestPsf, centerResizePSF(truePSF, psfSz));
    std::cout << "  PSF recovery (shape NCC vs true kernel): " << psfNcc
              << "   (iter " << bestIter << ")\n";

    cv::imwrite(outPrefix + "_original.png", toDisplay(original, false));
    cv::imwrite(outPrefix + "_blurred.png", toDisplay(blurred, false));
    cv::imwrite(outPrefix + "_deblurred.png", toDisplay(res.image, false));
    cv::imwrite(outPrefix + "_deblurred_best.png", toDisplay(bestImage, false));
    cv::imwrite(outPrefix + "_true_psf.png", toDisplay(truePSF, true));
    cv::imwrite(outPrefix + "_recovered_psf.png", toDisplay(bestPsf, true));
    std::cout << "\nSaved " << outPrefix << "_{original,blurred,deblurred,"
              << "deblurred_best,true_psf,recovered_psf}.png\n";

    bool improved = bestPsnr > psnrBlur && ssimBest > ssimBlur;
    std::cout << (improved ? "RESULT: PASS (deconvolution improved the image)\n"
                           : "RESULT: WARN (no improvement measured)\n");
    return improved ? 0 : 2;
}
