// 1D blind deconvolution sample.
//
// Given only a blurred signal, this sample jointly estimates both the
// deblurred (latent) signal and the blur kernel using an alternating
// multiplicative Richardson-Lucy scheme:
//   - update the signal while the kernel is kept fixed,
//   - update the kernel while the signal is kept fixed.
//
// The program also synthesizes its own ground-truth signal and kernel, blurs
// the signal, runs the blind deconvolution on the blurred result and checks the
// recovered signal and kernel against the originals.
//
// References:
//   https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
//   https://en.wikipedia.org/wiki/Blind_deconvolution
#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>

constexpr std::complex<double> convertToComplex(const fftw_complex &aValue) {
  return std::complex<double>{aValue[0], aValue[1]};
}

constexpr void convertFromComplex(const std::complex<double> &aInput,
                                  fftw_complex &aOutput) {
  aOutput[0] = aInput.real();
  aOutput[1] = aInput.imag();
}

template <size_t N>
std::ostream &operator<<(std::ostream &aOutStream, const double (&aData)[N]) {
  aOutStream << '[';
  for (int i = 0; i < N; i++) {
    aOutStream << aData[i] << " ";
  }
  aOutStream << ']';
  return aOutStream;
}

template <size_t N>
class FftBackward {
 public:
  FftBackward(double (&aArray)[N]) : mArray{aArray} {
    mArrayFft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
    mPlan = fftw_plan_dft_c2r_1d(N, mArrayFft, mArray, FFTW_ESTIMATE);
  }

  ~FftBackward() {
    fftw_destroy_plan(mPlan);
    fftw_free(mArrayFft);
  }

  operator fftw_complex *() { return mArrayFft; }

  void operator()() {
    fftw_execute(mPlan);
    for (int i = 0; i < N; i++) {
      mArray[i] /= N;
    }
  }

  friend std::ostream &operator<<(std::ostream &aOutStream,
                                  const FftBackward<N> &aData) {
    aOutStream << aData.mArray;
    return aOutStream;
  }

 private:
  double (&mArray)[N];
  fftw_complex *mArrayFft{};
  fftw_plan mPlan;
};

template <size_t N>
class FftForward {
 public:
  FftForward(double (&aArray)[N]) : mArray{aArray} {
    mArrayFft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
    mPlan = fftw_plan_dft_r2c_1d(N, mArray, mArrayFft, FFTW_ESTIMATE);
  }

  ~FftForward() {
    fftw_destroy_plan(mPlan);
    fftw_free(mArrayFft);
  }

  fftw_complex *operator()() {
    fftw_execute(mPlan);
    return mArrayFft;
  }

  friend std::ostream &operator<<(std::ostream &aOutStream,
                                  const FftForward<N> &aData) {
    aOutStream << aData.mArray;
    return aOutStream;
  }

 private:
  double (&mArray)[N];
  fftw_complex *mArrayFft{};
  fftw_plan mPlan;
};

// Circular convolution of two real signals via the frequency domain.
template <size_t N>
void convolve(FftForward<N> &aFirstInput, FftForward<N> &aSecondInput,
              FftBackward<N> &aOutput) {
  auto *firstFft = aFirstInput();
  auto *secondFft = aSecondInput();

  auto *convolvedFft = static_cast<fftw_complex *>(aOutput);

  for (int i = 0; i < N / 2 + 1; i++) {
    convertFromComplex(
        convertToComplex(firstFft[i]) * convertToComplex(secondFft[i]),
        convolvedFft[i]);
  }

  aOutput();
}

// Time reversal used to build the transpose (correlation) kernel:
// mirror[0] = data[0], mirror[i] = data[N - i] for i > 0.
template <size_t N>
void mirror(const double (&aData)[N], double (&aMirrored)[N]) {
  aMirrored[0] = aData[0];
  for (size_t i = 1; i < N; i++) {
    aMirrored[i] = aData[N - i];
  }
}

// Clamp to non-negative values and rescale so the kernel sums to one.
template <size_t N>
void normalizeKernel(double (&aKernelData)[N]) {
  std::ranges::for_each(aKernelData,
                        [](double &elem) { elem = std::max(elem, 0.0); });
  const auto kernelSum =
      std::accumulate(std::begin(aKernelData), std::end(aKernelData), 0.0);
  if (kernelSum > 0.0) {
    std::ranges::for_each(aKernelData,
                          [kernelSum](double &elem) { elem /= kernelSum; });
  }
}

// Compact-support prior: the blur kernel is known to be local, so force to zero
// every tap farther than aRadius from the center (index 0, with wrap-around).
// This breaks the scale/sharpness ambiguity inherent to blind deconvolution.
template <size_t N>
void constrainKernelSupport(double (&aKernelData)[N], size_t aRadius) {
  for (size_t i = 0; i < N; i++) {
    const size_t distance = std::min(i, N - i);
    if (distance > aRadius) {
      aKernelData[i] = 0.0;
    }
  }
}

template <size_t N>
double rmsError(const double (&lhs)[N], const double (&rhs)[N]) {
  double error = 0;
  for (size_t i = 0; i < N; i++) {
    const auto diff = lhs[i] - rhs[i];
    error += diff * diff;
  }
  return std::sqrt(error / N);
}

// Blind deconvolution: from the blurred signal alone, recover both the latent
// signal and the blur kernel. The kernel is stored centered at index 0 with
// wrap-around (a circular point spread function).
template <size_t N>
void blindDeconvolution(const double (&aBlurredData)[N],
                        double (&aDeblurredData)[N], double (&aKernelData)[N],
                        int aOuterIterations, int aSignalIterations,
                        int aKernelIterations, size_t aKernelRadius) {
  constexpr double epsilon = 1e-9;
  const auto safeDivide = [](double num, double den) {
    return num / (den + epsilon);
  };

  double blurredData[N];
  std::ranges::copy(aBlurredData, blurredData);

  // Start from the blurred signal and the caller-supplied kernel guess,
  // restricted to the known compact support.
  std::ranges::copy(aBlurredData, aDeblurredData);
  FftForward deblurredWrapper{aDeblurredData};

  constrainKernelSupport(aKernelData, aKernelRadius);
  normalizeKernel(aKernelData);
  FftForward kernelWrapper{aKernelData};

  double mirrorKernelData[N]{};
  FftForward mirrorKernelWrapper{mirrorKernelData};

  double mirrorDeblurredData[N]{};
  FftForward mirrorDeblurredWrapper{mirrorDeblurredData};

  double reblurredData[N]{};
  FftForward reblurredWrapperFw{reblurredData};
  FftBackward reblurredWrapperBw{reblurredData};

  double correctionData[N]{};
  FftBackward correctionWrapper{correctionData};

  for (int outer = 0; outer < aOuterIterations; outer++) {
    // Richardson-Lucy updates for the signal, kernel held fixed.
    mirror(aKernelData, mirrorKernelData);
    for (int it = 0; it < aSignalIterations; it++) {
      convolve(deblurredWrapper, kernelWrapper, reblurredWrapperBw);
      std::ranges::transform(blurredData, reblurredData, reblurredData,
                             safeDivide);
      convolve(reblurredWrapperFw, mirrorKernelWrapper, correctionWrapper);
      std::ranges::transform(correctionData, aDeblurredData, aDeblurredData,
                             std::multiplies{});
    }

    // Richardson-Lucy updates for the kernel, signal held fixed.
    mirror(aDeblurredData, mirrorDeblurredData);
    for (int it = 0; it < aKernelIterations; it++) {
      convolve(deblurredWrapper, kernelWrapper, reblurredWrapperBw);
      std::ranges::transform(blurredData, reblurredData, reblurredData,
                             safeDivide);
      convolve(reblurredWrapperFw, mirrorDeblurredWrapper, correctionWrapper);
      std::ranges::transform(correctionData, aKernelData, aKernelData,
                             std::multiplies{});
      constrainKernelSupport(aKernelData, aKernelRadius);
      normalizeKernel(aKernelData);
    }
  }
}

// Synthetic non-negative latent signal built from smooth Gaussian bumps on a
// small baseline. A band-limited signal like this is actually recoverable: a
// Gaussian blur irreversibly destroys high frequencies, so sharp edges could
// never be restored even with the exact kernel.
template <size_t N>
void generateOriginalSignal(double (&aData)[N]) {
  const auto bump = [](double x, double center, double amplitude,
                       double width) {
    const double d = x - center;
    return amplitude * std::exp(-(d * d) / (2.0 * width * width));
  };
  for (size_t i = 0; i < N; i++) {
    const double x = static_cast<double>(i);
    aData[i] = 2.0 + bump(x, 0.25 * N, 6.0, 4.0) + bump(x, 0.50 * N, 8.0, 6.0) +
               bump(x, 0.75 * N, 4.0, 3.0);
  }
}

// Symmetric Gaussian blur kernel centered at index 0 with wrap-around.
template <size_t N>
void generateOriginalKernel(double (&aData)[N], double aSigma) {
  for (size_t i = 0; i < N; i++) {
    const double distance = std::min(i, N - i);
    aData[i] = std::exp(-(distance * distance) / (2.0 * aSigma * aSigma));
  }
  normalizeKernel(aData);
}

int main(int argc, char *argv[]) {
  constexpr size_t N = 64;

  // --- Synthesize ground-truth data ---------------------------------------
  double originalData[N]{};
  generateOriginalSignal(originalData);

  double originalKernelData[N]{};
  generateOriginalKernel(originalKernelData, 1.2);
  FftForward originalKernelWrapper{originalKernelData};

  double originalDataCopy[N];
  std::ranges::copy(originalData, originalDataCopy);
  FftForward originalWrapper{originalDataCopy};

  double blurredData[N]{};
  FftBackward blurredWrapper{blurredData};
  convolve(originalWrapper, originalKernelWrapper, blurredWrapper);

  // --- Blind deconvolution: recover the signal and the kernel -------------
  // Seed the kernel with a rough guess of the blur: compact and Gaussian, but
  // deliberately the wrong width. The algorithm refines it toward the truth.
  double deblurredData[N]{};
  double kernelData[N]{};
  generateOriginalKernel(kernelData, 2.0);
  blindDeconvolution(blurredData, deblurredData, kernelData,
                     /*aOuterIterations=*/500, /*aSignalIterations=*/10,
                     /*aKernelIterations=*/5, /*aKernelRadius=*/2);

  // --- Report and verify against the synthetic ground truth ---------------
  std::cout << "originalData   " << originalData << '\n';
  std::cout << "deblurredData  " << deblurredData << '\n';
  std::cout << "originalKernel " << originalKernelData << '\n';
  std::cout << "estimatedKernel" << kernelData << '\n';

  const double signalError = rmsError(deblurredData, originalData);
  const double kernelError = rmsError(kernelData, originalKernelData);

  std::cout << "signal RMS error: " << signalError << '\n';
  std::cout << "kernel RMS error: " << kernelError << '\n';

  constexpr double signalTolerance = 0.1;
  constexpr double kernelTolerance = 0.025;
  const bool passed =
      signalError < signalTolerance && kernelError < kernelTolerance;

  std::cout << (passed ? "TEST PASSED" : "TEST FAILED") << '\n';

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
