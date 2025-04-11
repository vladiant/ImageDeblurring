// https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
// https://www.strollswithmydog.com/richardson-lucy-algorithm/
#include <fftw3.h>

#include <algorithm>
#include <complex>
#include <iostream>

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

template <size_t N>
void convolve(FftForward<N> &aFirstInput, FftForward<N> &aSecondInput,
              FftBackward<N> &aOutput) {
  auto *firstFft = aFirstInput();
  auto *seconfFft = aSecondInput();

  auto *convolvedFft = static_cast<fftw_complex *>(aOutput);

  for (int i = 0; i < N / 2 + 1; i++) {
    convertFromComplex(
        convertToComplex(firstFft[i]) * convertToComplex(seconfFft[i]),
        convolvedFft[i]);
  }

  aOutput();
}

int main(int argc, char *argv[]) {
  constexpr int N = 16;
  double originalData[N] = {1, 1, 1, 5, 5, 2, 2, 2, 2, 6, 6, 6, 6, 6, 0, 0};

  double blurredData[N] = {0.333333, 0.666667, 1, 2.33333, 3.66667, 4, 3, 2, 2,
                           3.33333,  4.66667,  6, 6,       6,       4, 2};
  FftForward blurredWrapper{blurredData};

  constexpr double w = 1.0 / 3.0;

  double kernelData[N] = {w, w, w, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  FftForward kernelWrapper{kernelData};

  double mirrorKernelData[N] = {w, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, w, w};
  FftForward mirrorKernelWrapper{mirrorKernelData};

  double deblurredData[N]{};
  FftForward deblurredWrapper{deblurredData};

  double reblurredData[N]{};
  FftForward reblurredWrapperFw{reblurredData};
  FftBackward reblurredWrapperBw{reblurredData};

  double reblurredTransposedData[N]{};
  FftBackward reblurredTransposedWrapper{reblurredTransposedData};

  float residualNorm{};  // norm of the images

  // Richardson-Lucy starts here
  double alpha = 0.9;
  int iteration = 0;
  std::ranges::copy(blurredData, deblurredData);
  do {
    convolve(deblurredWrapper, kernelWrapper, reblurredWrapperBw);

    std::ranges::transform(blurredData, reblurredData, reblurredData,
                           [](auto lhs, auto rhs) { return 1.0 - lhs / rhs; });

    convolve(reblurredWrapperFw, mirrorKernelWrapper,
             reblurredTransposedWrapper);

    double oldDeblurredData[N];
    std::ranges::copy(deblurredData, oldDeblurredData);

    std::ranges::transform(
        deblurredData, reblurredTransposedData, deblurredData,
        [alpha](auto lhs, auto rhs) { return lhs - alpha * rhs; });

    residualNorm = 0;
    for (int i = 0; i < N; i++) {
      const auto diff = oldDeblurredData[i] - deblurredData[i];
      residualNorm += diff * diff;
    }
    residualNorm = std::sqrt(residualNorm) / N;

    std::cout << deblurredData << '\n';
    std::cout << originalData << '\n';

    std::cout << "iteration: " << iteration
              << "  residualNorm: " << residualNorm << std::endl;

    iteration++;
  } while ((residualNorm > 5e-7) && (iteration < 7000));

  return EXIT_SUCCESS;
}