#include <fftw3.h>

#include <algorithm>
#include <complex>
#include <iostream>
#include <span>

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

  std::span<fftw_complex, N> getFft() {
    return std::span<fftw_complex, N>{mArrayFft, N};
  }

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

  std::span<fftw_complex, N> operator()() {
    fftw_execute(mPlan);
    return std::span<fftw_complex, N>{mArrayFft, N};
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
  auto firstFft = aFirstInput();
  auto seconfFft = aSecondInput();

  auto convolvedFft = aOutput.getFft();

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

  double laplacianKernelData[N] = {-2, 1, 0, 0, 0, 0, 0, 0,
                                   0,  0, 0, 0, 0, 0, 0, 1};
  FftForward laplacianKernelWrapper{laplacianKernelData};

  double deblurredData[N]{};
  FftForward deblurredWrapper{deblurredData};

  double reblurredData[N]{};
  FftBackward reblurredWrapperBw{reblurredData};

  double reblurredTransposedData[N]{};
  FftBackward reblurredTransposedWrapper{reblurredTransposedData};

  double residualData[N]{};
  FftForward residualWrapper{residualData};

  float residualNorm{};  // norm of the images

  // van Cittert starts here
  float betha = 2.0;
  int iteration = 0;
  std::ranges::copy(blurredData, deblurredData);
  do {
    convolve(deblurredWrapper, kernelWrapper, reblurredWrapperBw);

    std::ranges::transform(blurredData, reblurredData, residualData,
                           std::minus{});

    convolve(residualWrapper, mirrorKernelWrapper, reblurredWrapperBw);

    double oldDeblurredData[N];
    std::ranges::copy(deblurredData, oldDeblurredData);

    std::ranges::transform(
        deblurredData, reblurredData, deblurredData,
        [betha](auto lhs, auto rhs) { return lhs + betha * rhs; });

    residualNorm = 0;
    for (int i = 0; i < N; i++) {
      const auto diff = oldDeblurredData[i] - deblurredData[i];
      residualNorm += diff * diff;
    }
    residualNorm = std::sqrt(residualNorm) / N;

    std::cout << "iteration: " << iteration
              << "  residualNorm: " << residualNorm << std::endl;

    iteration++;
  } while ((residualNorm > 5e-6) && (iteration < 2000));

  std::cout << deblurredData << '\n';
  std::cout << originalData << '\n';

  return EXIT_SUCCESS;
}