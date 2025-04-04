#include <fftw3.h>

#include <algorithm>
#include <complex>
#include <iostream>
#include <numeric>
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

template <size_t N>
class Deblurrer {
 public:
  Deblurrer(double (&aBlurredData)[N], double (&aKernelData)[N],
            double (&aDeblurredData)[N])
      : mBlurredData{aBlurredData},
        mKernelData{aKernelData},
        mDeblurredData{aDeblurredData} {}

  void deblur(double aAlpha, int aIterations, double aThreshold) {
    mMirrorKernelData[0] = mKernelData[0];
    for (int i = 1; i < N; i++) {
      mMirrorKernelData[i] = mKernelData[N - i];
    }

    // van Cittert starts here
    int iteration = 0;
    do {
      convolve(mDeblurredWrapper, mKernelWrapper, mReblurredWrapperBw);

      std::ranges::transform(mBlurredData, mReblurredData, mResidualData,
                             std::minus{});

      convolve(mResidualWrapper, mMirrorKernelWrapper, mReblurredWrapperBw);

      double oldDeblurredData[N];
      std::ranges::copy(mDeblurredData, oldDeblurredData);

      std::ranges::transform(
          mDeblurredData, mReblurredData, mDeblurredData,
          [aAlpha](auto lhs, auto rhs) { return lhs + aAlpha * rhs; });

      mResidualNorm = 0;
      for (int i = 0; i < N; i++) {
        const auto diff = oldDeblurredData[i] - mDeblurredData[i];
        mResidualNorm += diff * diff;
      }
      mResidualNorm = std::sqrt(mResidualNorm) / N;

      // std::cout << "iteration: " << iteration
      //           << "  residualNorm: " << mResidualNorm << std::endl;

      iteration++;
    } while ((mResidualNorm > 5e-7) && (iteration < aIterations));

    std::cout << "iterations: " << iteration
              << "  residualNorm: " << mResidualNorm << std::endl;
  }

 private:
  double (&mBlurredData)[N];
  FftForward<N> mBlurredWrapper{mBlurredData};

  double (&mKernelData)[N];
  FftForward<N> mKernelWrapper{mKernelData};

  double mMirrorKernelData[N]{};
  FftForward<N> mMirrorKernelWrapper{mMirrorKernelData};

  double (&mDeblurredData)[N]{};
  FftForward<N> mDeblurredWrapper{mDeblurredData};

  double mReblurredData[N]{};
  FftBackward<N> mReblurredWrapperBw{mReblurredData};

  double mReblurredTransposedData[N]{};
  FftBackward<N> mReblurredTransposedWrapper{mReblurredTransposedData};

  double mResidualData[N]{};
  FftForward<N> mResidualWrapper{mResidualData};

  float mResidualNorm{};
};

template <size_t N>
void regularizeKernel(double (&aKernelData)[N]) {
  std::ranges::for_each(aKernelData, [](double &elem) {
    elem = std::clamp<double>(elem, 0.0, N);
  });
  const auto kernelSum =
      std::accumulate(std::begin(aKernelData), std::end(aKernelData), 0.0);
  std::ranges::for_each(aKernelData,
                        [kernelSum](double &elem) { elem /= kernelSum; });
}

int main(int argc, char *argv[]) {
  constexpr int N = 16;
  double originalData[N] = {1, 1, 1, 5, 5, 2, 2, 2, 2, 6, 6, 6, 6, 6, 0, 0};

  constexpr double w = 1.0 / 3.0;
  double originalKernelData[N] = {w, w, w, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0};

  double blurredData[N] = {0.333333, 0.666667, 1, 2.33333, 3.66667, 4, 3, 2, 2,
                           3.33333,  4.66667,  6, 6,       6,       4, 2};

  double kernelData[N]{};

  double deblurredData[N]{};

  {
    double deblurredData[N]{};
    std::ranges::copy(blurredData, deblurredData);
    Deblurrer test{blurredData, originalKernelData, deblurredData};
    test.deblur(1.5, 5000, 5e-7);
    std::cout << "deblurredData " << deblurredData << '\n';
    std::cout << "originalData " << originalData << '\n';
  }

  double kernelInitialData[N] = {0.5, 0.25, 0.0, 0.0, 0.1, 0.0, 0, 0,
                                 0,   0,    0,   0,   0,   0,   0, 0.25};
  {
    double restoredKernelData[N]{};
    std::ranges::copy(kernelInitialData, restoredKernelData);
    Deblurrer test{blurredData, originalData, restoredKernelData};
    test.deblur(0.00011, 6000, 5e-7);

    regularizeKernel(restoredKernelData);

    std::cout << "restoredKernelData " << restoredKernelData << '\n';
    std::cout << "originalKernelData " << originalKernelData << '\n';
  }

  std::ranges::copy(blurredData, deblurredData);
  Deblurrer dataDeblurrer{blurredData, kernelData, deblurredData};

  std::ranges::copy(kernelInitialData, kernelData);
  Deblurrer kernelDeblurrer{blurredData, deblurredData, kernelData};

  for (int i = 0; i < 20; i++) {
    dataDeblurrer.deblur(1.5, 5, 5e-7);
    std::cout << "deblurredData " << deblurredData << '\n';
    std::cout << "originalData " << originalData << '\n';

    kernelDeblurrer.deblur(0.00011, 5, 5e-7);
    regularizeKernel(kernelData);

    std::cout << "kernelData " << kernelData << '\n';
    std::cout << "originalKernelData " << originalKernelData << '\n';
  }

  return EXIT_SUCCESS;
}