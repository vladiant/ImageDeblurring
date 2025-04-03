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

template <size_t N>
double calculateNorm(const double (&lhs)[N], const double (&rhs)[N]) {
  double residualNorm = 0;
  for (int i = 0; i < N; i++) {
    const auto diff = lhs[i] - rhs[i];
    residualNorm += diff * diff;
  }
  residualNorm = std::sqrt(residualNorm) / N;
  return residualNorm;
}

template <size_t N>
double calculateNorm(const double (&aArray)[N]) {
  double residualNorm = 0;
  for (int i = 0; i < N; i++) {
    residualNorm += aArray[i] * aArray[i];
  }
  residualNorm = std::sqrt(residualNorm) / N;
  return residualNorm;
}

template <size_t N>
double dotProduct(const double (&lhs)[N], const double (&rhs)[N]) {
  double result = 0;
  for (int i = 0; i < N; i++) {
    result += lhs[i] * rhs[i];
  }
  return result;
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

  double residualData[N]{};
  FftForward residualWrapperFw{residualData};
  FftBackward residualWrapperBw{residualData};

  // CGA starts here
  double residualNorm{};

  // initial approximation of the restored image
  std::ranges::copy(blurredData, deblurredData);

  // initial approximation of the residual
  std::ranges::copy(blurredData, residualData);

  // Normal blur.
  convolve(deblurredWrapper, kernelWrapper, residualWrapperBw);
  std::ranges::transform(blurredData, residualData, residualData, std::minus{});
  double initialNorm = calculateNorm(residualData);

  double differenceResidualData[N]{};
  FftBackward differenceResidualWrapper{differenceResidualData};

  double preconditionedResidualData[N]{};
  FftForward preconditionedResidualWrapperFw{preconditionedResidualData};
  FftBackward preconditionedResidualWrapperBw{preconditionedResidualData};

  // Transposed blur.
  convolve(residualWrapperFw, mirrorKernelWrapper,
           preconditionedResidualWrapperBw);

  // initial approximation of preconditioner
  double preconditionedData[N]{};
  FftForward preconditionedWrapper(preconditionedData);
  std::ranges::copy(preconditionedResidualData, preconditionedData);

  // initial approximation of preconditioned blurred image
  double blurredPreconditionedData[N]{};
  FftBackward blurredPreconditionedWrapper{blurredPreconditionedData};
  std::ranges::copy(blurredData, blurredPreconditionedData);

  double bestNorm = initialNorm;
  double bestRestoredData[N]{};
  std::ranges::copy(deblurredData, bestRestoredData);

  // set iteration counter
  int iteration = 0;

  do {
    // init
    double weightDivisor = dotProduct(preconditionedResidualData, residualData);
    double updateWeight = 1.0 / weightDivisor;

    // alpha_k
    // Normal blur.
    convolve(preconditionedWrapper, kernelWrapper,
             blurredPreconditionedWrapper);
    double preconditionWeight =
        weightDivisor /
        dotProduct(preconditionedData, blurredPreconditionedData);

    // x_k
    std::ranges::transform(deblurredData, preconditionedData, deblurredData,
                           [preconditionWeight](auto lhs, auto rhs) {
                             return lhs + preconditionWeight * rhs;
                           });

    // r_k
    std::ranges::copy(residualData, differenceResidualData);
    std::ranges::transform(residualData, blurredPreconditionedData,
                           residualData,
                           [preconditionWeight](auto lhs, auto rhs) {
                             return lhs - preconditionWeight * rhs;
                           });

    std::ranges::transform(residualData, differenceResidualData,
                           differenceResidualData, std::minus{});

    // norm calculation
    residualNorm = calculateNorm(residualData);

    // z_k
    convolve(residualWrapperFw, mirrorKernelWrapper,
             preconditionedResidualWrapperBw);

    // beta_k
    updateWeight *=
        dotProduct(preconditionedResidualData, differenceResidualData);

    std::ranges::transform(preconditionedResidualData, preconditionedData,
                           preconditionedData,
                           [updateWeight](auto lhs, auto rhs) {
                             return lhs + updateWeight * rhs;
                           });

    if (residualNorm < bestNorm) {
      bestNorm = residualNorm;
      std::ranges::copy(deblurredData, bestRestoredData);
    }

    std::cout << " Iteration: " << iteration << " Norm: " << residualNorm
              << " preconditionWeight: " << preconditionWeight
              << " updateWeight: " << updateWeight << std::endl;

    iteration++;
  } while ((residualNorm > 5e-6) && (iteration < 2000));

  std::cout << deblurredData << '\n';
  std::cout << originalData << '\n';

  return EXIT_SUCCESS;
}