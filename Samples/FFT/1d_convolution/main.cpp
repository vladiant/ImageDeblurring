#include <fftw3.h>

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

int main(int argc, char *argv[]) {
  constexpr int N = 16;

  double inputData[N] = {5, 5, 5, 0, 4, 4, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0};
  double kernel[N] = {1.0 / 3.0, 1.0 / 3.0, 0, 0, 0, 0, 0, 0,
                      0,         0,         0, 0, 0, 0, 0, 1.0 / 3.0};

  double outputData[N]{};

  auto *inputFft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
  auto *kernelFft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
  auto *convolvedFft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));

  auto forwardTransform =
      fftw_plan_dft_r2c_1d(N, inputData, inputFft, FFTW_ESTIMATE);

  fftw_execute(forwardTransform);

  auto kernelTransform =
      fftw_plan_dft_r2c_1d(N, kernel, kernelFft, FFTW_ESTIMATE);

  fftw_execute(kernelTransform);

  for (int i = 0; i < N / 2 + 1; i++) {
    convertFromComplex(
        convertToComplex(inputFft[i]) * convertToComplex(kernelFft[i]),
        convolvedFft[i]);
    std::cout << "convolvedFft[" << i
              << "] = " << convertToComplex(convolvedFft[i]) << "\n";
  }

  auto backardTransform =
      fftw_plan_dft_c2r_1d(N, convolvedFft, outputData, FFTW_ESTIMATE);
  fftw_execute(backardTransform);

  for (int i = 0; i < N; i++) {
    std::cout << "outputData[" << i << "] = {" << outputData[i] / N << "}\n";
  }

  fftw_destroy_plan(forwardTransform);
  fftw_destroy_plan(kernelTransform);
  fftw_destroy_plan(backardTransform);

  fftw_free(inputFft);
  fftw_free(kernelFft);
  fftw_free(convolvedFft);

  return EXIT_SUCCESS;
}
