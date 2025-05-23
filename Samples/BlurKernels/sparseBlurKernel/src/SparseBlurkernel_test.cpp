#include "SparseBlurkernel.hpp"

#include <stdlib.h>

#include <iostream>

#define VALUE_TO_CHANGE 0.03

#define X_TO_ADD 5
#define Y_TO_ADD 6
#define VALUE_TO_ADD 0.04

#define VALUE_TO_SUM 0.21

#define X_UNSET 100
#define Y_UNSET 100

using namespace Test::Deblurring;

int main(int argc, char* argv[]) {
  // Test coordinates and values.
  int x[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
  int y[] = {10, 8, 6, 4, 2, 0, 1, 2, 3, 4, 5};
  float values[] = {0.05, 0.07, 0.1,  0.2,  0.4, 0.3,
                    0.15, 0.11, 0.09, 0.08, 0.06};

  std::cout << "Create sparse kernel." << std::endl;

  SparseBlurKernel testSparseKernel;

  std::cout << "Set values of the sparse kernel." << std::endl;

  for (unsigned int i = 0; i < sizeof(x) / sizeof(x[0]); i++) {
    std::cout << "x: " << x[i] << "\ty: " << y[i] << "\tvalue: " << values[i]
              << std::endl;
    testSparseKernel.setPointValue(x[i], y[i], values[i]);
  }

  std::cout << "Size of the sparse kernel." << std::endl;

  int sparseKernelSize = testSparseKernel.getKernelSize();

  std::cout << "sparseKernelSize: " << sparseKernelSize << std::endl;

  std::cout << "Extract data of the sparse kernel." << std::endl;

  point_coord_t* xExtracted = new point_coord_t[sparseKernelSize];
  point_coord_t* yExtracted = new point_coord_t[sparseKernelSize];
  point_value_t* valuesExtracted = new point_value_t[sparseKernelSize];

  testSparseKernel.extractKernelPoints(xExtracted, yExtracted, valuesExtracted);

  std::cout << "Print extracted values." << std::endl;

  for (int i = 0; i < sparseKernelSize; i++) {
    std::cout << "x: " << xExtracted[i] << "\ty: " << yExtracted[i]
              << "\tvalue: " << valuesExtracted[i] << std::endl;
  }

  delete[] xExtracted;
  delete[] yExtracted;
  delete[] valuesExtracted;
  xExtracted = NULL;
  yExtracted = NULL;
  valuesExtracted = NULL;

  xExtracted = new point_coord_t[sparseKernelSize];
  yExtracted = new point_coord_t[sparseKernelSize];
  valuesExtracted = new point_value_t[sparseKernelSize];

  std::cout << "Change a point in sparse kernel." << std::endl;

  testSparseKernel.setPointValue(x[sizeof(x) / (2 * sizeof(x[0]))],
                                 y[sizeof(x) / (2 * sizeof(x[0]))],
                                 VALUE_TO_CHANGE);

  std::cout << "x: " << x[sizeof(x) / (2 * sizeof(x[0]))]
            << "  y: " << y[sizeof(x) / (2 * sizeof(x[0]))] << std::endl;
  std::cout << values[sizeof(x) / (2 * sizeof(x[0]))] << " => "
            << VALUE_TO_CHANGE << std::endl;

  std::cout << "Extract data of the sparse kernel." << std::endl;

  testSparseKernel.extractKernelPoints(xExtracted, yExtracted, valuesExtracted);

  std::cout << "Print extracted values." << std::endl;

  for (int i = 0; i < sparseKernelSize; i++) {
    std::cout << "x: " << xExtracted[i] << "\ty: " << yExtracted[i]
              << "\tvalue: " << valuesExtracted[i] << std::endl;
  }

  delete[] xExtracted;
  delete[] yExtracted;
  delete[] valuesExtracted;
  xExtracted = NULL;
  yExtracted = NULL;
  valuesExtracted = NULL;

  std::cout << "Add point to sparse kernel:" << std::endl;
  std::cout << "x: " << X_TO_ADD << "  y: " << Y_TO_ADD
            << "  value: " << VALUE_TO_ADD << std::endl;

  testSparseKernel.setPointValue(X_TO_ADD, Y_TO_ADD, VALUE_TO_ADD);

  sparseKernelSize = testSparseKernel.getKernelSize();

  std::cout << "sparseKernelSize: " << sparseKernelSize << std::endl;

  std::cout << "Extract data of the sparse kernel." << std::endl;

  xExtracted = new point_coord_t[sparseKernelSize];
  yExtracted = new point_coord_t[sparseKernelSize];
  valuesExtracted = new point_value_t[sparseKernelSize];

  testSparseKernel.extractKernelPoints(xExtracted, yExtracted, valuesExtracted);

  std::cout << "Print extracted values." << std::endl;

  for (int i = 0; i < sparseKernelSize; i++) {
    std::cout << "x: " << xExtracted[i] << "\ty: " << yExtracted[i]
              << "\tvalue: " << valuesExtracted[i] << std::endl;
  }

  delete[] xExtracted;
  delete[] yExtracted;
  delete[] valuesExtracted;
  xExtracted = NULL;
  yExtracted = NULL;
  valuesExtracted = NULL;

  std::cout << "Add value to point to sparse kernel:" << std::endl;
  std::cout << "x: " << X_TO_ADD << "  y: " << Y_TO_ADD
            << "  += value: " << VALUE_TO_SUM << std::endl;

  testSparseKernel.addToPointValue(X_TO_ADD, Y_TO_ADD, VALUE_TO_SUM);

  sparseKernelSize = testSparseKernel.getKernelSize();

  std::cout << "sparseKernelSize: " << sparseKernelSize << std::endl;

  std::cout << "Extract data of the sparse kernel." << std::endl;

  xExtracted = new point_coord_t[sparseKernelSize];
  yExtracted = new point_coord_t[sparseKernelSize];
  valuesExtracted = new point_value_t[sparseKernelSize];

  testSparseKernel.extractKernelPoints(xExtracted, yExtracted, valuesExtracted);

  std::cout << "Print extracted values." << std::endl;

  for (int i = 0; i < sparseKernelSize; i++) {
    std::cout << "x: " << xExtracted[i] << "\ty: " << yExtracted[i]
              << "\tvalue: " << valuesExtracted[i] << std::endl;
  }

  std::cout << "Remove point from sparse kernel:" << std::endl;
  std::cout << "x: " << X_TO_ADD << "  y: " << Y_TO_ADD << std::endl;

  testSparseKernel.clearPointValue(X_TO_ADD, Y_TO_ADD);

  sparseKernelSize = testSparseKernel.getKernelSize();

  std::cout << "sparseKernelSize: " << sparseKernelSize << std::endl;

  std::cout << "Extract data of the sparse kernel." << std::endl;

  xExtracted = new point_coord_t[sparseKernelSize];
  yExtracted = new point_coord_t[sparseKernelSize];
  valuesExtracted = new point_value_t[sparseKernelSize];

  testSparseKernel.extractKernelPoints(xExtracted, yExtracted, valuesExtracted);

  std::cout << "Print extracted values." << std::endl;

  for (int i = 0; i < sparseKernelSize; i++) {
    std::cout << "x: " << xExtracted[i] << "\ty: " << yExtracted[i]
              << "\tvalue: " << valuesExtracted[i] << std::endl;
  }

  delete[] xExtracted;
  delete[] yExtracted;
  delete[] valuesExtracted;
  xExtracted = NULL;
  yExtracted = NULL;
  valuesExtracted = NULL;

  std::cout << "Attempt to remove unset point from sparse kernel:" << std::endl;
  std::cout << "x: " << X_UNSET << "  y: " << Y_UNSET << std::endl;

  testSparseKernel.clearPointValue(X_TO_ADD, Y_TO_ADD);

  sparseKernelSize = testSparseKernel.getKernelSize();

  std::cout << "sparseKernelSize: " << sparseKernelSize << std::endl;

  std::cout << "Extract data of the sparse kernel." << std::endl;

  xExtracted = new point_coord_t[sparseKernelSize];
  yExtracted = new point_coord_t[sparseKernelSize];
  valuesExtracted = new point_value_t[sparseKernelSize];

  testSparseKernel.extractKernelPoints(xExtracted, yExtracted, valuesExtracted);

  std::cout << "Print extracted values." << std::endl;

  for (int i = 0; i < sparseKernelSize; i++) {
    std::cout << "x: " << xExtracted[i] << "\ty: " << yExtracted[i]
              << "\tvalue: " << valuesExtracted[i] << std::endl;
  }

  delete[] xExtracted;
  delete[] yExtracted;
  delete[] valuesExtracted;
  xExtracted = NULL;
  yExtracted = NULL;
  valuesExtracted = NULL;

  std::cout << "Extract data of the sparse kernel into vectors" << std::endl;

  std::vector<point_coord_t> xVec;
  std::vector<point_coord_t> yVec;
  std::vector<point_value_t> valueVec;

  testSparseKernel.extractKernelPoints(xVec, yVec, valueVec);

  std::cout << "Vector size: " << xVec.size() << std::endl;

  for (int i = 0; i < sparseKernelSize; i++) {
    std::cout << "x: " << xVec[i] << "\ty: " << yVec[i]
              << "\tvalue: " << valueVec[i] << std::endl;
  }

  std::cout << "Sum of vector elements: "
            << testSparseKernel.calcSumOfElements() << std::endl;

  std::cout << "Normalize kernel." << std::endl;

  testSparseKernel.normalize();

  xVec.clear();
  yVec.clear();
  valueVec.clear();

  testSparseKernel.extractKernelPoints(xVec, yVec, valueVec);

  std::cout << "Vector size: " << xVec.size() << std::endl;

  for (int i = 0; i < sparseKernelSize; i++) {
    std::cout << "x: " << xVec[i] << "\ty: " << yVec[i]
              << "\tvalue: " << valueVec[i] << std::endl;
  }

  std::cout << "Sum of vector elements: "
            << testSparseKernel.calcSumOfElements() << std::endl;

  std::cout << "Calc kernel span" << std::endl;

  point_value_t xCoordMin;
  point_value_t yCoordMin;
  point_value_t xCoordMax;
  point_value_t yCoordMax;

  testSparseKernel.calcCoordsSpan(&xCoordMin, &yCoordMin, &xCoordMax,
                                  &yCoordMax);

  std::cout << "xCoordMin: " << xCoordMin << std::endl;
  std::cout << "yCoordMin: " << yCoordMin << std::endl;
  std::cout << "xCoordMax: " << xCoordMax << std::endl;
  std::cout << "yCoordMax: " << yCoordMax << std::endl;

  std::cout << "Calc partial kernel span" << std::endl;

  testSparseKernel.calcCoordsSpan(&xCoordMin, &yCoordMin, &xCoordMax, NULL);

  std::cout << "xCoordMin: " << xCoordMin << std::endl;
  std::cout << "yCoordMin: " << yCoordMin << std::endl;
  std::cout << "xCoordMax: " << xCoordMax << std::endl;

  std::cout << "Done." << std::endl;

  return EXIT_SUCCESS;
}
