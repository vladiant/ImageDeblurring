/*
 * File:   main.cpp
 * Author: vantonov
 *
 * Created on March 1, 2013, 5:34 PM
 */

#include <BlurKernel.hpp>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace Test::Deblurring;

void printBlurKernel(PSF* inputPSF) {
  if ((inputPSF->width != 0) && (inputPSF->height != 0) &&
      (inputPSF->buffer != nullptr)) {
    float* pData = inputPSF->buffer;
    int row, col;

    for (row = 0; row < inputPSF->height; ++row) {
      for (col = 0; col < inputPSF->width; ++col) {
        cout << *pData << '\t';
        pData++;
      }

      cout << endl;
    }

    cout << endl;

  } else {
    cout << "PSF empty!\n" << endl;
  }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
  //    Test::Deblurring::BlurKernel test1;

  PSF tst1;
  tst1.height = 10;
  tst1.width = 10;

  tst1.buffer = (float*)malloc(tst1.height * tst1.width * sizeof(float));

  BlurKernel test10(tst1);

  cout << "test10(PSF(10,10));" << endl;
  cout << test10.isEmpty() << endl;
  cout << test10.getWidth() << endl;
  cout << test10.getHeight() << endl;
  cout << test10.getData() << endl;
  printBlurKernel((PSF*)&test10);
  cout << endl;

  free(tst1.buffer);

  PSF tst2;

  BlurKernel test11(tst2);

  cout << "test11(PSF(empty));" << endl;
  cout << test11.isEmpty() << endl;
  cout << test11.getWidth() << endl;
  cout << test11.getHeight() << endl;
  cout << test11.getData() << endl;
  cout << endl;

  BlurKernel test1;  //(10,10);

  cout << "test1" << endl;
  cout << test1.isEmpty() << endl;
  cout << test1.getWidth() << endl;
  cout << test1.getHeight() << endl;
  cout << test1.getData() << endl;
  printBlurKernel((PSF*)&test1);
  cout << endl;

  test1.init(20, 20);

  cout << "test1.init(20,20)" << endl;
  cout << test1.isEmpty() << endl;
  cout << test1.getWidth() << endl;
  cout << test1.getHeight() << endl;
  cout << test1.getData() << endl;
  printBlurKernel((PSF*)&test1);
  cout << endl;

  test1.init(30, 30);
  // Set test1
  float* pData = test1.getData();
  for (int row = 0; row < test1.getHeight(); ++row) {
    for (int col = 0; col < test1.getWidth(); ++col) {
      *pData = row + col;
      pData++;
    }
  }

  cout << "reinit test1.init(30,30)" << endl;
  cout << test1.isEmpty() << endl;
  cout << test1.getWidth() << endl;
  cout << test1.getHeight() << endl;
  cout << test1.getData() << endl;
  printBlurKernel((PSF*)&test1);
  cout << endl;

  BlurKernel test2(test1);

  cout << "test2(test1)" << endl;
  cout << test2.isEmpty() << endl;
  cout << test2.getWidth() << endl;
  cout << test2.getHeight() << endl;
  cout << test2.getData() << endl;
  printBlurKernel((PSF*)&test2);
  cout << endl;

  test1.destroy();

  cout << "test1.destroy()" << endl;
  cout << test1.isEmpty() << endl;
  cout << test1.getWidth() << endl;
  cout << test1.getHeight() << endl;
  cout << test1.getData() << endl;
  cout << endl;

  test1 = test2;

  cout << "test1=test2" << endl;
  cout << test1.isEmpty() << endl;
  cout << test1.getWidth() << endl;
  cout << test1.getHeight() << endl;
  cout << test1.getData() << endl;
  printBlurKernel((PSF*)&test1);
  cout << endl;

  test2.init(10, 10);
  pData = test2.getData();
  for (int row = 0; row < test2.getHeight(); ++row) {
    for (int col = 0; col < test2.getWidth(); ++col) {
      *pData = row - col;  // 1.0;
      pData++;
    }
  }

  test1 = test2;

  cout << "test2.init(40,40);" << endl;
  cout << "test1 = test2;" << endl;
  cout << test1.isEmpty() << endl;
  cout << test1.getWidth() << endl;
  cout << test1.getHeight() << endl;
  cout << test1.getData() << endl;
  printBlurKernel((PSF*)&test1);
  cout << endl;

  test1.regularizeKernel();

  cout << "test1.regularizeKernel();" << endl;
  cout << test1.isEmpty() << endl;
  cout << test1.getWidth() << endl;
  cout << test1.getHeight() << endl;
  cout << test1.getData() << endl;
  printBlurKernel((PSF*)&test1);

  test1.normalizeKernel();

  cout << "test1.normalizeKernel();" << endl;
  cout << test1.isEmpty() << endl;
  cout << test1.getWidth() << endl;
  cout << test1.getHeight() << endl;
  cout << test1.getData() << endl;
  printBlurKernel((PSF*)&test1);

  test1.setKernelSize();
  test1.fillKernel();
  cout << "test1.normalizeKernel();" << endl;
  cout << test1.isEmpty() << endl;
  cout << test1.getWidth() << endl;
  cout << test1.getHeight() << endl;
  cout << test1.getData() << endl;
  printBlurKernel((PSF*)&test1);

  test1.clear();

  cout << "test1.clear();" << endl;
  cout << test1.isEmpty() << endl;
  cout << test1.getWidth() << endl;
  cout << test1.getHeight() << endl;
  cout << test1.getData() << endl;
  printBlurKernel((PSF*)&test1);
  cout << endl;

  return 0;
}
