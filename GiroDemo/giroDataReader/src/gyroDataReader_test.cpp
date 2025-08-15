
/**
 *
 * @file gyroDataReader_test.cpp
 *
 * ^path /gyroDataReader/cpp/gyroDataReader_test.cpp
 *
 * @author Vladislav Antonov
 *
 * @date April 12, 2012
 *
 * @version 1.00
 *
 */
/* -----------------------------------------------------------------------------
 *!
 *! Revision History
 *! ===================================
 *! April 12, 2012 : Vladislav Antonov
 *! Created.
 *!
 * ===========================================================================
 */

#include "gyroDataReader.h"

#include <stdlib.h>

#include <iostream>

#define IMAGE_FILENAME "IMG_20000109_195543.jpg"
#define TEXT_DATA_FILENAME "IMG_20000109_195543.jpg.txt"
#define EXPOSURE_DURATION 100000000

using namespace Test::Deblurring;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "\nProgram to read giro data from image \nand from the "
                 "parsegyro data file.\n"
              << std::endl;
    std::cout << "Usage: " << argv[0] << "  image_file  parsegyro_data_file";
    return 0;
  }

  // Sample calibration data
  Versor<float> gyroSpaceToCameraSpaceQuaternion =
      Versor<float>(-0.0470602, 0.698666, 0.71093, -0.0650423);
  bool isZAxisInvertedInGyroSpace = true;

  std::cout << "Starting gyro data reading tests" << std::endl;

  // Reading from JPEG file
  {
    std::string imageFilename(argv[1]);

    Test::Deblurring::GyroDataReader gyroDataReader(200);

    std::cout << "Reading data from image file: " << argv[1] << std::endl;

    gyroDataReader.readDataFromJPEGFile(imageFilename);

    int numberRawSamples = gyroDataReader.getNumberOfGyroSamples();

    Test::Time* timeStampRaw = new Test::Time[numberRawSamples];
    float* wXraw = new float[numberRawSamples];
    float* wYraw = new float[numberRawSamples];
    float* wZraw = new float[numberRawSamples];

    std::cout << "Raw samples read: " << std::endl;

    gyroDataReader.getAngularVelocities(timeStampRaw, wXraw, wYraw, wZraw);

    for (int i = 0; i < numberRawSamples; i++) {
      std::cout << i << "\ttimeStampRaw: " << timeStampRaw[i]
                << "\twXraw: " << wXraw[i] << "\twYraw: " << wYraw[i]
                << "\twZraw: " << wZraw[i] << std::endl;
    }

    delete[] timeStampRaw;
    delete[] wXraw;
    delete[] wYraw;
    delete[] wZraw;

    std::cout << "Correcting angular velocities. " << std::endl;

    gyroDataReader.correctAngularVelocities(gyroSpaceToCameraSpaceQuaternion,
                                            isZAxisInvertedInGyroSpace);

    std::cout << "Reading capture timestamp. " << std::endl;

    Test::Time captureTimeStamp = gyroDataReader.getCaptureTimestamp();

    std::cout << "Capture timestamp: " << captureTimeStamp << std::endl;

    int captureTimeStampIndex = gyroDataReader.getCaptureTimestampIndex();

    std::cout << "Capture timestamp index: " << captureTimeStampIndex
              << std::endl;

    std::cout << "Reading angular velocities. " << std::endl;

    int numberSamples = gyroDataReader.getNumberOfGyroSamples();

    std::cout << "numberSamples: " << numberSamples << std::endl;

    Test::Time* timeStamp = new Test::Time[numberSamples];
    float* wX = new float[numberSamples];
    float* wY = new float[numberSamples];
    float* wZ = new float[numberSamples];

    gyroDataReader.getAngularVelocities(timeStamp, wX, wY, wZ);

    std::cout << "Samples read: " << std::endl;

    for (int i = 0; i < numberSamples; i++) {
      std::cout << i << "\ttimeStamp: " << timeStamp[i] << "\twX: " << wX[i]
                << "\twY: " << wY[i] << "\twZ: " << wZ[i] << std::endl;

      Test::Time nextTimeStamp;

      if (i < numberSamples - 1) {
        nextTimeStamp = timeStamp[i + 1];
      }

      if ((captureTimeStamp >= timeStamp[i]) &&
          (captureTimeStamp <= nextTimeStamp)) {
        std::cout << ">>> CAPTURE <<<" << std::endl;
      }
    }

    int numberSamplesDuringExposure =
        gyroDataReader.calcNumberOfGyroSamplesDuringExposure(EXPOSURE_DURATION);

    std::cout << "\nnumberSamplesDuringExposure: "
              << numberSamplesDuringExposure << std::endl;

    Test::Time* timeStampExposure = new Test::Time[numberSamplesDuringExposure];
    float* wXexposure = new float[numberSamplesDuringExposure];
    float* wYexposure = new float[numberSamplesDuringExposure];
    float* wZexposure = new float[numberSamplesDuringExposure];

    gyroDataReader.getAngularVelocitiesDuringExposure(
        EXPOSURE_DURATION, timeStampExposure, wXexposure, wYexposure,
        wZexposure);

    for (int i = 0; i < numberSamplesDuringExposure; i++) {
      std::cout << i << "\ttimeStampExposure: " << timeStampExposure[i]
                << "\twXexposure: " << wXexposure[i]
                << "\twYexposure: " << wYexposure[i]
                << "\twZexposure: " << wZexposure[i] << std::endl;
    }

    delete[] timeStampExposure;
    delete[] wXexposure;
    delete[] wYexposure;
    delete[] wZexposure;

    delete[] timeStamp;
    delete[] wX;
    delete[] wY;
    delete[] wZ;
  }

  // Reading from parsed file
  {
    std::string textDataFilename(argv[2]);

    Test::Deblurring::GyroDataReader gyroDataReader(200);

    std::cout << "Reading data from image file: " << argv[1] << std::endl;

    gyroDataReader.readDataFromParsedTextFile(textDataFilename);

    int numberRawSamples = gyroDataReader.getNumberOfGyroSamples();

    Test::Time* timeStampRaw = new Test::Time[numberRawSamples];
    float* wXraw = new float[numberRawSamples];
    float* wYraw = new float[numberRawSamples];
    float* wZraw = new float[numberRawSamples];

    std::cout << "Raw samples read: " << std::endl;

    gyroDataReader.getAngularVelocities(timeStampRaw, wXraw, wYraw, wZraw);

    for (int i = 0; i < numberRawSamples; i++) {
      std::cout << i << "\ttimeStampRaw: " << timeStampRaw[i]
                << "\twXraw: " << wXraw[i] << "\twYraw: " << wYraw[i]
                << "\twZraw: " << wZraw[i] << std::endl;
    }

    delete[] timeStampRaw;
    delete[] wXraw;
    delete[] wYraw;
    delete[] wZraw;

    std::cout << "Correcting angular velocities. " << std::endl;

    gyroDataReader.correctAngularVelocities(gyroSpaceToCameraSpaceQuaternion,
                                            isZAxisInvertedInGyroSpace);

    std::cout << "Reading capture timestamp. " << std::endl;

    Test::Time captureTimeStamp = gyroDataReader.getCaptureTimestamp();

    std::cout << "Capture timestamp: " << captureTimeStamp << std::endl;

    int captureTimeStampIndex = gyroDataReader.getCaptureTimestampIndex();

    std::cout << "Capture timestamp index: " << captureTimeStampIndex
              << std::endl;

    std::cout << "Reading angular velocities. " << std::endl;

    int numberSamples = gyroDataReader.getNumberOfGyroSamples();

    std::cout << "numberSamples: " << numberSamples << std::endl;

    Test::Time* timeStamp = new Test::Time[numberSamples];
    float* wX = new float[numberSamples];
    float* wY = new float[numberSamples];
    float* wZ = new float[numberSamples];

    gyroDataReader.getAngularVelocities(timeStamp, wX, wY, wZ);

    std::cout << "Samples read: " << std::endl;

    for (int i = 0; i < numberSamples; i++) {
      std::cout << i << "\ttimeStamp: " << timeStamp[i] << "\twX: " << wX[i]
                << "\twY: " << wX[i] << "\twZ: " << wX[i] << std::endl;

      Test::Time nextTimeStamp;

      if (i < numberSamples - 1) {
        nextTimeStamp = timeStamp[i + 1];
      }

      if ((captureTimeStamp >= timeStamp[i]) &&
          (captureTimeStamp <= nextTimeStamp)) {
        std::cout << ">>> CAPTURE <<<" << std::endl;
      }
    }

    int numberSamplesDuringExposure =
        gyroDataReader.calcNumberOfGyroSamplesDuringExposure(EXPOSURE_DURATION);

    std::cout << "\nnumberSamplesDuringExposure: "
              << numberSamplesDuringExposure << std::endl;

    Test::Time* timeStampExposure = new Test::Time[numberSamplesDuringExposure];
    float* wXexposure = new float[numberSamplesDuringExposure];
    float* wYexposure = new float[numberSamplesDuringExposure];
    float* wZexposure = new float[numberSamplesDuringExposure];

    gyroDataReader.getAngularVelocitiesDuringExposure(
        EXPOSURE_DURATION, timeStampExposure, wXexposure, wYexposure,
        wZexposure);

    for (int i = 0; i < numberSamplesDuringExposure; i++) {
      std::cout << i << "\ttimeStampExposure: " << timeStampExposure[i]
                << "\twXexposure: " << wXexposure[i]
                << "\twYexposure: " << wYexposure[i]
                << "\twZexposure: " << wZexposure[i] << std::endl;
    }

    delete[] timeStamp;
    delete[] wX;
    delete[] wY;
    delete[] wZ;
  }

  std::cout << "Done." << std::endl;

  return EXIT_SUCCESS;
}
