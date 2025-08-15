
/**
 *
 * @file gyroDataReader.cpp
 *
 * ^path /gyroDataReader/cpp/gyroDataReader.cpp
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

#include <stdio.h>

#include <fstream>
#include <iostream>

#define GYRO_DATA_SEGMENT_LENGTH 16384
#define MAXIMUM_TOLERATED_DELAY 10000

/// Samples to be used for averaging
#define SAMPLES_TO_AVERAGE 20

using namespace Test::OpticalFlow;
using namespace Test::Math;

namespace Test {
namespace Deblurring {

GyroDataReader::GyroDataReader(int numberGyroSamples)
    : status(GYRO_DATA_READER_NOT_INITIALIZED),
      pGyroSamples(NULL),
      pGyroCorrector(NULL),
      pGyroDataHeader(NULL),
      total_samples(0),
      last_sample(0),
      capture_timestamp(0),
      captureTimestampIndex(-1),
      wx(NULL),
      wy(NULL),
      wz(NULL),
      timestamp(NULL) {
  init(numberGyroSamples);

  status = GYRO_DATA_READER_INIT_OK;

  return;
}

GyroDataReader::~GyroDataReader() {
  destroyGyroDataContainer();
  deinit();
}

void GyroDataReader::init(int numberGyroSamples) {
  try {
    // Initialize discrete angular velocities integrator.
    pAngularVelocityIntegrator =
        new DiscreteAngularVelocitiesIntegrator<Versor<float>, float>(
            numberGyroSamples);

    // Initialize container for the gyro data samples.
    pGyroDataHeader = new MovementDataHeader;

    /// Initialize gyro data correction class - first value is tolerance in ns,
    /// second - N samples for averaging; this means the first N samples will be
    /// skipped
    pGyroCorrector = new GyroDataCorrection<float>(MAXIMUM_TOLERATED_DELAY,
                                                   SAMPLES_TO_AVERAGE);

  } catch (std::bad_alloc& e) {
    status = GYRO_DATA_READER_MEM_ALLOC_FAIL;
    deinit();
  }
}

void GyroDataReader::deinit() {
  // Release the buffers and clear the pointers.
  delete pAngularVelocityIntegrator;
  pAngularVelocityIntegrator = NULL;
  delete pGyroDataHeader;
  pGyroDataHeader = NULL;
  delete pGyroCorrector;
  pGyroCorrector = NULL;
}

int GyroDataReader::readDataFromJPEGFile(std::string& fileName) {
  char* buf;

  try {
    buf = new char[GYRO_DATA_SEGMENT_LENGTH];
  } catch (std::bad_alloc& e) {
    status = GYRO_DATA_READER_MEM_ALLOC_FAIL;
    return GYRO_DATA_READER_MEM_ALLOC_FAIL;
  }

  FILE* pFile = fopen(fileName.c_str(), "r");

  if (pFile == NULL) {
    delete[] buf;
    status = GYRO_DATA_ERROR_FILE_OPEN;
    return GYRO_DATA_ERROR_FILE_OPEN;
  }

  // Read the end of the file.
  fseek(pFile, 0, SEEK_END);
  long size = ftell(pFile);
  size -= GYRO_DATA_SEGMENT_LENGTH;
  fseek(pFile, size, SEEK_SET);

  if (fread(buf, sizeof(char), GYRO_DATA_SEGMENT_LENGTH, pFile) !=
      GYRO_DATA_SEGMENT_LENGTH) {
    delete[] buf;
    fclose(pFile);
    status = GYRO_DATA_ERROR_FILE_READ;
    return GYRO_DATA_ERROR_FILE_READ;
  }

  fclose(pFile);

  memcpy(pGyroDataHeader, buf, sizeof(MovementDataHeader));

  // Check the header.
  if (strcmp(pGyroDataHeader->md_identifier, GYRO_DATA_HEADER_ID) != 0) {
    delete[] buf;
    status = GYRO_DATA_WRONG_DATA_TYPE;
    return GYRO_DATA_WRONG_DATA_TYPE;
  }

  // Initialize gyro data parameters.
  capture_timestamp = pGyroDataHeader->md_capture_timestamp;
  last_sample = pGyroDataHeader->last_sample_index;

  initGyroDataContainer(pGyroDataHeader->total_gyro_samples,
                        buf + sizeof(MovementDataHeader));

  delete[] buf;

  return GYRO_DATA_READER_INIT_OK;
}

int GyroDataReader::checkParsedTextFile(std::string& fileName,
                                        int& numberOfGyroSamples,
                                        Time& captureTimestamp) {
  int parsedFileStatus = GYRO_DATA_READER_INIT_OK;

  // string constants to be searched in data file
  const char CAPTURE_MARK[] = "capture timestamp";
  const char GYRO_SAMPLES_MARK[] = "total gyro samples";

  // number of rows in data file
  int dataFileRows;

  // temporal variable for lines in text file
  std::string lineInFile;

  dataFileRows = 0;

  std::ifstream gyroDataFile(fileName.c_str());

  if (gyroDataFile.is_open()) {
    // check flags for availability of the data
    bool isCaptureTimestampFound = false, isNumberOfGyroSamplesFound = false;

    while (gyroDataFile.good()) {
      char* pStringPosition;  // pointer to string position
      dataFileRows++;
      getline(gyroDataFile, lineInFile);

      // find the capture timestamp
      pStringPosition = strstr(&lineInFile[0], CAPTURE_MARK);
      if (pStringPosition != NULL) {
        char* pEnd;
        pStringPosition += strlen(CAPTURE_MARK);  //
        capture_timestamp = strtol(pStringPosition, &pEnd, 10);
        isCaptureTimestampFound = true;
      }

      // find the gyro samples number
      pStringPosition = strstr(&lineInFile[0], GYRO_SAMPLES_MARK);
      if (pStringPosition != NULL) {
        pStringPosition += strlen(GYRO_SAMPLES_MARK);  //
        numberOfGyroSamples = atoi(pStringPosition);
        isNumberOfGyroSamplesFound = true;
      }
    }
    gyroDataFile.close();

    if (!(isCaptureTimestampFound)) {
      return GYRO_DATA_CAPTURE_MARK_NOT_FOUND;
    }

    if (!(isNumberOfGyroSamplesFound)) {
      return GYRO_DATA_GYRO_SAMPLES_COUNT_NOT_FOUND;
    }

  } else {
    parsedFileStatus = GYRO_DATA_ERROR_FILE_OPEN;
  }

  return parsedFileStatus;
}

int GyroDataReader::readDataFromParsedTextFile(std::string& fileName) {
  int numberOfGyroSamples;
  Time captureTimestamp;

  int status =
      checkParsedTextFile(fileName, numberOfGyroSamples, captureTimestamp);

  if (status == GYRO_DATA_READER_INIT_OK) {
    last_sample = numberOfGyroSamples;

    status = createGyroDataContainer(numberOfGyroSamples);

    if (status == GYRO_DATA_READER_INIT_OK) {
      // read the values from file
      std::ifstream gyroDataFile(fileName.c_str());
      for (int j = 0; j < numberOfGyroSamples; j++) {
        gyroDataFile >> pGyroSamples[j].timestamp >> pGyroSamples[j].x >>
            pGyroSamples[j].y >> pGyroSamples[j].z;
        // No extra data
        pGyroSamples[j].extra = 0;
      }

      copyRawGyroDataToOutput();

      gyroDataFile.close();
    }
  }

  return GYRO_DATA_READER_INIT_OK;
}

int GyroDataReader::createGyroDataContainer(unsigned int totalSamples) {
  // Reallocates array for gyro data if needed.
  if (totalSamples != this->total_samples) {
    destroyGyroDataContainer();

    try {
      pGyroSamples = new GyroDataSample[totalSamples];
      wx = new float[totalSamples];
      wy = new float[totalSamples];
      wz = new float[totalSamples];
      timestamp = new Time[totalSamples];

    } catch (std::bad_alloc& e) {
      destroyGyroDataContainer();
      status = GYRO_DATA_READER_MEM_ALLOC_FAIL;
      return GYRO_DATA_READER_MEM_ALLOC_FAIL;
    }

    total_samples = totalSamples;
  }

  return GYRO_DATA_READER_INIT_OK;
}

int GyroDataReader::initGyroDataContainer(unsigned int totalSamples,
                                          void* gyroDataBuffer) {
  int retVal = createGyroDataContainer(totalSamples);

  if (retVal == GYRO_DATA_READER_INIT_OK) {
    memcpy(pGyroSamples, gyroDataBuffer,
           sizeof(GyroDataSample) * total_samples);

    //    GyroDataSample* pData = (GyroDataSample*) gyroDataBuffer;
    //
    //    for (unsigned int i = 0; i < totalSamples; i++) {
    //
    //      pGyroSamples[i].timestamp = pData->timestamp;
    //      pGyroSamples[i].x = pData->x;
    //      pGyroSamples[i].y = pData->y;
    //      pGyroSamples[i].z = pData->z;
    //      pGyroSamples[i].extra = pData->extra;
    //
    //      pData++;
    //    }  // for( ...

    copyRawGyroDataToOutput();

  }  // if (retVal ...

  return retVal;
}

int GyroDataReader::initGyroDataContainer(unsigned int totalSamples,
                                          Time* timestamp, float* x, float* y,
                                          float* z, int* extra) {
  int retVal = createGyroDataContainer(totalSamples);

  // Load the data from input array.
  if (retVal == GYRO_DATA_READER_INIT_OK) {
    for (unsigned int i = 0; i < totalSamples; i++) {
      pGyroSamples[i].timestamp = timestamp[i];
      pGyroSamples[i].x = x[i];
      pGyroSamples[i].y = y[i];
      pGyroSamples[i].z = z[i];
      if (extra != NULL) {
        pGyroSamples[i].extra = extra[i];
      }
    }
  }

  copyRawGyroDataToOutput();

  return retVal;
}

void GyroDataReader::destroyGyroDataContainer() {
  delete[] pGyroSamples;
  pGyroSamples = NULL;
  delete[] wx;
  wx = NULL;
  delete[] wy;
  wy = NULL;
  delete[] wz;
  wz = NULL;
  delete[] timestamp;
  timestamp = NULL;
}

void GyroDataReader::copyRawGyroDataToOutput() {
  int j = 0;

  for (unsigned int i = last_sample; i < total_samples; i++, j++) {
    timestamp[j] = pGyroSamples[i].timestamp;
    wx[j] = pGyroSamples[i].x;
    wy[j] = pGyroSamples[i].y;
    wz[j] = pGyroSamples[i].z;
  }

  for (unsigned int i = 0; i < last_sample; i++, j++) {
    timestamp[j] = pGyroSamples[i].timestamp;
    wx[j] = pGyroSamples[i].x;
    wy[j] = pGyroSamples[i].y;
    wz[j] = pGyroSamples[i].z;
  }
}

void GyroDataReader::correctAngularVelocities(
    Versor<float>& gyroSpaceToCameraSpaceQuaternion,
    bool isZAxisInvertedInGyroSpace) {
  int j = 0;

  // Gyro axis correction
  for (unsigned int i = last_sample; i < total_samples; i++, j++) {
    if (isZAxisInvertedInGyroSpace) {
      wz[j] *= -1.0;
    }
  }

  for (unsigned int i = 0; i < last_sample; i++, j++) {
    if (isZAxisInvertedInGyroSpace) {
      wz[j] *= -1.0;
    }
  }

  // Data correction and calculation of rotation versors.
  for (unsigned int j = 0; j < total_samples; j++) {
    GyroDataCorrection<float>::GyroSample_t gyroSample(
        timestamp[j], Vector<float, 3>(wx[j], wy[j], wz[j]));

    // Correct the displacement between the gyro and image sensor.
    gyroSpaceToCameraSpaceQuaternion.rotateVector(gyroSample.velocity);

    // Correct the gyro data.
    calcAngularVelocity(*pGyroCorrector, gyroSample);

    // Uncomment this to add the samples without correction and comment the
    // upper line.
    //    pAngularVelocityIntegrator->addAngularVelocity(gyroSample);
  }

  calcTimeStampIndex();
}

void GyroDataReader::calcTimeStampIndex() {
  for (unsigned int j = 0; j < total_samples; j++) {
    Test::Time nextTimeStamp;

    if (j < total_samples - 1) {
      nextTimeStamp = timestamp[j + 1];
    } else {
      nextTimeStamp = timestamp[j] + 1000;
    }

    if ((capture_timestamp >= timestamp[j]) &&
        (capture_timestamp <= nextTimeStamp)) {
      captureTimestampIndex = j;
      break;
    }
  }
}

unsigned int GyroDataReader::calcNumberOfGyroSamplesDuringExposure(
    Time exposureDuration) {
  unsigned int numberOfSamplesDuringExposure = 0;
  Time exposureEndTimeStamp = capture_timestamp + exposureDuration;

  for (unsigned int j = 0; j < total_samples; j++) {
    if ((timestamp[j] >= capture_timestamp) &&
        (timestamp[j] <= exposureEndTimeStamp)) {
      numberOfSamplesDuringExposure++;
    }
  }

  return numberOfSamplesDuringExposure;
}

void GyroDataReader::getAngularVelocities(Time* timeStamp, float* wX, float* wY,
                                          float* wZ) {
  for (unsigned int i = 0; i < total_samples; i++) {
    timeStamp[i] = timestamp[i];
    wX[i] = wx[i];
    wY[i] = wy[i];
    wZ[i] = wz[i];
  }
}

void GyroDataReader::getAngularVelocitiesDuringExposure(Time exposureDuration,
                                                        Time* timeStamp,
                                                        float* wX, float* wY,
                                                        float* wZ) {
  Time exposureEndTimeStamp = capture_timestamp + exposureDuration;
  int j = 0;

  for (unsigned int i = 0; i < total_samples; i++) {
    if ((timestamp[i] >= capture_timestamp) &&
        (timestamp[i] <= exposureEndTimeStamp)) {
      timeStamp[j] = timestamp[i];
      wX[j] = wx[i];
      wY[j] = wy[i];
      wZ[j] = wz[i];
      j++;
    }
  }
}

void GyroDataReader::calcAngularVelocity(
    GyroDataCorrection<float>& gyroCorrector,
    GyroDataCorrection<float>::GyroSample_t gyroSample) {
  if (gyroCorrector.startup(gyroSample)) {
    while (gyroCorrector.sweepForInsertOrCorrect(gyroSample)) {
      GyroDataCorrection<float>::GyroSample_t gyroSample2 = gyroCorrector.get();
      // Recursive call of this function
      calcAngularVelocity(gyroCorrector, gyroSample2);
    }

    // Corrected angular velocity added to container
    pAngularVelocityIntegrator->addAngularVelocity(gyroSample);
  }
}

}  // namespace Deblurring
}  // namespace Test
