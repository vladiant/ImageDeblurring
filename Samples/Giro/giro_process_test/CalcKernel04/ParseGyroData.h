/*
 * ParseGyroData.h
 *
 *  Created on: Apr 25, 2012
 *      Author: vantonov
 */

#pragma once

#include "DiscreteAngularVelocitiesIntegrator.h"

class ParseGyroData {
 private:
  struct DataSample {
    Test::Time timestamp;
    float x;
    float y;
    float z;
    int extra;
  };
  typedef struct DataSample DataSample;

  struct MovementDataHeader {
    char md_identifier[16];
    Test::Time md_capture_timestamp;
    unsigned int total_gyro_samples;
    unsigned int last_sample_index;
    Test::Time capture_command_timestamp;
  };

  typedef struct MovementDataHeader MovementDataHeader;

  static const unsigned int segment_len = 16384;

  char GYRO_HEAD[9];

  DataSample *gyroSamples;

  MovementDataHeader *gyroDataHeader;

  // name of the file
  FILE *pFile;

  unsigned int total_samples, last_sample;

  // buffer for data storage
  char *buf;

  // error code
  int ErrCode{};

  // capture timestamp
  Test::Time CaptTimestamp;

  // angular velocity x [rad/s]
  float *wx;

  // angular velocity y [rad/s]
  float *wy;

  // angular velocity z [rad/s]
  float *wz;

  // timestamp [ns]
  // Test::Time *timestamp;

 public:
  // timestamp [ns]
  Test::Time *timestamp;

  ParseGyroData(char *FileName);
  ~ParseGyroData();

  /* Status of the operation:
   * 0 - no errors
   * 1 - file can not be open
   * 2 - file contains no gyro data
   */
  int Status(void);

  // returns the number of total samples
  unsigned int TotalSamples(void);

  // returns capture timestamp
  Test::Time CaptureTimestamp(void);

  // returns an element with index i from timestamp and angular velocities
  // arrays
  Test::Time TimeStamp(int i);
  float Wx(int i);
  float Wy(int i);
  float Wz(int i);
};

ParseGyroData::ParseGyroData(char *FileName) {
  // position for data transfer
  int j = 0;

  strcpy(GYRO_HEAD, "GYRODATA");

  pFile = fopen(FileName, "r");

  if (pFile == NULL) {
    ErrCode = 1;  // File can not be open
  } else {
    buf = new char[segment_len];
    gyroDataHeader = (MovementDataHeader *)malloc(sizeof(MovementDataHeader));

    fseek(pFile, 0, SEEK_END);
    long size = ftell(pFile);
    size -= segment_len;
    fseek(pFile, size, SEEK_SET);
    fread(buf, sizeof(char), segment_len, pFile);

    memcpy(gyroDataHeader, buf, sizeof(MovementDataHeader));

    if (strcmp(gyroDataHeader->md_identifier, GYRO_HEAD) == 0) {
      total_samples = gyroDataHeader->total_gyro_samples;

      // set the capture end timestamp
      CaptTimestamp = gyroDataHeader->md_capture_timestamp;

      last_sample = gyroDataHeader->last_sample_index;

      gyroSamples = (DataSample *)malloc(sizeof(DataSample) * total_samples);

      memcpy(gyroSamples, buf + sizeof(MovementDataHeader),
             sizeof(DataSample) * total_samples);

      // set timestamp and angular velocities
      wx = new float[total_samples];
      wy = new float[total_samples];
      wz = new float[total_samples];
      timestamp = new Test::Time[total_samples];

      // read the values from file
      for (unsigned int i = last_sample; i < total_samples; i++, j++) {
        timestamp[j] = gyroSamples[i].timestamp;
        wx[j] = gyroSamples[i].x;
        wy[j] = gyroSamples[i].y;
        wz[j] = gyroSamples[i].z;
      }

      for (unsigned int i = 0; i < last_sample; i++, j++) {
        timestamp[j] = gyroSamples[i].timestamp;
        wx[j] = gyroSamples[i].x;
        wy[j] = gyroSamples[i].y;
        wz[j] = gyroSamples[i].z;
      }

      delete[] buf;
      free(gyroSamples);
      free(gyroDataHeader);
    } else {
      ErrCode = 2;  // no gyro data in file
    }

    // data transfer completed, release memory
    fclose(pFile);
  }
}

ParseGyroData::~ParseGyroData() {
  // release the memory
  delete[] timestamp;
  delete[] wx;
  delete[] wy;
  delete[] wz;
}

// status of the operation
int ParseGyroData::Status(void) { return (ErrCode); }

// returns the number of total samples
unsigned int ParseGyroData::TotalSamples(void) { return (total_samples); }

// returns capture timestamp
Test::Time ParseGyroData::CaptureTimestamp(void) { return (CaptTimestamp); }

// timestamp element
Test::Time ParseGyroData::TimeStamp(int i) { return (timestamp[i]); }

// angular velocity x
float ParseGyroData::Wx(int i) { return (wx[i]); }

// angular velocity y
float ParseGyroData::Wy(int i) { return (wy[i]); }

// angular velocity z
float ParseGyroData::Wz(int i) { return (wz[i]); }
