
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <iostream>

struct DataSample {
  unsigned long long int timestamp;
  float x;
  float y;
  float z;
  int extra;
};
typedef struct DataSample DataSample;

struct MovementDataHeader {
  char md_identifier[16];
  long long int md_capture_timestamp;
  unsigned int total_gyro_samples;
  unsigned int last_sample_index;
  long long int capture_command_timestamp;
};

typedef struct MovementDataHeader MovementDataHeader;

unsigned int segment_len = 16384;

const char delim[] = {' ', ';', '\t'};
int delim_idx = 0;

DataSample *gyroSamples;
MovementDataHeader *gyroDataHeader;

int main(int argc, char *argv[]) {
  FILE *pFile;
  int filesize;
  unsigned int total_samples, last_sample;
  char *buf;

  if (argc < 2) {
    printf("\nUsage: parsegyro <jpeg file> <option>\n");
    printf("\t<option>: -d  use ; delimiter\n");
    printf("\t          -t  use tab delimiter\n");
    printf("By default uses space delimiter\n\n");
    return 0;
  }

  if (argc == 3) {
    if (strcmp(argv[2], "-d") == 0) delim_idx = 1;
    if (strcmp(argv[2], "-t") == 0) delim_idx = 2;
  }

  pFile = fopen(argv[1], "r");

  if (pFile == NULL) {
    printf("\nFile couldn't be opened.\n");
    return 0;
  }

  buf = new char[segment_len];
  gyroDataHeader = (MovementDataHeader *)malloc(sizeof(MovementDataHeader));

  fseek(pFile, 0, SEEK_END);
  long size = ftell(pFile);
  size -= segment_len;
  fseek(pFile, size, SEEK_SET);
  fread(buf, sizeof(char), segment_len, pFile);

  memcpy(gyroDataHeader, buf, sizeof(MovementDataHeader));

  total_samples = gyroDataHeader->total_gyro_samples;

  last_sample = gyroDataHeader->last_sample_index;

  gyroSamples = (DataSample *)malloc(sizeof(DataSample) * total_samples);

  memcpy(gyroSamples, buf + sizeof(MovementDataHeader),
         sizeof(DataSample) * total_samples);

  printf("\n");
  for (int i = last_sample; i < total_samples; i++) {
    std::cout << gyroSamples[i].timestamp - gyroSamples[last_sample].timestamp
              << " " << gyroSamples[i].x << " " << gyroSamples[i].y << " "
              << gyroSamples[i].z << "\n";
  }

  for (int i = 0; i < last_sample; i++) {
    std::cout << gyroSamples[i].timestamp - gyroSamples[last_sample].timestamp
              << " " << gyroSamples[i].x << " " << gyroSamples[i].y << " "
              << gyroSamples[i].z << "\n";
  }

  std::cout << "\ncapture timestamp "
            << gyroDataHeader->md_capture_timestamp -
                   gyroSamples[last_sample].timestamp
            << "\n";
  std::cout << "user press timestamp = "
            << gyroDataHeader->capture_command_timestamp -
                   gyroSamples[last_sample].timestamp
            << "\n";
  std::cout << "total gyro samples " << gyroDataHeader->total_gyro_samples
            << "\n";

  free(gyroSamples);
  free(gyroDataHeader);
  fclose(pFile);
}
