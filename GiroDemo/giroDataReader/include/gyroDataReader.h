
/**
 *
 * @file gyroDataReader.h
 *
 * ^path /gyroDataReader/include/gyroDataReader.h
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

#ifndef GYRODATAREADER_H_
#define GYRODATAREADER_H_

#include <string.h>

// video stabilization libraries
#include <DiscreteAngularVelocitiesIntegrator.h>
#include <GyroDataCorrection.h>
#include <Time.h>
#include <Versor.h>

#define DEFAULT_NUMBER_GYRO_SAMPLES 200
#define GYRO_DATA_HEADER_ID "GYRODATA"

using namespace Test::OpticalFlow;
using namespace Test::Math;

namespace Test {
namespace Deblurring {

/// Gyro Data Sample
typedef struct {
  Time timestamp;
  float x;
  float y;
  float z;
  int extra;
} GyroDataSample;

/// Header of the gyro data embedded in JPG file.
typedef struct {
  char md_identifier[16];
  Time md_capture_timestamp;
  unsigned int total_gyro_samples;
  unsigned int last_sample_index;
  Time capture_command_timestamp;
} MovementDataHeader;

class GyroDataReader {
 public:
  /// Status of the initialization
  enum {
    GYRO_DATA_GYRO_SAMPLES_COUNT_NOT_FOUND =
        -7,  ///< Capture timestamp mark not found.
    GYRO_DATA_CAPTURE_MARK_NOT_FOUND =
        -6,                          ///< Capture timestamp mark not found.
    GYRO_DATA_WRONG_DATA_TYPE = -5,  ///< The data in block is not gyro samples.
    GYRO_DATA_ERROR_FILE_READ = -4,  ///< Error reading the file.
    GYRO_DATA_ERROR_FILE_OPEN = -3,  ///< Error opening the file.
    GYRO_DATA_READER_NOT_INITIALIZED = -2,  ///< Context not initialized.
    GYRO_DATA_READER_MEM_ALLOC_FAIL = -1,   ///< Memory allocation failed.
    GYRO_DATA_READER_INIT_OK = 0,           ///< Initialization OK.
  };

  /// Constructor with specified number of gyro samples.
  GyroDataReader(int numberGyroSamples = DEFAULT_NUMBER_GYRO_SAMPLES);

  /// Default destructor.
  ~GyroDataReader();

  /// Return the status of the last operation.
  int getStatus() { return status; }

  /// Read gyro data from JPEG file
  int readDataFromJPEGFile(std::string& fileName);

  /// Read gyro data from text file
  int readDataFromParsedTextFile(std::string& fileName);

  /// Checks text file with extracted data.
  int checkParsedTextFile(std::string& fileName, int& numberOfGyroSamples,
                          Time& captureTimestamp);

  /// Initialize the gyro data parameters.
  int createGyroDataContainer(unsigned int totalSamples);

  /// Initialize the gyro data parameters from buffer.
  int initGyroDataContainer(unsigned int totalSamples, void* gyroDataBuffer);

  /**
   * @brief Initialize the gyro data parameters from data arrays.
   *
   * The extra array may be sent to NULL in order to be skipped.
   *
   * @param totalSamples Number of Gyro samples.
   * @param timestamp Pointer to array of timestamps.
   * @param x Pointer to gyro data X component.
   * @param y Pointer to gyro data Y component.
   * @param y Pointer to gyro data Z component.
   * @param extra Pointer to extra data.
   *
   */
  int initGyroDataContainer(unsigned int totalSamples, Time* timestamp,
                            float* x, float* y, float* z, int* extra);

  /// Corrects angular velocities to timesamples positions.
  void correctAngularVelocities(Versor<float>& gyroSpaceToCameraSpaceQuaternion,
                                bool isZAxisInvertedInGyroSpace);

  unsigned int getNumberOfGyroSamples() { return total_samples; }

  /**
   * @brief Calculates the number of gyro data samples from the capture start
   * for the exposure duration.
   *
   * The length of the array is calculated in a way to include the capture start
   * (between first and second sample) and exposure end (between last two
   * samples) moments. So the length will be actually longer than the exposure.
   */
  unsigned int calcNumberOfGyroSamplesDuringExposure(Time exposureDuration);

  Time getCaptureTimestamp() { return capture_timestamp; }

  int getCaptureTimestampIndex() { return captureTimestampIndex; }

  /**
   * @brief Fills the angular velocities in supplied buffers.
   *
   * If any of the buffers is set to NULL, it will not be filled.
   *
   * @param timestamp Pointer to allocated buffer for the timestamps.
   * @param wX Pointer to allocated buffer for the X component of angular
   * velocity.
   * @param wY Pointer to allocated buffer for the Y component of angular
   * velocity.
   * @param wZ Pointer to allocated buffer for the Z component of angular
   * velocity.
   */
  void getAngularVelocities(Time* timeStamp, float* wX, float* wY, float* wZ);

  /**
   * @brief Fills the angular velocities during exposure in supplied buffers.
   *
   * If any of the buffers is set to NULL, it will not be filled.
   *
   * @param exposureDuration Exposure duration in Time units.
   * @param timestamp Pointer to allocated buffer for the timestamps.
   * @param wX Pointer to allocated buffer for the X component of angular
   * velocity.
   * @param wY Pointer to allocated buffer for the Y component of angular
   * velocity.
   * @param wZ Pointer to allocated buffer for the Z component of angular
   * velocity.
   */
  void getAngularVelocitiesDuringExposure(Time exposureDuration,
                                          Time* timeStamp, float* wX, float* wY,
                                          float* wZ);

 private:
  /// Container for the gyro data.
  DiscreteAngularVelocitiesIntegrator<Versor<float>, float>*
      pAngularVelocityIntegrator;

  /// Gyro data reader status.
  int status;

  /// Pointer to buffer of gyro data samples.
  GyroDataSample* pGyroSamples;

  /// Pointer to the gyro data correction class.
  GyroDataCorrection<float>* pGyroCorrector;

  /// Gyro data header block.
  MovementDataHeader* pGyroDataHeader;

  /// Total number of gyro samples.
  unsigned int total_samples;

  /// Last sample position.
  unsigned int last_sample;

  /// Time stamp of the image capture start [ns].
  Time capture_timestamp;

  /// Lower index of capture timestamp range.
  int captureTimestampIndex;

  /// Angular velocity x [rad/s]
  float* wx;

  /// Angular velocity y [rad/s]
  float* wy;

  /// Angular velocity z [rad/s]
  float* wz;

  /// Timestamp [ns]
  Time* timestamp;

  /// Initializes internal variables.
  void init(int numberGyroSamples);

  /// Deinitializes internal variables.
  void deinit();

  /// Procedure to correct the entered gyro data.
  void calcAngularVelocity(GyroDataCorrection<float>& gyroCorrector,
                           GyroDataCorrection<float>::GyroSample_t gyroSample);

  /// Find lower array index of capture timestamp.
  void calcTimeStampIndex();

  /// Transfers the raw gyro data to the output container.
  void copyRawGyroDataToOutput();

  /// Deinitialize the gyro data parameters.
  void destroyGyroDataContainer();
};

}  // namespace Deblurring
}  // namespace Test

#endif /* GYRODATAREADER_H_ */
