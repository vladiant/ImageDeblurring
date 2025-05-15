#pragma once

#ifdef ANDROID
#include <android/log.h>
#include <jni.h>
#else
#endif
#ifdef _WIN32
#include <Windows.h>  //needed for timeval
#endif

#include <stdint.h>
#include <stdio.h>
#include <time.h>

typedef long hdr_clock_t;

#define LOG_TAG "Deblurring"
// #define ANDROID_LOG_FILE "/sdcard/DCIM/Frames/rHDRlog.txt"

#ifdef ANDROID
#ifdef ANDROID_LOG_FILE

#define LOGI(...)                                                \
  {                                                              \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); \
    FILE* outp = fopen(ANDROID_LOG_FILE, "a");                   \
    fprintf(outp, "%s I ", LOG_TAG);                             \
    fprintf(outp, __VA_ARGS__);                                  \
    fclose(outp);                                                \
  }
#define LOGE(...)                                                 \
  {                                                               \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__); \
    FILE* outp = fopen(ANDROID_LOG_FILE, "a");                    \
    fprintf(outp, "%s E ", LOG_TAG);                              \
    fprintf(outp, __VA_ARGS__);                                   \
    fclose(outp);                                                 \
  }

#else

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#endif

#else
//	#define  LOGI(...)  {FILE* outp=fopen("test.txt", "a"); fprintf(outp,
// LOG_TAG); fprintf(outp,__VA_ARGS__); fclose(outp);}
#define LOGI(...)                    \
  {                                  \
    fprintf(stdout, "%s ", LOG_TAG); \
    fprintf(stdout, __VA_ARGS__);    \
  }
#define LOGE(...)                    \
  {                                  \
    fprintf(stdout, "%s ", LOG_TAG); \
    fprintf(stderr, __VA_ARGS__);    \
  }
#endif

//============= Debug control =============//

#ifdef ANDROID

#define INITIAL_RGB_FILENAMENAME "/data/yuvBuffer/dbg_initial_rgb_"

#else

#define INITIAL_RGB_FILENAMENAME "dbg_initial_rgb_"

#endif

#define IMAGE_FILENAME_SUFFIX ".bmp"

//=========================================//

#ifdef __cplusplus
extern "C" {
#endif

hdr_clock_t u_clock(void);
#define HDR_CLOCKS_PER_SEC 1000.0

#ifdef __cplusplus
}
#endif

//=========================================//
// appropriate Windows gettimeofday()
#if defined(_MSC_VER) || defined(_WIN32)
int gettimeofday(struct timeval* tv /*in*/, void* tz /*in*/);
#endif

#define getTime() hdr_getTime()
#define getuTime() hdr_getuTime()

#ifdef __cplusplus
extern "C" {
#endif

long hdr_getTime(void);
long hdr_getuTime(void);

#ifdef __cplusplus
}
#endif
