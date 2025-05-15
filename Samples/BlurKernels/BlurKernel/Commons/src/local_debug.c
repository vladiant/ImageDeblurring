#include "local_debug.h"

#if defined(_MSC_VER) || defined(_WIN32)
#include <time.h>
#else
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#endif

#include <time.h>

const int64_t DELTA_EPOCH_IN_MICROSECS = 11644473600000000;

//=======================================================================//
#if defined(_POSIX_VERSION)

hdr_clock_t u_clock(void) {
#ifdef ANDROID

  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (hdr_clock_t)(t.tv_sec * 1000) + (hdr_clock_t)(t.tv_nsec / 1000000);

#else

  struct tms tm;
  times(&tm);
  long per_sec = sysconf(_SC_CLK_TCK);
  return HDR_CLOCKS_PER_SEC * tm.tms_utime / per_sec;

#endif
}

#elif defined(_WIN32)

hdr_clock_t u_clock(void) {
  // #error "WIN32 version of getUserTime not tested yet, please uncomment line
  // and test."

  FILETIME creationTime, exitTime, kernelTime, userTime;
  ::GetProcessTimes(::GetCurrentProcess(),  //_In_   HANDLE hProcess,
                    &creationTime,          //_Out_  LPFILETIME lpCreationTime,
                    &exitTime,              //_Out_  LPFILETIME lpExitTime,
                    &kernelTime,            //_Out_  LPFILETIME lpKernelTime,
                    &userTime               //_Out_  LPFILETIME lpUserTime
  );

  ULARGE_INTEGER userTime64;
  userTime64.u.LowPart = userTime.dwLowDateTime;
  userTime64.u.HighPart = userTime.dwHighDateTime;

  // convert 0.1us units to HDR_CLOCKS_PER_SEC units
  long rtn = double(userTime64.QuadPart) * HDR_CLOCKS_PER_SEC / 10000000.0;
  return rtn;
}
#else

#error "neither WIN32 nor _POSIX_VERSION defined"

#endif

//=====================================================================//
long hdr_getuTime(void) {
#ifdef ANDROID
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (long)(t.tv_sec * 1000) + (long)(t.tv_nsec / 1000000);
#else
  struct timeval now;
  gettimeofday(&now, nullptr);
  return (long)(now.tv_sec * 1000000 + now.tv_usec);
#endif
}

long hdr_getTime(void) {
#ifdef ANDROID
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (long)(t.tv_sec * 1000) + (long)(t.tv_nsec / 1000000);
#else
  struct timeval now;
  gettimeofday(&now, nullptr);
  return (long)(now.tv_sec * 1000 + now.tv_usec / 1000);
#endif
}

#if defined(_MSC_VER) || defined(_WIN32)
int gettimeofday(struct timeval *tv, void *tz) {
  FILETIME ft;
  int64 tmpres = 0;
  TIME_ZONE_INFORMATION tz_winapi;
  int rez = 0;

  ZeroMemory(&ft, sizeof(ft));
  ZeroMemory(&tz_winapi, sizeof(tz_winapi));

  GetSystemTimeAsFileTime(&ft);

  tmpres = ft.dwHighDateTime;
  tmpres <<= 32;
  tmpres |= ft.dwLowDateTime;

  /*converting file time to unix epoch*/
  tmpres /= 10; /*convert into microseconds*/
  tmpres -= DELTA_EPOCH_IN_MICROSECS;
  tv->tv_sec = (int32)(tmpres * 0.000001);
  tv->tv_usec = (tmpres % 1000000);

  return 0;
}

#endif
