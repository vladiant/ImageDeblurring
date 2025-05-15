#pragma once

#if defined WIN32 || defined __CYGWIN__
#ifdef common_EXPORTS
#ifdef __GNUC__
#define EXP __attribute__((dllexport))
#define SXP __attribute__((dllexport))
#else
#define EXP                                                               \
  __declspec(dllexport) /* Note: actually gcc seems to also supports this \
                           syntax. */
#define SXP __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
#define EXP /* __attribute__ ((dllimport)) */
#define SXP
#else
#define EXP /* __declspec(dllimport) Note: actually gcc seems to also supports \
               this syntax. */
#define SXP
#endif
#endif
#else
#if __GNUC__ >= 4
#define EXP __attribute__((visibility("default")))
#define NEXP __attribute__((visibility("hidden")))
#define SXP
#else
#define EXP
#define NEXP
#define SXP
#endif
#endif
