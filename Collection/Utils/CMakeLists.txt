add_library(
    Utils
    CircularBuffer.h
    Common.h
    DiscreteAngularVelocitiesIntegrator.h
    GyroDataCorrection.h
    Matrix.h
    Quaternion.h
    Vector.h
    CoreCommon.h
    DiscreteOrientationPath.h
    MathCommon.h
    OrientationPath.h
    Time.h
    Versor.h
)

set_target_properties(
    Utils
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        LINKER_LANGUAGE CXX
        SOVERSION 1
        VERSION 1.0.0
)

target_compile_options(
    Utils
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

target_include_directories(
    Utils
    PUBLIC
        .
)





