cmake_minimum_required(VERSION 3.10)

project(SW4 C CXX Fortran)
enable_language(Fortran)

set(SW4_PRECISION "double" CACHE PATH "")
set(SW4_COMPUTE_LANG "cxx" CACHE PATH "")

# ascent olcf
# >    ml cmake cuda gcc fftw netlib-lapack hdf5

if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++11" CACHE STRING "")
endif()

include(blt/SetupBLT.cmake)

include_directory (src)
