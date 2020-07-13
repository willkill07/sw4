cmake_minimum_required(VERSION 3.10)

project(SW4 C CXX Fortran)
enable_language(Fortran)

set(SW4_PRECISION "double" CACHE PATH "")

# ascent olcf
# >    ml cmake cuda gcc fftw netlib-lapack hdf5

if (NOT BLT_LOADED)
  if (DEFINED BLT_SOURCE_DIR)
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
  else ()
    set (BLT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/blt CACHE PATH "")

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR
          "The BLT git submodule is not present. "
          "Either run the following two commands in your git repository: \n"
          "    git submodule init\n"
          "    git submodule update\n"
          "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif ()
  endif ()

  include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()

if (NOT BLT_CXX_STD)
  set(BLT_CXX_STD "c++11" CACHE STRING "")
endif()

find_package(HDF5 REQUIRED COMPONENTS Fortran HL)
find_package(FFTW REQUIRED)

find_package(RAJA REQUIRED)
find_package(umpire REQUIRED)
find_package(Caliper REQUIRED)

blt_register_library(
  NAME                     proj
  INCLUDES                 ${PROJ_DIR}/include
  TREAT_INCLUDES_AS_SYSTEM On
  LINK_FLAGS               -L${PROJ_HOME}/lib64 -lproj -Wl,-rpath=$(PROJ_HOME)/lib64
)

blt_add_library(
  NAME
    sw4_meta
  DEFINES
    SW4_CROUTINES
    SW4_MASS_PREFETCH=1
    ENABLE_FFTW=1
    ENABLE_MPI_TIMING_BARRIER=1
    USE_DIRECT_INVERSE=1
    USE_HDF5=1 
    DISABLE_PREFETCH
  INCLUDES
    "${CMAKE_SRC_DIRECTORY}/src/${SW4_PRECISION}"
  DEPENDS_ON
    HDF5
    FFTW
)

blt_add_library(
  NAME sw4_meta_cxx
  DEFINES
    SW4_USE_UMPIRE=1
    RAJA_USE_RESTRICT_PTR
    RAJA_USE_CUDA
    ENABLE_CALIPER=1
    ENABLE_CUDA
    USE_NVTX
    CUDA_CODE
  DEPENDS_ON
    sw4_meta
    RAJA
    umpire
    Caliper
)

blt_add_library(
  NAME
    sw4_meta_cuda
  DEPENDS_ON
    sw4_meta_cxx
    cuda
)

include_directory (src)
