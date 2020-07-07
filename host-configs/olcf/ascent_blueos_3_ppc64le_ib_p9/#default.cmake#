#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------

set(ENABLE_FORTRAN ON CACHE BOOL "")

set(CMAKE_C_COMPILER   "mpicc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "mpicxx" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "mpif90" CACHE PATH "")

set(BLT_CXX_STD "c++11" CACHE STRING "")

#------------------------------------------------------------------------------
# MPI Support
#------------------------------------------------------------------------------

set(ENABLE_MPI ON CACHE BOOL "")
set(MPI_C_COMPILER         "mpicc" CACHE PATH "")
set(MPI_CXX_COMPILER       "mpicxx" CACHE PATH "")
set(MPI_Fortran_COMPILER   "mpif90" CACHE PATH "")

set(MPIEXEC                "mpirun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG   "-np" CACHE PATH "")
set(BLT_MPI_COMMAND_APPEND "mpibind" CACHE PATH "")

#------------------------------------------------------------------------------
# Other machine specifics
#------------------------------------------------------------------------------

set(CMAKE_Fortran_COMPILER_ID "XL" CACHE PATH "All of BlueOS compilers report clang due to nvcc, override to proper compiler family")
set(BLT_FORTRAN_FLAGS "-WF,-C!" CACHE PATH "Converts C-style comments to Fortran style in preprocessed files")

#------------------------------------------------------------------------------
# CUDA support
#------------------------------------------------------------------------------

set(ENABLE_CUDA ON CACHE BOOL "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_HOST_COMPILER "${MPI_CXX_COMPILER}" CACHE PATH "")

set(_cuda_arch "sm_70")
set(CMAKE_CUDA_FLAGS "-restrict -arch ${_cuda_arch} -std=${BLT_CXX_STD} --expt-extended-lambda -G" CACHE STRING "")

set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )

# nvcc does not like gtest's 'pthreads' flag
set(gtest_disable_pthreads ON CACHE BOOL "")

