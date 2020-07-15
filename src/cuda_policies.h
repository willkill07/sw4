#ifndef __CUDA_POLICIES_H__
#define __CUDA_POLICIES_H__
#include "RAJA/RAJA.hpp"
#include "RAJA/policy/cuda/policy.hpp"

#define SW4_FORCEINLINE __forceinline__
#define SYNC_DEVICE SW4_CheckDeviceError(cudaDeviceSynchronize())
#define SYNC_STREAM SW4_CheckDeviceError(cudaStreamSynchronize(0))
#define SW4_PEEK SW4_CheckDeviceError(cudaPeekAtLastError());
//   SW4_CheckDeviceError(cudaStreamSynchronize(0));
typedef RAJA::cuda_exec<1024> DEFAULT_LOOP1;
typedef RAJA::cuda_exec<1024, true> DEFAULT_LOOP1_ASYNC;
using REDUCTION_POLICY = RAJA::cuda_reduce;

typedef RAJA::cuda_exec<1024> PREDFORT_LOOP_POL;
typedef RAJA::cuda_exec<512, true> PREDFORT_LOOP_POL_ASYNC;

typedef RAJA::cuda_exec<1024> CORRFORT_LOOP_POL;
typedef RAJA::cuda_exec<512, true> CORRFORT_LOOP_POL_ASYNC;

typedef RAJA::cuda_exec<1024> DPDMTFORT_LOOP_POL;

typedef RAJA::cuda_exec<256, true> DPDMTFORT_LOOP_POL_ASYNC;

typedef RAJA::cuda_exec<1024> SARRAY_LOOP_POL1;

template <int BlockX, int BlockY, int BlockZ, int ThreadsPerBlock = BlockX * BlockY * BlockZ>
using KernelPolicyBlockLoop =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernelFixed<ThreadsPerBlock, // product
        RAJA::statement::Tile<0, RAJA::tile_fixed<BlockX>, RAJA::cuda_block_x_loop, // blockDim.x
          RAJA::statement::Tile<1, RAJA::tile_fixed<BlockY>, RAJA::cuda_block_y_loop, // blockDim.y
            RAJA::statement::Tile<2, RAJA::tile_fixed<BlockZ>, RAJA::cuda_block_z_loop, // blockDim.y
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                  RAJA::statement::For<2, RAJA::cuda_thread_z_direct,
                    RAJA::statement::Lambda<0>>>>>>>>>;

template <int BlockX, int BlockY, int BlockZ, int ThreadsPerBlock = BlockX * BlockY * BlockZ>
using KernelPolicyThreadLoop =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernelFixed<ThreadsPerBlock, // product
        RAJA::statement::Tile<0, RAJA::tile_fixed<BlockX>, RAJA::cuda_block_x_direct,
          RAJA::statement::Tile<1, RAJA::tile_fixed<BlockY>, RAJA::cuda_block_y_direct,
            RAJA::statement::Tile<2, RAJA::tile_fixed<BlockZ>, RAJA::cuda_block_z_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
                RAJA::statement::For<1, RAJA::cuda_thread_y_loop,
                  RAJA::statement::For<2, RAJA::cuda_thread_z_loop,
                    RAJA::statement::Lambda<0>>>>>>>>>;

// NERSC HACKATHON

// can also try KernelPolicyThreadLoop

// was 16, 4, 4
using XRHS_POL_ASYNC = KernelPolicyBlockLoop<16, 4, 4>;
using CURV_POL_LOOP_N1 = KernelPolicyBlockLoop<16, 4, 6>;

using CURV_POL_LOOP_0 = KernelPolicyBlockLoop<16, 4, 4>;
using CURV_POL_LOOP_1 = KernelPolicyBlockLoop<16, 4, 4>;
using CURV_POL_LOOP_2 = KernelPolicyBlockLoop<16, 4, 4>;

using DEFAULT_LOOP2X =
  RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      RAJA::statement::For<1, RAJA::cuda_block_x_loop,
        RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
          RAJA::statement::Lambda<0>>>>>;

using DEFAULT_LOOP2X_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernelAsync<
        RAJA::statement::For<1, RAJA::cuda_block_x_loop,
          RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
            RAJA::statement::Lambda<0>>>>>;

using DEFAULT_LOOP3 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernelFixed<64, // product
        RAJA::statement::Tile<0, RAJA::tile_fixed<4>, RAJA::cuda_block_x_loop, // blockDim.x
          RAJA::statement::Tile<1, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop, // blockDim.y
            RAJA::statement::Tile<2, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop, // blockDim.y
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                  RAJA::statement::For<2, RAJA::cuda_thread_z_direct,
                    RAJA::statement::Lambda<0>>>>>>>>>;

using SARRAY_LOOP_POL2 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernelFixed<512,
        RAJA::statement::Tile<2, RAJA::tile_fixed<1024>, RAJA::cuda_block_z_loop,
          RAJA::statement::For<0, RAJA::cuda_thread_y_direct,
            RAJA::statement::For<1, RAJA::cuda_thread_x_direct,
              RAJA::statement::For<2, RAJA::cuda_thread_z_direct,
                RAJA::statement::For<3, RAJA::seq_exec,
                  RAJA::statement::Lambda<0>>>>>>>>;

// using RHS4_EXEC_POL =
//   RAJA::KernelPolicy<
//   RAJA::statement::CudaKernel<
//     RAJA::statement::For<0, RAJA::cuda_threadblock_exec<4>,
// 			 RAJA::statement::For<1, RAJA::cuda_threadblock_exec<4>,
// 					      RAJA::statement::For<2,
// RAJA::cuda_threadblock_exec<16 >,
// RAJA::statement::Lambda<0> >>>>>;

using RHS4_EXEC_POL =
  RAJA::KernelPolicy<
    RAJA::statement::CudaKernelFixed<384,
      RAJA::statement::Tile<0, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<1, RAJA::tile_fixed<4>, RAJA::cuda_block_x_loop,
          RAJA::statement::Tile<2, RAJA::tile_fixed<16>, RAJA::cuda_block_z_loop,
            RAJA::statement::For<0, RAJA::cuda_thread_y_direct,
              RAJA::statement::For<1, RAJA::cuda_thread_x_direct,
                RAJA::statement::For<2, RAJA::cuda_thread_z_direct,
                  RAJA::statement::Lambda<0>>>>>>>>>;

using ICSTRESS_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::Tile<0, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
          RAJA::statement::Tile<1, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
            RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
              RAJA::statement::For<1, RAJA::cuda_thread_y_loop,
                RAJA::statement::Lambda<0>>>>>>>;

using ICSTRESS_EXEC_POL_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernelFixedAsync<256,
        RAJA::statement::Tile<0, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
          RAJA::statement::Tile<1, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
            RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                RAJA::statement::Lambda<0>>>>>>>;

// using RHS4_EXEC_POL_ASYNC =
//   RAJA::KernelPolicy<
//   RAJA::statement::CudaKernelAsync<
//     RAJA::statement::For<0, RAJA::cuda_threadblock_exec<4>,
// 			 RAJA::statement::For<1, RAJA::cuda_threadblock_exec<4>,
// 					      RAJA::statement::For<2,
// RAJA::cuda_threadblock_exec<16 >,
// RAJA::statement::Lambda<0> >>>>>;

using RHS4_EXEC_POL_ASYNC_OLDE =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernelAsync<
        RAJA::statement::Tile<0, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
          RAJA::statement::Tile<1, RAJA::tile_fixed<4>, RAJA::cuda_block_x_loop,
            RAJA::statement::Tile<2, RAJA::tile_fixed<16>, RAJA::cuda_block_z_loop,
              RAJA::statement::For<0, RAJA::cuda_thread_y_loop,
                RAJA::statement::For<1, RAJA::cuda_thread_x_loop,
                  RAJA::statement::For<2, RAJA::cuda_thread_z_loop,
                    RAJA::statement::Lambda<0>>>>>>>>>;

using RHS4_EXEC_POL_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernelFixedAsync<256,
        RAJA::statement::Tile<0, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop,
          RAJA::statement::Tile<1, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
            RAJA::statement::Tile<2, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
              RAJA::statement::For<0, RAJA::cuda_thread_z_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                  RAJA::statement::For<2, RAJA::cuda_thread_x_direct,
                    RAJA::statement::Lambda<0>>>>>>>>>;

using CONSINTP_EXEC_POL1 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::Tile<0, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
          RAJA::statement::Tile<1, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
            RAJA::statement::For<0, RAJA::cuda_thread_y_direct,
              RAJA::statement::For<1, RAJA::cuda_thread_x_direct,
                RAJA::statement::Lambda<0>>>>>>>;

using ODDIODDJ_EXEC_POL1_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelFixedAsync<
        256,
        RAJA::statement::Tile<
            0, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
            RAJA::statement::Tile<
                1, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
                RAJA::statement::For<
                    0, RAJA::cuda_thread_y_direct,
                    RAJA::statement::For<1, RAJA::cuda_thread_x_direct,
                                         RAJA::statement::Lambda<0>>>>>>>;

using ODDIODDJ_EXEC_POL2_ASYNC = RHS4_EXEC_POL_ASYNC;

using EVENIODDJ_EXEC_POL_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<RAJA::statement::For<
        1, RAJA::cuda_block_x_loop,
        RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
                             RAJA::statement::Lambda<0>>>>>;

using EVENIEVENJ_EXEC_POL_ASYNC = EVENIODDJ_EXEC_POL_ASYNC;

using ODDIEVENJ_EXEC_POL1_ASYNC = ICSTRESS_EXEC_POL_ASYNC;

using ODDIEVENJ_EXEC_POL2_ASYNC = RHS4_EXEC_POL_ASYNC;

using XRHS_POL =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::For<
        0, RAJA::cuda_block_x_loop,
        RAJA::statement::For<
            1, RAJA::cuda_block_y_loop,
            RAJA::statement::For<2, RAJA::cuda_thread_x_loop,
                                 RAJA::statement::Lambda<0>>>>>>;

using TWILIGHTSG_POL =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::Tile<
        0, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
            RAJA::statement::Tile<
                2, RAJA::tile_fixed<64>, RAJA::cuda_block_x_loop,
                RAJA::statement::For<
                    0, RAJA::cuda_thread_z_direct,
                    RAJA::statement::For<
                        1, RAJA::cuda_thread_y_direct,
                        RAJA::statement::For<2, RAJA::cuda_thread_x_direct,
                                             RAJA::statement::Lambda<0>>>>>>>>>;

using CONSINTP_EXEC_POL4 = ICSTRESS_EXEC_POL;

using CONSINTP_EXEC_POL5 = ICSTRESS_EXEC_POL;

using PRELIM_CORR_EXEC_POL1 = DEFAULT_LOOP2X;
using PRELIM_CORR_EXEC_POL1_ASYNC = DEFAULT_LOOP2X_ASYNC;

using PRELIM_PRED_EXEC_POL1 = ICSTRESS_EXEC_POL;
using PRELIM_PRED_EXEC_POL1_ASYNC = ICSTRESS_EXEC_POL_ASYNC;

// using ENFORCEBC_CORR_EXEC_POL1 =  ICSTRESS_EXEC_POL;

using ENFORCEBC_CORR_EXEC_POL1 =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelFixed<
        256,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
            RAJA::statement::Tile<
                0, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
                RAJA::statement::For<
                    1, RAJA::cuda_thread_x_direct,
                    RAJA::statement::For<0, RAJA::cuda_thread_y_direct,
                                         RAJA::statement::Lambda<0>>>>>>>;

using BCFORT_EXEC_POL1 = RHS4_EXEC_POL;
using BCFORT_EXEC_POL2 = ICSTRESS_EXEC_POL;

using ENERGY4CI_EXEC_POL = RHS4_EXEC_POL;

// Next 4 in solve.C
using DHI_POL_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<RAJA::statement::For<
        0, RAJA::cuda_block_x_loop,
        RAJA::statement::For<
            1, RAJA::cuda_block_y_loop,
            RAJA::statement::For<2, RAJA::cuda_thread_x_loop,
                                 RAJA::statement::Lambda<0>>>>>>;

using GIG_POL_ASYNC = RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<
    RAJA::statement::For<0, RAJA::cuda_block_x_loop,
                         RAJA::statement::For<1, RAJA::cuda_thread_x_loop,
                                              RAJA::statement::Lambda<0>>>>>;

using AVS_POL_ASYNC = RAJA::KernelPolicy<RAJA::statement::CudaKernelFixedAsync<
    256, RAJA::statement::Tile<
             0, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
             RAJA::statement::Tile<
                 1, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
                 RAJA::statement::For<
                     0, RAJA::cuda_thread_y_direct,
                     RAJA::statement::For<1, RAJA::cuda_thread_x_direct,
                                          RAJA::statement::Lambda<0>>>>>>>;

using EBFA_POL =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::Tile<
        0, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
            RAJA::statement::For<
                0, RAJA::cuda_thread_x_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                                     RAJA::statement::Lambda<0>>>>>>>;

// void EW::get_exact_point_source in EW.C
using GEPS_POL = RAJA::KernelPolicy<
    // RAJA::statement::CudaKernelExt<RAJA::cuda_explicit_launch<false, 0,
    // 256>,
    RAJA::statement::CudaKernelFixed<
        256,
        // RAJA::statement::CudaKernel<
        // RAJA::statement::CudaKernelOcc<
        RAJA::statement::Tile<
            0, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
            RAJA::statement::Tile<
                1, RAJA::tile_fixed<4>, RAJA::cuda_block_x_loop,
                RAJA::statement::Tile<
                    2, RAJA::tile_fixed<16>, RAJA::cuda_block_z_loop,
                    RAJA::statement::For<
                        0, RAJA::cuda_thread_y_direct,
                        RAJA::statement::For<
                            1, RAJA::cuda_thread_x_direct,
                            RAJA::statement::For<
                                2, RAJA::cuda_thread_z_direct,
                                RAJA::statement::Lambda<0>>>>>>>>>;

// CurvilinearInterface2::bnd_zero
using BZ_POL_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<RAJA::statement::Tile<
        0, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
            RAJA::statement::Tile<
                2, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
                RAJA::statement::For<
                    0, RAJA::cuda_thread_z_direct,
                    RAJA::statement::For<
                        1, RAJA::cuda_thread_y_direct,
                        RAJA::statement::For<2, RAJA::cuda_thread_x_direct,
                                             RAJA::statement::Lambda<0>>>>>>>>>;

// CurvilinearInterface2::injection
using INJ_POL_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<RAJA::statement::Tile<
        0, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
            RAJA::statement::Tile<
                2, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
                RAJA::statement::For<
                    0, RAJA::cuda_thread_z_direct,
                    RAJA::statement::For<
                        1, RAJA::cuda_thread_y_direct,
                        RAJA::statement::For<2, RAJA::cuda_thread_x_direct,
                                             RAJA::statement::Lambda<0>>>>>>>>>;

using INJ_POL2_ASYNC = RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<
    RAJA::statement::For<1, RAJA::cuda_block_x_loop,
                         RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
                                              RAJA::statement::Lambda<0>>>>>;

// CurvilinearInterface2::communicate_array

using CA_POL =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::Tile<
        0, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
            RAJA::statement::Tile<
                2, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
                RAJA::statement::For<
                    0, RAJA::cuda_thread_z_direct,
                    RAJA::statement::For<
                        1, RAJA::cuda_thread_y_direct,
                        RAJA::statement::For<2, RAJA::cuda_thread_x_direct,
                                             RAJA::statement::Lambda<0>>>>>>>>>;

// Sarray::assign

using SAA_POL =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::Tile<
        1, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<
            3, RAJA::tile_fixed<64>, RAJA::cuda_block_x_loop,
            RAJA::statement::Tile<
                2, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop,
                RAJA::statement::For<
                    1, RAJA::cuda_thread_y_direct,
                    RAJA::statement::For<
                        3, RAJA::cuda_thread_x_direct,
                        RAJA::statement::For<
                            2, RAJA::cuda_thread_z_direct,
                            RAJA::statement::For<
                                0, RAJA::seq_exec,
                                RAJA::statement::Lambda<0>>>>>>>>>>;

// Sarray::insert_intersection(
using SII_POL =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::Tile<
        0, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
            RAJA::statement::Tile<
                2, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
                RAJA::statement::For<
                    0, RAJA::cuda_thread_z_direct,
                    RAJA::statement::For<
                        1, RAJA::cuda_thread_y_direct,
                        RAJA::statement::For<2, RAJA::cuda_thread_x_direct,
                                             RAJA::statement::Lambda<0>>>>>>>>>;

// TestEcons::get_ubnd(
using TGU_POL_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<RAJA::statement::Tile<
        0, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
            RAJA::statement::Tile<
                2, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
                RAJA::statement::For<
                    0, RAJA::cuda_thread_z_direct,
                    RAJA::statement::For<
                        1, RAJA::cuda_thread_y_direct,
                        RAJA::statement::For<2, RAJA::cuda_thread_x_direct,
                                             RAJA::statement::Lambda<0>>>>>>>>>;

// in addmemvarforcing2.C
using AMVPCa_POL =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::Tile<
        0, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<4>, RAJA::cuda_block_x_loop,
            RAJA::statement::Tile<
                2, RAJA::tile_fixed<64>, RAJA::cuda_block_z_loop,
                RAJA::statement::For<
                    0, RAJA::cuda_thread_y_loop,
                    RAJA::statement::For<
                        1, RAJA::cuda_thread_x_loop,
                        RAJA::statement::For<2, RAJA::cuda_thread_z_loop,
                                             RAJA::statement::Lambda<0>>>>>>>>>;

using AMVPCu_POL =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::Tile<
        0, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<4>, RAJA::cuda_block_x_loop,
            RAJA::statement::Tile<
                2, RAJA::tile_fixed<64>, RAJA::cuda_block_z_loop,
                RAJA::statement::For<
                    0, RAJA::cuda_thread_y_loop,
                    RAJA::statement::For<
                        1, RAJA::cuda_thread_x_loop,
                        RAJA::statement::For<2, RAJA::cuda_thread_z_loop,
                                             RAJA::statement::Lambda<0>>>>>>>>>;

using AMVC2Ca_POL_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::Tile<
        0, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<4>, RAJA::cuda_block_x_loop,
            RAJA::statement::Tile<
                2, RAJA::tile_fixed<64>, RAJA::cuda_block_z_loop,
                RAJA::statement::For<
                    0, RAJA::cuda_thread_y_loop,
                    RAJA::statement::For<
                        1, RAJA::cuda_thread_x_loop,
                        RAJA::statement::For<2, RAJA::cuda_thread_z_loop,
                                             RAJA::statement::Lambda<0>>>>>>>>>;

using AMVC2Cu_POL =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::Tile<
        0, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<4>, RAJA::cuda_block_x_loop,
            RAJA::statement::Tile<
                2, RAJA::tile_fixed<64>, RAJA::cuda_block_z_loop,
                RAJA::statement::For<
                    0, RAJA::cuda_thread_y_loop,
                    RAJA::statement::For<
                        1, RAJA::cuda_thread_x_loop,
                        RAJA::statement::For<2, RAJA::cuda_thread_z_loop,
                                             RAJA::statement::Lambda<0>>>>>>>>>;

// In addsg4windc.C
using ASG4WC_POL_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<RAJA::statement::For<
        2, RAJA::cuda_block_z_loop,
        RAJA::statement::For<
            1, RAJA::cuda_block_y_loop,
            RAJA::statement::For<
                0, RAJA::cuda_thread_x_direct,
                RAJA::statement::For<3, RAJA::seq_exec,
                                     RAJA::statement::Lambda<0>>>>>>>;

// In addsgdc.C
using ADDSGD_POL_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelFixedAsync<256,
        RAJA::statement::Tile<1, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
            RAJA::statement::Tile<3, RAJA::tile_fixed<64>, RAJA::cuda_block_x_loop,
                RAJA::statement::Tile<2, RAJA::tile_fixed<1>, RAJA::cuda_block_z_loop,
                    RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                        RAJA::statement::For<3, RAJA::cuda_thread_x_direct,
                            RAJA::statement::For<2, RAJA::cuda_thread_z_direct,
                                RAJA::statement::For<0, RAJA::seq_exec,
                                    RAJA::statement::Lambda<0>>>>>>>>>>;


constexpr int SGD_BLOCK_X = 4;
constexpr int SGD_BLOCK_Y = 4;
constexpr int SGD_BLOCK_Z = 4;
constexpr int SGD_THREADS_PER_BLOCK = SGD_BLOCK_X * SGD_BLOCK_Y * SGD_BLOCK_Z; 
constexpr int SGD_GHOST_CELLS = 2;

using ADDSGD_TILE =
  RAJA::LocalArray<
    float_sw4, 
    RAJA::Perm<3, 2, 1, 0>,
    RAJA::SizeList<
      SGD_BLOCK_X + 2 * SGD_GHOST_CELLS,
      SGD_BLOCK_Y + 2 * SGD_GHOST_CELLS,
      SGD_BLOCK_Z + 2 * SGD_GHOST_CELLS,
      4
    >
  >;

using ADDSGD_POL_SHMEM =
  RAJA::KernelPolicy<
    RAJA::statement::CudaKernelFixed<SGD_THREADS_PER_BLOCK,

      /* 3-D tile of block dims */
      RAJA::statement::Tile<0, RAJA::tile_fixed<SGD_BLOCK_X>, RAJA::cuda_block_x_loop,
        RAJA::statement::Tile<1, RAJA::tile_fixed<SGD_BLOCK_Y>, RAJA::cuda_block_y_loop,
          RAJA::statement::Tile<2, RAJA::tile_fixed<SGD_BLOCK_Z>, RAJA::cuda_block_z_loop,

            /* **** initialize shared memory **** */
            RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<3>>,
            RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<4>>,

            /* **** global memory -> shared memory **** */
            /* ForICount used to acquire local tile index */
            RAJA::statement::ForICount<0, RAJA::statement::Param<0>, RAJA::cuda_thread_x_direct,
              RAJA::statement::ForICount<1, RAJA::statement::Param<1>, RAJA::cuda_thread_y_direct,
                RAJA::statement::ForICount<2, RAJA::statement::Param<2>, RAJA::cuda_thread_z_direct,
                  RAJA::statement::For<3, RAJA::seq_exec,
                    RAJA::statement::Lambda<0>
                  >,
                >,
              >,
            >,

            /* sync threads so shared memory is initialized */
            RAJA::statement::CudaSyncThreads,

            /* **** compute kernel **** */
            /* ForICount used to acquire local tile index */
            RAJA::statement::ForICount<0, RAJA::statement::Param<0>, RAJA::cuda_thread_x_direct,
              RAJA::statement::ForICount<1, RAJA::statement::Param<1>, RAJA::cuda_thread_y_direct,
                RAJA::statement::ForICount<2, RAJA::statement::Param<2>, RAJA::cuda_thread_z_direct,
                  RAJA::statement::For<3, RAJA::seq_exec,
                    RAJA::statement::Lambda<1>
                  >,
                >,
              >,
            >,

            /* sync threads */
            RAJA::statement::CudaSyncThreads,
          >,
        >,
      >,
    >,
  >;

using ADDSGD_POL2_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelFixedAsync<
        256,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
            RAJA::statement::Tile<
                3, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
                RAJA::statement::Tile<
                    2, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop,
                    RAJA::statement::For<
                        1, RAJA::cuda_thread_y_direct,
                        RAJA::statement::For<
                            3, RAJA::cuda_thread_x_direct,
                            RAJA::statement::For<
                                2, RAJA::cuda_thread_z_direct,
                                RAJA::statement::For<
                                    0, RAJA::seq_exec,
                                    RAJA::statement::Lambda<0>>>>>>>>>>;

// in bcforce.C
using BCFORT_EXEC_POL2_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<RAJA::statement::Tile<
        0, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<4>, RAJA::cuda_block_x_loop,
            RAJA::statement::Tile<
                2, RAJA::tile_fixed<64>, RAJA::cuda_block_z_loop,
                RAJA::statement::For<
                    0, RAJA::cuda_thread_y_direct,
                    RAJA::statement::For<
                        1, RAJA::cuda_thread_x_direct,
                        RAJA::statement::For<2, RAJA::cuda_thread_z_direct,
                                             RAJA::statement::Lambda<0>>>>>>>>>;

using BCFORT_EXEC_POL3_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<RAJA::statement::Tile<
        0, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
            RAJA::statement::For<
                0, RAJA::cuda_thread_x_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                                     RAJA::statement::Lambda<0>>>>>>>;

// in curvilinear4sgc.C
using CURV_POL_ORG =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::For<
        0, RAJA::cuda_block_x_loop,
        RAJA::statement::For<
            1, RAJA::cuda_block_y_loop,
            RAJA::statement::For<2, RAJA::cuda_thread_x_loop,
                                 RAJA::statement::Lambda<0>>>>>>;
using CURV_POL = DEFAULT_LOOP3;

// in parallelStuff.C
using BUFFER_POL =
  RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<
    RAJA::statement::For<1, RAJA::cuda_block_x_loop,
                         RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
                                              RAJA::statement::Lambda<0>>>>>;

// in rhs3cuvilinearsgc.C
using RHS4CU_POL_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<RAJA::statement::Tile<
        0, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
            RAJA::statement::For<
                0, RAJA::cuda_thread_x_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                                     RAJA::statement::Lambda<0>>>>>>>;

// in rhs4th3fortc.C
using XRHS_POL2 =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::For<
        0, RAJA::cuda_block_x_loop,
        RAJA::statement::For<
            1, RAJA::cuda_block_y_loop,
            RAJA::statement::For<2, RAJA::cuda_thread_x_loop,
                                 RAJA::statement::Lambda<0>>>>>>;
using RHS4TH3_POL_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelFixedAsync<
        256,
        RAJA::statement::Tile<
            0, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop,
            RAJA::statement::Tile<
                1, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
                RAJA::statement::Tile<
                    2, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
                    RAJA::statement::For<
                        0, RAJA::cuda_thread_z_direct,
                        RAJA::statement::For<
                            1, RAJA::cuda_thread_y_direct,
                            RAJA::statement::For<
                                2, RAJA::cuda_thread_x_direct,
                                RAJA::statement::Lambda<0>>>>>>>>>;

using RHS4TH3_POL2_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelFixedAsync<
        256,
        RAJA::statement::Tile<
            0, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop,
            RAJA::statement::Tile<
                1, RAJA::tile_fixed<4>, RAJA::cuda_block_y_loop,
                RAJA::statement::Tile<
                    2, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
                    RAJA::statement::For<
                        0, RAJA::cuda_thread_z_direct,
                        RAJA::statement::For<
                            1, RAJA::cuda_thread_y_direct,
                            RAJA::statement::For<
                                2, RAJA::cuda_thread_x_direct,
                                RAJA::statement::Lambda<0>>>>>>>>>;

/* using RHS4TH3_POL2_ASYNC = RAJA::statement::CudaKernelFixedAsync< */
/*   256, */
/*   RAJA::statement::Tile< */
/*   0, RAJA::tile_fixed<4>, RAJA::cuda_block_z_loop, */
/*   RAJA::statement::Tile< */
/*   1, RAJA::tile_fixed<4>, */
/*   RAJA::cuda_block_y_loop, */
/*   RAJA::statement::Tile< */
/*   2, RAJA::tile_fixed<16>, */
/*   RAJA::cuda_block_x_loop, */
/*   RAJA::statement::For< */
/*   0, RAJA::cuda_thread_z_direct, */
/*   RAJA::statement::For< */
/*   1, RAJA::cuda_thread_y_direct, */
/*   RAJA::statement::For< */
/*   2, RAJA::cuda_thread_x_direct, */
/*   RAJA::statement::Lambda<0>>>>>>>>>; */

using VBSC_POL = RAJA::KernelPolicy<RAJA::statement::CudaKernel<
    RAJA::statement::For<0, RAJA::cuda_block_x_loop,
                         RAJA::statement::For<1, RAJA::cuda_block_y_loop,
                                              RAJA::statement::Lambda<0>>>>>;

using AFCC_POL_ASYNC =
    RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<RAJA::statement::Tile<
        0, RAJA::tile_fixed<16>, RAJA::cuda_block_x_loop,
        RAJA::statement::Tile<
            1, RAJA::tile_fixed<16>, RAJA::cuda_block_y_loop,
            RAJA::statement::For<
                0, RAJA::cuda_thread_x_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                                     RAJA::statement::Lambda<0>>>>>>>;

// In updatememvarc.C
using MPFC_POL_ASYNC = RAJA::KernelPolicy<RAJA::statement::CudaKernelAsync<RAJA::statement::For<
          0, RAJA::cuda_block_x_loop,
          RAJA::statement::For<
              1, RAJA::cuda_block_y_loop,
              RAJA::statement::For<2, RAJA::cuda_thread_x_direct,
                                   RAJA::statement::Lambda<0>>>>>>;
#endif
