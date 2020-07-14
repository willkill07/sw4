#!/bin/bash
# Begin LSF Directives
#BSUB -P GEN138
#BSUB -W 0:30
#BSUB -nnodes 1
#BSUB -alloc_flags "gpumps"
#BSUB -J sw4-compute
#BSUB -o sw4-compute.%J
#BSUB -e sw4-compute.%J

sw4binary=${HOME}/sw4/optimize/sw4
type=compute
file=/gpfs/wolf/gen138/proj-shared/eqsim/killian-test/berkeley-s-h25.in

ml cmake gcc fftw netlib-lapack hdf5 
ml cuda
ml nsight-compute nsight-systems

export CALI_SERVICES_ENABLE=event:recorder:timestamp:trace:nvprof

export KERNEL_NAME=CudaKernelLauncherFixed
# export KERNEL_NAME=forall3kernel

if [[ ${type} == "compute" ]]
then
    # -s <start id> -- 60 for beginning of evalRHS
    # -c <count> -- 5 for half timestep.
    run_command="ncu --set full -f -o profile -k ${KERNEL_NAME} -s 60 -c 5"
elif [[ ${type} == "system" ]]
then
    run_command="nsys profile -t nvtx,cuda -s none"
else
    run_command=''
fi

jsrun -n1 -c1 -g1 -a1 -bpacked:1 ${run_command} ${sw4binary} ${file}
