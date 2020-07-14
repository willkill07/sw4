#!/bin/bash
# Begin LSF Directives
#BSUB -P GEN138
#BSUB -W 0:30
#BSUB -nnodes 1
#BSUB -alloc_flags "gpumps"
#BSUB -J sw4-run
#BSUB -o sw4-run.%J
#BSUB -e sw4-run.%J

sw4binary=${HOME}/sw4/optimize/sw4
type=run
file=/gpfs/wolf/gen138/proj-shared/eqsim/killian-test/berkeley-s-h25.in

ml cmake cuda gcc fftw netlib-lapack hdf5 
ml nsight-compute nsight-systems

export CALI_SERVICES_ENABLE=event:recorder:timestamp:trace:nvprof

if [[ ${type} == "compute" ]]
then
    run_command='ncu -f -o profile'
elif [[ ${type} == "system" ]]
then
    run_command='nsys profile -t nvtx,cuda -s none'
else
    run_command=''
fi

jsrun -n1 -c1 -g1 -a1 -bpacked:1 ${run_command} ${sw4binary} ${file}
