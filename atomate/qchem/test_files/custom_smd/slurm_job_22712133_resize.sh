export SLURM_NODELIST="nid00[905,909,981,999]"
export SLURM_JOB_NODELIST="nid00[905,909,981,999]"
export SLURM_NNODES=4
export SLURM_JOB_NUM_NODES=4
export SLURM_JOB_CPUS_PER_NODE="64(x4)"
unset SLURM_NPROCS
unset SLURM_NTASKS
unset SLURM_TASKS_PER_NODE
