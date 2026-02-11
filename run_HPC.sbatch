#!/bin/bash
#SBATCH --job-name="bench"
#SBATCH --time=04:00:00
#SBATCH --partition=GPU_queue
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=20GB

if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
  raw_line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scan_params.txt | sed 's/#.*//')
  trimmed=$(echo "$raw_line" | awk '{$1=$1;print}')
  if [[ -n "$trimmed" ]]; then
    label=$(echo "$trimmed" | tr -s '[:space:]' '_' | tr -cd '[:alnum:]_' | sed 's/_*$//')
    directoryName=${SLURM_ARRAY_JOB_ID}/${label}
  else
    directoryName=${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
  fi
else
  directoryName=${SLURM_JOB_ID}
fi
mkdir -p ${directoryName}


module purge
module load OpenMPI/4.0.3-GCC-9.3.0 GCC/9.3.0 CUDA/12.4.0 HDF5/1.10.6-gompi-2020a

mkdir -p ${directoryName}/logs ${directoryName}/mesh ${directoryName}/anchor ${directoryName}/parameter
#bash clean_all.sh

# --- 1. PRE-EXECUTION ---
echo "Job started at: $(date)"

# Start GPU monitor in background
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l 5 > ${directoryName}/gpu_usage_$SLURM_JOB_ID.csv &
GPU_MON_PID=$!


python3 init_parameters.py --simnum ${directoryName} --scan-file scan_params.txt --scan-index ${SLURM_ARRAY_TASK_ID:-0}

time {
  mpirun -np 2 python3 multibubble.py --equil --simnum ${directoryName}
  mpirun -np 2 python3 multibubble.py --restart --simnum ${directoryName}
}

# --- 3. POST-EXECUTION ---
kill $GPU_MON_PID
echo "Job finished at: $(date)"

# Instead of seff (which is broken), we print the stats using sacct directly
echo "--- RESOURCE USAGE SUMMARY ---"
sacct -j $SLURM_JOB_ID -X --format=JobName,Elapsed,MaxRSS,MaxDiskWrite,State
