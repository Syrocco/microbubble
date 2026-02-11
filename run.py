import os

with open("scan_params.txt", "r", encoding="ascii") as f:
	line_count = sum(1 for ln in f if ln.strip() and not ln.strip().startswith("#"))

if line_count > 0:
	os.system(f"sbatch --array=1-{line_count} run_HPC.sbatch")
else:
	os.system("sbatch run_HPC.sbatch")