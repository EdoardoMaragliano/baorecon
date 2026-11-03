#!/bin/bash
# CPU benchmark sweep of bench_bao_reconstructor.py (multigrid-jacobi + ifft-scipy),
# same configuration as the 30 July 2026 run (benchmarks/run_bench30lug.sh):
# N=1e6, nmesh 128/256/512/1024, repeats=5, 8 threads, baorecon_cpu + pyrecon.
# Both sweeps run in ONE job so they share the node (same CPU model in the CSV
# provenance). g2farm10 (the July node) is DOWN in slurm, so the node is chosen by
# the scheduler inside the `long` partition.
#
# Submit (from a node where sbatch works, e.g. teogpu02):
#   BENCH_OUT=results_20261103 sbatch --export=ALL,BENCH_OUT benchmarks/slurm/bench_cpu.sbatch
#
#SBATCH --job-name=bench_cpu
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 8 worker threads (THREADS="8" in bench_bao_reconstructor.py); 16 logical CPUs so
# that with ThreadsPerCore=2 the 8 threads get 8 physical cores, as on g2farm10.
#SBATCH --cpus-per-task=16
# 1024^3 point: baorecon_cpu peak RSS 39.4 GB, pyrecon 25 GB (sequential workers).
#SBATCH --mem=64G
# July on g2farm10: jacobi 1h35 + ifft 1h30. Generous margin for a slower node.
#SBATCH --time=12:00:00
# CentOS 7 nodes (GLIBC 2.17, wheels in the BAOFit env do not import there) and the
# GPU node (a visible GPU would add the baorecon_gpu backend to the ifft sweep).
#SBATCH --exclude=aiace12,aiace14,aiace15,aiace16,aiace17,e4farm02,e4farm03,e4farm04,g2farm8,geant15,geant16,jlab01,jlab02,jlab03,jlab04,jlab05,jlab06,jlab07,jlab08,totem08,teogpu02
#SBATCH --output=/mnt/project_mnt/home_fs/emaragliano/Work/Projects/Dottorato/baorecon/benchmarks/slurm/%x_%j.out
#SBATCH --error=/mnt/project_mnt/home_fs/emaragliano/Work/Projects/Dottorato/baorecon/benchmarks/slurm/%x_%j.out
set -u

PY=/farmdisk1/emaragliano/miniconda3/envs/BAOFit/bin/python
BENCH_DIR=/mnt/project_mnt/home_fs/emaragliano/Work/Projects/Dottorato/baorecon/benchmarks
OUT=${BENCH_OUT:-results_$(date +%Y%m%d)}
LOGS="$BENCH_DIR/$OUT/logs"
REPEATS=5

cd "$BENCH_DIR" || exit 1
mkdir -p "$LOGS"
# Everything below goes to the dated results dir; the slurm %x_%j.out only keeps
# failures that happen before this line.
exec > "$LOGS/session_cpu_${SLURM_JOB_ID}.log" 2>&1

echo "=== job $SLURM_JOB_ID on $(hostname) | $(date '+%F %T') ==="
echo "cpus-per-task=$SLURM_CPUS_PER_TASK mem=$SLURM_MEM_PER_NODE MB"
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core"
echo "git: $(git -C "$BENCH_DIR/.." rev-parse --short HEAD) ($(git -C "$BENCH_DIR/.." rev-parse --abbrev-ref HEAD))"
echo "python: $PY"

run() {
    local tag=$1; shift
    echo "=== [$(date '+%F %T')] START $tag ==="
    "$PY" -u bench_bao_reconstructor.py --results_dir "$OUT" \
        --repeats "$REPEATS" "$@" 2>&1 | tee "$LOGS/$tag.log"
    echo "=== [$(date '+%F %T')] END $tag (exit ${PIPESTATUS[0]}) ==="
}

run multigrid_jacobi --solver multigrid --smoother jacobi
run ifft_scipy       --solver ifft --fft scipy

echo "=== ALL DONE [$(date '+%F %T')] — CSVs in $BENCH_DIR/$OUT ==="
