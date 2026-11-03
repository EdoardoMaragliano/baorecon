#!/bin/bash
# GPU benchmark sweep of bench_bao_reconstructor.py (ifft, baorecon_gpu only), same
# configuration as the 27 July 2026 run on teogpu02 (results_teogpu02/
# bao_reconstructor_ifft_scipy_gpu.csv): N=1e6, nmesh 128/256/512/1024, repeats=5.
#
# Submit (from a node where sbatch works, e.g. teogpu02):
#   BENCH_OUT=results_20261103 sbatch --export=ALL,BENCH_OUT benchmarks/slurm/bench_gpu.sbatch
#
#SBATCH --job-name=bench_gpu
#SBATCH --partition=teogpu
#SBATCH --nodelist=teogpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# A whole A40: the 1024^3 point needs ~45 GiB of VRAM (CuPy pool high-water mark
# 47.3 GB in July), so the GPU cannot be shared.
#SBATCH --gres=gpu:teslaga102:1
#SBATCH --mem=64G
# July: ~7 min for the four points; margin for compilation and queueing of workers.
#SBATCH --time=02:00:00
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
exec > "$LOGS/session_gpu_${SLURM_JOB_ID}.log" 2>&1

echo "=== job $SLURM_JOB_ID on $(hostname) | $(date '+%F %T') ==="
echo "cpus-per-task=$SLURM_CPUS_PER_TASK mem=$SLURM_MEM_PER_NODE MB CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo "git: $(git -C "$BENCH_DIR/.." rev-parse --short HEAD) ($(git -C "$BENCH_DIR/.." rev-parse --abbrev-ref HEAD))"
echo "python: $PY"

run() {
    local tag=$1; shift
    echo "=== [$(date '+%F %T')] START $tag ==="
    "$PY" -u bench_bao_reconstructor.py --results_dir "$OUT" \
        --repeats "$REPEATS" "$@" 2>&1 | tee "$LOGS/$tag.log"
    echo "=== [$(date '+%F %T')] END $tag (exit ${PIPESTATUS[0]}) ==="
}

run ifft_scipy_gpu --solver ifft --fft scipy --gpu_only

echo "=== ALL DONE [$(date '+%F %T')] — CSVs in $BENCH_DIR/$OUT ==="
