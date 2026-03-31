#!/bin/bash
#SBATCH --job-name=judge-eval
#SBATCH --partition=gpu4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/judge_eval_%j.out
#SBATCH --error=logs/judge_eval_%j.err

# ===========================================================================
# Judge Evaluation & Forgetting Analysis on UTSA Arc HPC
# ---------------------------------------------------------------------------
# Partition: gpu1a100 (1x A100 40GB)
#
# NOTE on judge model:
#   - If using Llama 3.1 70B locally, you'll need a DGX node (8x A100 80GB)
#     or multiple H100s. Change the partition accordingly:
#       #SBATCH --partition=<your-dgx-partition>
#       #SBATCH --gres=gpu:4
#       #SBATCH --mem=512G
#   - If using an API (e.g., Together AI, Fireworks, etc.), this script
#     works fine on the A100 partition since the judge runs remotely.
#
# Prerequisite: inference outputs must exist in outputs/ for all 3 checkpoints
# ===========================================================================

echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "=============================================="

# ---------------------------------------------------------------------------
# 1. Environment setup
# ---------------------------------------------------------------------------
module purge
module load cuda/12.1
module load anaconda3
conda activate llm

# ---------------------------------------------------------------------------
# 2. Verify environment
# ---------------------------------------------------------------------------
echo ""
echo "--- GPU Info ---"
nvidia-smi
echo ""

cd /home/xso947/fluffy-umbrella || { echo "Project directory not found!"; exit 1; }

# Check that inference outputs exist
for ckpt in checkpoint_0_base checkpoint_1_alpaca checkpoint_2_json; do
    if [ ! -d "outputs/$ckpt" ]; then
        echo "WARNING: outputs/$ckpt/ not found. Judge will skip this checkpoint."
    else
        echo ">>> Found outputs/$ckpt/"
    fi
done

mkdir -p logs eval_results

# ---------------------------------------------------------------------------
# 3. Run Alpaca evaluation (pairwise judge comparison)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Running Alpaca pairwise judge evaluation..."
echo "    Comparing: Checkpoint 0 vs 1, Checkpoint 1 vs 2, Checkpoint 0 vs 2"
python modular-code/judge.py \
    --eval-type alpaca \
    --outputs-dir outputs \
    --results-dir eval_results

# ---------------------------------------------------------------------------
# 4. Run JSON evaluation (automatic metrics + judge scoring)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Running JSON structured output evaluation..."
python modular-code/judge.py \
    --eval-type json \
    --outputs-dir outputs \
    --results-dir eval_results

# ---------------------------------------------------------------------------
# 5. Run forgetting analysis
# ---------------------------------------------------------------------------
echo ""
echo ">>> Running forgetting analysis (Checkpoint 1 vs Checkpoint 2)..."
python modular-code/result-aggregation.py \
    --results-dir eval_results \
    --output-file eval_results/forgetting_analysis.json

# ---------------------------------------------------------------------------
# 6. Aggregate all results into summary tables
# ---------------------------------------------------------------------------
echo ""
echo ">>> Aggregating results..."
python modular-code/result-aggregation.py \
    --results-dir eval_results \
    --output-file eval_results/summary.json \
    --generate-tables

echo ""
echo "=============================================="
echo "Judge evaluation complete at $(date)"
echo "Results saved to: eval_results/"
echo "=============================================="
