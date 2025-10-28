#!/usr/bin/env bash
set -u -o pipefail

# --- sweep grids ---
TEMPS=(0.6 0.7)
TOPKS=(30 20)
TOPPS=(0.95 0.8)
NOISE_BLOCKS=(0 1)   # 0 = without noise, 1 = with noise

# --- misc knobs ---
DRY_RUN=${DRY_RUN:-0}               # DRY_RUN=1 to print only
NUM_GPUS=${NUM_GPUS:-4}
BASE_MODEL="./models/Qwen/Qwen2.5-7B-Instruct"
OUTROOT="${OUTROOT:-runs/$(date +%Y%m%d_%H%M%S)}"
DIRICHLET_FLAG_NAME="${DIRICHLET_FLAG_NAME:-dirichlet_alpha}"  # or: dirichlet_temperature
NO_REDIRECT=${NO_REDIRECT:-0}       # NO_REDIRECT=1 to stream logs to console

mkdir -p "$OUTROOT"

total=$(( ${#TEMPS[@]} * ${#TOPKS[@]} * ${#TOPPS[@]} * ${#NOISE_BLOCKS[@]} ))
idx=0
fail_log="$OUTROOT/_failures.txt"
: > "$fail_log"

for temp in "${TEMPS[@]}"; do
  for topk in "${TOPKS[@]}"; do
    for topp in "${TOPPS[@]}"; do
      for noise in "${NOISE_BLOCKS[@]}"; do
        idx=$((idx + 1))

        RUN_NAME="temp${temp}_topk${topk}_topp${topp}_noise${noise}"
        OUTDIR="${OUTROOT}/${RUN_NAME}"
        mkdir -p "$OUTDIR"

        cmd=(
          python custom_softthinking.py
          --model_name "$BASE_MODEL"
          --max_topk 10
          --max_generated_tokens 2048
          --temperature "$temp"
          --top_p "$topp"
          --top_k "$topk"
          --min_p 0.001
          --after_thinking_temperature 0.6
          --after_thinking_top_p 0.95
          --after_thinking_top_k 30
          --after_thinking_min_p 0.0
          --early_stopping_entropy_threshold 0.01
          --early_stopping_length_threshold 256
          --mem_fraction_static 0.8
          --start_idx 0
          --end_idx 100000
          --num_gpus "$NUM_GPUS"
          --num_samples 5
          --enable_soft_thinking
        )

        if [[ "$noise" -eq 1 ]]; then
          cmd+=(
            --add_noise_gumbel_softmax
            --gumbel_softmax_temperature 0.5
            --add_noise_dirichlet
            "--${DIRICHLET_FLAG_NAME}" 1.0
          )
        fi

        echo "[$idx/$total] $RUN_NAME"
        echo "  -> ${cmd[*]}"

        if [[ "$DRY_RUN" -eq 1 ]]; then
          continue
        fi

        if [[ "$NO_REDIRECT" -eq 1 ]]; then
          "${cmd[@]}"
          status=$?
        else
          # Save full stdout/stderr for the run
          "${cmd[@]}" >"${OUTDIR}/stdout.log" 2>"${OUTDIR}/stderr.log"
          status=$?
        fi

        if [[ $status -ne 0 ]]; then
          echo "✗ FAILED: $RUN_NAME (exit $status)" | tee -a "$fail_log"
        else
          echo "✓ OK: $RUN_NAME"
        fi

      done
    done
  done
done

echo "Done. Expected runs: $total"
[[ -s "$fail_log" ]] && echo "Some runs failed. See $fail_log"
echo "Outputs in: $OUTROOT"
