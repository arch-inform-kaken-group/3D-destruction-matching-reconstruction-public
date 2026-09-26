#!/usr/bin/env bash
# ============================================================
# Compare sampling strategies (uniform, poisson, wpd)
# ------------------------------------------------------------
# MODE=train -> 3 full finetunes (saves to logs/GARF-Jomon/jomon_finetune_<s>/)
# MODE=eval  -> 3 inference runs. Accepts exact checkpoint paths via env vars:
#                 CKPT_uniform="logs/.../last.ckpt"
#                 CKPT_poisson="logs/.../last.ckpt"
#                 CKPT_wpd="logs/.../last.ckpt"
#               (Falls back to auto-discovery if variables are empty).
#
# Usage (eval with explicit paths):
#   CKPT_uniform="logs/GARF-Jomon/jomon_finetune_uniform/version_0/checkpoints/last.ckpt" \
#   CKPT_poisson="logs/GARF-Jomon/jomon_finetune_poisson/version_0/checkpoints/last.ckpt" \
#   CKPT_wpd="logs/GARF-Jomon/jomon_finetune_wpd/version_0/checkpoints/last.ckpt" \
#   bash scripts/compare_sampling.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # run from GARF/ root

MODE="${MODE:-eval}"
STRATEGIES="${STRATEGIES:-uniform poisson wpd}"

for s in $STRATEGIES; do
  echo "================ strategy: $s ================"
  
  if [[ "$MODE" == "train" ]]; then
    # Train mode: passes the strategy and names the experiment directory
    SAMPLE_STRATEGY="$s" EXPERIMENT_NAME="jomon_finetune_$s" bash scripts/train_jomon.sh
    
  else
    # Eval mode: resolve the checkpoint path for this specific strategy
    
    # 1. Check for explicit environment variable (e.g., CKPT_wpd)
    CKPT_VAR="CKPT_${s}"
    CKPT_PATH="${!CKPT_VAR:-}"
    
    # 2. Fallback: auto-discover the newest last.ckpt under the strategy's train dir
    if [[ -z "$CKPT_PATH" ]]; then
      TRAIN_DIR="logs/GARF-Jomon/jomon_finetune_${s}"
      if [[ -d "$TRAIN_DIR" ]]; then
        CKPT_PATH=$(find "$TRAIN_DIR" -name "last.ckpt" -printf '%T@ %p\n' 2>/dev/null \
                    | sort -rn | head -n 1 | cut -d' ' -f2-)
      fi
    fi

    # 3. Validate
    if [[ -z "$CKPT_PATH" || ! -f "$CKPT_PATH" ]]; then
      echo "[ERROR] No checkpoint found for strategy '$s'." >&2
      echo "        Set CKPT_${s}=/path/to/last.ckpt or run MODE=train first." >&2
      exit 1
    fi

    echo "[INFO] Using checkpoint: $CKPT_PATH"
    
    # Pass the resolved checkpoint to the inference script
    LORA_CKPT="$CKPT_PATH" \
    SAMPLE_STRATEGY="$s" \
    EXPERIMENT_NAME="jomon_infer_$s" \
    bash scripts/infer_jomon.sh
  fi
done