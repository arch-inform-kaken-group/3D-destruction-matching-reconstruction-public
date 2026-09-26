#!/usr/bin/env bash
# ============================================================
# Jomon Pottery — Inference on held-out objects (GARF)
# ------------------------------------------------------------
# Loads base GARF.ckpt, applies the fine-tuned LoRA checkpoint,
# and runs two-session flow matching on the held-out Jomon
# objects (on-the-fly dataset, fixed test seeds).
# Writes json_results/ for src/render_local.py.
#
# Usage:
#   bash scripts/infer_jomon.sh
#   LORA_CKPT=logs/GARF-Jomon/<run>/version_0/checkpoints/last.ckpt bash scripts/infer_jomon.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # run from GARF/ root

# ------------------------- config ---------------------------
POTTERY_DIR="${POTTERY_DIR:-../pottery}"     # intact meshes (on-the-fly dataset)
GARF_CKPT="${GARF_CKPT:-output/GARF.ckpt}"   # base PTv3 + FM weights
DEVICES="${DEVICES:-[0]}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LORA_CKPT="${LORA_CKPT:-}"                   # empty => auto-discover newest
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-poisson}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"

# Auto-discover the newest LoRA checkpoint if not provided
if [[ -z "${LORA_CKPT}" ]]; then
  LORA_CKPT=$(find output -name "last.ckpt" -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -n 1 | cut -d' ' -f2-)
fi

# ------------------------- sanity ---------------------------
[[ -n "${LORA_CKPT}" && -f "${LORA_CKPT}" ]] || {
  echo "[ERROR] LoRA checkpoint not found. Run train_jomon.sh first," >&2
  echo "        or set LORA_CKPT=logs/.../checkpoints/last.ckpt" >&2; exit 1; }
[[ -f "${GARF_CKPT}" ]] || { echo "[ERROR] GARF checkpoint not found: ${GARF_CKPT}" >&2; exit 1; }
[[ -d "${POTTERY_DIR}" ]] || { echo "[ERROR] Pottery dir not found: ${POTTERY_DIR}" >&2; exit 1; }

echo "[INFO] Base GARF ckpt : ${GARF_CKPT}"
echo "[INFO] LoRA ckpt      : ${LORA_CKPT}"
echo "[INFO] Pottery dir    : ${POTTERY_DIR}"

# ------------------------- launch ---------------------------
HYDRA_FULL_ERROR=1 python infer_jomon.py \
    experiment=jomon_finetune \
    project_name="GARF-Jomon" \
    experiment_name="jomon_infer" \
    loggers.csv.save_dir=logs/GARF-Jomon \
    data.pottery_dir="${POTTERY_DIR}" \
    data.batch_size="${BATCH_SIZE}" \
    data.num_workers=0 \
    +base_ckpt_path="${GARF_CKPT}" \
    ckpt_path="${LORA_CKPT}" \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true \
    trainer.devices="${DEVICES}" \
    trainer.num_nodes=1 \
    data.mesh_sample_strategy="${SAMPLE_STRATEGY}" \
    ${EXPERIMENT_NAME:+experiment_name=$EXPERIMENT_NAME} \
    tags='["jomon","infer","lora"]'

echo ""
echo "[DONE] Per-object reassembly results (json_results/) written under logs/GARF-Jomon/."
echo "       Pass that json_results/ folder to src/render_local.py for animations."