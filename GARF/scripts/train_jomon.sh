# #!/usr/bin/env bash
# # ============================================================
# # Jomon Pottery — LoRA Fine-Tuning (GARF)
# # ------------------------------------------------------------
# # Fine-tunes the bundled GARF.ckpt (PTv3 + Flow-Matching) on
# # the Jomon dataset produced by src/destruct.py.
# #
# # Uses GARF's native train.py + enable_lora(). The Jomon data
# # module (assembly/data/jomon) reads destruct.py output
# # directly — no HDF5 conversion needed.
# #
# # Usage (from anywhere):
# #   bash GARF/scripts/train_jomon.sh
# # Override defaults via env vars, e.g.:
# #   DEVICES=[0,1] BATCH_SIZE=8 bash GARF/scripts/train_jomon.sh
# # ============================================================
# set -euo pipefail
# cd "$(dirname "$0")/.."   # run from GARF/ root

# # ------------------------- config ---------------------------
# DATA_ROOT="${DATA_ROOT:-fragmented_v1}"      # destruct.py output dir
# GARF_CKPT="${GARF_CKPT:-output/GARF.ckpt}"   # bundled PTv3 + FM weights
# DEVICES="${DEVICES:-[0]}"                    # GPU id(s)
# BATCH_SIZE="${BATCH_SIZE:-4}"
# NUM_WORKERS="${NUM_WORKERS:-4}"
# MAX_EPOCHS="${MAX_EPOCHS:-500}"
# HELD_OUT="${HELD_OUT:-10}"                   # last N objects reserved for inference

# # ------------------------- sanity ---------------------------
# [[ -f "${GARF_CKPT}" ]] || { echo "[ERROR] GARF checkpoint not found: ${GARF_CKPT}" >&2; exit 1; }
# [[ -d "${DATA_ROOT}" ]] || { echo "[ERROR] Jomon data dir not found: ${DATA_ROOT}" >&2; exit 1; }

# echo "[INFO] Data root : ${DATA_ROOT}"
# echo "[INFO] GARF ckpt : ${GARF_CKPT}"
# echo "[INFO] Devices   : ${DEVICES}"

# # ------------------------- launch ---------------------------
# HYDRA_FULL_ERROR=1 python train.py \
#     experiment=jomon_finetune \
#     project_name="GARF-Jomon" \
#     loggers.csv.save_dir=logs \
#     data.data_root="${DATA_ROOT}" \
#     data.batch_size="${BATCH_SIZE}" \
#     data.num_workers="${NUM_WORKERS}" \
#     data.held_out="${HELD_OUT}" \
#     model.feature_extractor_ckpt=null \
#     ckpt_path="${GARF_CKPT}" \
#     finetuning=true \
#     trainer.devices="${DEVICES}" \
#     trainer.num_nodes=1 \
#     trainer.max_epochs="${MAX_EPOCHS}" \
#     tags='["jomon","finetune","lora"]'

# echo ""
# echo "[DONE] LoRA checkpoint saved under logs/ (find it with: find logs -name last.ckpt)"













# ------------------------- config ---------------------------
POTTERY_DIR="${POTTERY_DIR:-../pottery}"     # intact source meshes
GARF_CKPT="${GARF_CKPT:-output/GARF.ckpt}"
DEVICES="${DEVICES:-[0]}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-500}"
HELD_OUT="${HELD_OUT:-10}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-poisson}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"

# ------------------------- sanity ---------------------------
[[ -f "${GARF_CKPT}" ]] || { echo "[ERROR] GARF checkpoint not found: ${GARF_CKPT}" >&2; exit 1; }
[[ -d "${POTTERY_DIR}" ]] || { echo "[ERROR] Pottery dir not found: ${POTTERY_DIR}" >&2; exit 1; }

echo "[INFO] Pottery dir : ${POTTERY_DIR}"
echo "[INFO] GARF ckpt   : ${GARF_CKPT}"
echo "[INFO] Devices     : ${DEVICES}"

export MALLOC_ARENA_MAX=2     # stop glibc arena hoarding in workers
export OMP_NUM_THREADS=2      # limit per-worker OpenMP thread buffers
NUM_WORKERS="${NUM_WORKERS:-2}"

# ------------------------- launch ---------------------------
HYDRA_FULL_ERROR=1 python train.py \
    experiment=jomon_finetune \
    project_name="GARF-Jomon" \
    loggers.csv.save_dir=logs \
    data.pottery_dir="${POTTERY_DIR}" \
    data.batch_size="${BATCH_SIZE}" \
    data.num_workers="${NUM_WORKERS}" \
    data.held_out="${HELD_OUT}" \
    model.feature_extractor_ckpt=null \
    ckpt_path="${GARF_CKPT}" \
    finetuning=true \
    trainer.devices="${DEVICES}" \
    trainer.num_nodes=1 \
    trainer.max_epochs="${MAX_EPOCHS}" \
    data.mesh_sample_strategy="${SAMPLE_STRATEGY}" \
    ${EXPERIMENT_NAME:+experiment_name=$EXPERIMENT_NAME} \
    tags='["jomon","finetune","lora"]'