Here is the cleaned-up, consolidated, and properly formatted README. I fixed the path inconsistencies (standardizing on `jomon_data`), integrated the bash scripts as the primary workflow, and tucked the raw Python commands into collapsible sections to keep it clean.

***

# Jomon Pottery Reassembly with GARF (LoRA Fine-Tuning)

Fragment 85 Jomon pots with `destruct.py`, LoRA fine-tune **GARF** on an A100, run inference on 10 held-out pots, and render reconstruction videos **locally**.

```text
A100  : environment + data + fine-tune + inference   (NO animation libs)
Local : animation rendering only                      (animation libs here)
```

---

## 1. A100 Environment (GARF-recommended: `uv`)

> No animation libraries are installed on the A100.

```bash
# Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd GARF
uv sync                 # base deps
uv sync --extra post    # flash-attn, pytorch3d, torch-scatter, torch-cluster
source .venv/bin/activate
```

Place the downloaded checkpoint inside the GARF directory:
```text
GARF/output/GARF.ckpt      # bundles PTv3 feature extractor + Flow-Matching model
```

Verify it (should print `feature_extractor.*` and `denoiser.*` keys):
```bash
python - <<'PY'
import torch
sd = torch.load("output/GARF.ckpt", map_location="cpu", weights_only=False)["state_dict"]
print("PTv3 :", [k for k in sd if k.startswith("feature_extractor.")][:2])
print("FM   :", [k for k in sd if k.startswith("denoiser.")][:2])
PY
```

---

## 2. Generate Fracture Data (`destruct.py`)

Produces `jomon_data/<name>/{fragments/*.ply, adjacency.json}` for all 85 pots. Point count stays high here (2M); the GARF dataset automatically downsamples to 5000 pts/object during training.

```bash
# Run from project root
python src/destruct.py -i pottery -o jomon_data -n 8 --num-samples 2000000 --seed 42
```

---

## 3. LoRA Fine-Tuning (A100)

Uses GARF's native `train.py` + `enable_lora()` via our wrapper script. The Jomon data module (`GARF/assembly/data/jomon/`) reads the `destruct.py` output directly (no HDF5 conversion needed).

```bash
cd GARF
chmod +x scripts/train_jomon.sh scripts/infer_jomon.sh

# Fine-tune on the 75 training objects
# (Defaults to DATA_ROOT=../jomon_data and GARF_CKPT=output/GARF.ckpt)
bash scripts/train_jomon.sh
```

- **Train split:** First 75 objects (sorted alphabetically).
- **Held-out split:** Last 10 objects (sorted alphabetically).
- **Output:** LoRA weights save to `logs/GARF-Jomon/<run>/checkpoints/last.ckpt`.

<details>
<summary><i>Alternative: Run raw python command</i></summary>

```bash
python train.py \
    experiment=jomon_finetune \
    data.data_root=../jomon_data \
    ckpt_path=output/GARF.ckpt \
    finetuning=true \
    model.feature_extractor_ckpt=null \
    trainer.devices=[0]
```
</details>

---

## 4. Inference on the 10 Held-Out Objects (A100)

Loads the base `GARF.ckpt`, applies your fine-tuned LoRA checkpoint, runs two-session flow matching, and writes `json_results/` (needed for animation).

```bash
# Run from GARF/ directory
# (Automatically finds the newest last.ckpt if LORA_CKPT is not specified)
bash scripts/infer_jomon.sh
```

**Results:** `logs/GARF-Jomon/<run>/json_results/*.json`

<details>
<summary><i>Alternative: Run raw python command</i></summary>

```bash
python infer_jomon.py \
    experiment=jomon_finetune \
    data.data_root=../jomon_data \
    +base_ckpt_path=output/GARF.ckpt \
    ckpt_path=logs/GARF-Jomon/<run>/checkpoints/last.ckpt \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true \
    trainer.devices=[0]
```
</details>

---

## 5. Animation Rendering (Local PC)

> Animation libraries are installed **only here**, never on the A100.

**Transfer data from A100 to your PC:**
1. `jomon_data/` (the fragmented meshes/point clouds)
2. `GARF/logs/GARF-Jomon/<run>/json_results/` (the predicted poses)

**Set up a local environment:**
```bash
python -m venv .venv-anim
# Windows: .venv-anim\Scripts\activate   
# Linux/macOS: source .venv-anim/bin/activate

pip install numpy scipy matplotlib "imageio[ffmpeg]" trimesh
```

**Render MP4s:**
```bash
python src/render_local.py \
    --jomon_data ./jomon_data \
    --results ./json_results \
    --out ./animations
```

**Output:** `animations/<name>.mp4` (scattered → reassembled).

---

## 6. Troubleshooting

| Issue | Fix |
|---|---|
| `UnpicklingError: Weights only load failed` | Use `weights_only=False` in `torch.load` (already patched in `train.py` / `infer_jomon.py`). |
| `ModuleNotFoundError: omegaconf` | `pip install omegaconf hydra-core` (present if you used `uv sync`). |
| OOM during fine-tune | Lower `data.batch_size` (e.g., 2) and raise `trainer.accumulate_grad_batches`. |
| `flash_attn` import error | Ensure `uv sync --extra post` completed; or set `model.denoiser.use_flash_attn=False` in config to use SDPA. |
| `FileNotFoundError: adjacency.json` | Ensure `destruct.py` finished successfully and your `DATA_ROOT` path points to the folder *containing* the object directories (e.g., `../jomon_data`). |