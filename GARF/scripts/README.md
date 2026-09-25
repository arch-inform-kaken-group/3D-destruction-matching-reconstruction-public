### compare_sampling.sh

**Option 1: Fully automated (Train then Eval)**
If you just ran the training loop for all three and want to immediately evaluate them, the auto-discovery fallback will find them automatically:
```bash
# First, train all 3
MODE=train bash scripts/compare_sampling.sh

# Then, evaluate all 3 (auto-finds the checkpoints)
MODE=eval bash scripts/compare_sampling.sh
```

**Option 2: Explicit Checkpoint Paths**
If you have checkpoints stored in custom locations or want to test specific older versions, pass them directly:
```bash
MODE=eval \
CKPT_uniform="logs/GARF-Jomon/jomon_finetune_uniform/version_0/checkpoints/last.ckpt" \
CKPT_poisson="logs/GARF-Jomon/jomon_finetune_poisson/version_1/checkpoints/epoch=140.ckpt" \
CKPT_wpd="/mnt/c/Users/User/Desktop/custom_wpd_run/last.ckpt" \
bash scripts/compare_sampling.sh
```

### Collecting the final comparison table
After the eval runs finish, use this snippet to aggregate the metrics side-by-side:

```bash
cd GARF && python - <<'PY'
import json, glob
import numpy as np

print(f"{'Strategy':<12} {'PA↑':>7} {'RMSE(R)↓':>9} {'RMSE(T)↓':>9} {'CD↓':>8}")
print("-" * 50)
for s in ["uniform", "poisson", "wpd"]:
    pattern = f"logs/GARF-Jomon/jomon_infer_{s}/*/json_results/*.json"
    files = glob.glob(pattern)
    if not files:
        print(f"{s:<12} {'(no results)':>7}")
        continue
        
    a, r, t, c = [], [], [], []
    for f in files:
        j = json.load(open(f))
        a.append(j.get("part_acc", 0))
        r.append(j.get("rmse_r", 0))
        t.append(j.get("rmse_t", 0))
        c.append(j.get("shape_cd", 0))
        
    print(f"{s:<12} {np.mean(a)*100:6.2f}% {np.mean(r):9.2f} {np.mean(t):9.4f} {np.mean(c):8.5f}")
PY
```