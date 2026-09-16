# 3D Destruction Reconstruction

Destruction of 3D models into fragments. [Test Pottery Download](https://drive.google.com/file/d/1B1qDF82Va_aGWw5gxO1H_afQ3ONYufqd/view?usp=sharing) unzip into the project root.

## SETUP

- SETUP Python 3.12 ENV [Python 3.12](https://www.python.org/downloads/release/python-31211/)

    ```
    python3.12 -m venv descon
    ```

    OR use search bar from VS Code

- Source ENV

    ```
    descon/Scripts/activate.bat
    ```

    OR set as interpreter from VS Code

- Install packages

    ```
    pip install -r requirements.txt
    ```

## USAGE

```python
========================================================================
COLOR CONFIGURATION (Source Code Constants)
========================================================================
Adjust these constants at the top of the file to change default material colors:
  - DEFAULT_GRAY (float): Fallback color for untextured meshes (0.0 to 1.0).
  - CLAY (RGB array): Default oxidized surface clay color. Used for fill and surface.
  - CORE (RGB array): Default reduced/dark clay core color for cross-sections.
  - LUM (RGB array): Luminance weights for shading palette colors based on source texture.

These can be overridden via CLI using --surface-color, --core-color, and --fill-color.

========================================================================
COMMAND LINE INTERFACE (CLI) USAGE
========================================================================
Basic Usage:
  python destruct.py -i pottery/mesh.glb -o output_dir/

1. POINT CLOUD & FILL BASICS
  --num-samples N         Number of surface points to sample (default: 2,000,000).
  --fill-color R G B      RGB color for fracture fill fallback (default: CLAY).

2. STRESS & VULNERABILITY FIELD
  --stress-model MODEL    ['fem_lite', 'impact', 'gravity', 'curvature', 'none'].
                          'fem_lite' uses Jacobi iteration; 'impact' uses directional impact;
                          'gravity' uses center-of-mass leverage; 'curvature' uses mean curvature.
  --grid-res N            Voxel grid resolution for FEM-lite (default: 64).
  --jacobi-iters N        Iterations for FEM-lite solver (default: 150).
  --impact-dir X Y Z      Direction vector for 'impact' stress model.
  --w-vol / --w-curv / --w-thin
                          Weights for volumetric stress, curvature, and inverse thickness.
  --thickness-alpha       Exponent for thickness gain (t_ref/t)^alpha (default: 1.5).
  --thickness-cap         Maximum cap for thickness gain (default: 5.0).
  --noise-sigma           Standard deviation for stochastic stress multiplier (default: 0.5).
  --noise-freqs           Number of frequency bands for band-limited noise (default: 48).
  --noise-length-factor   Wavelength scaling factor for stress noise (default: 0.33).

3. SEED GENERATION & SHARD SIZES
  --seed-radius-sigma     Jitter for NMS exclusion radius (0=uniform, >0=stochastic).
  --seed-score-noise      Jitter for NMS stress scores (default: 0.05).
  --rim-weight            Bias multiplier for seeds near mesh boundaries/rims.
  --rim-length-scale      Decay length for rim proximity bias (fraction of bbox diag).
  --curv-bias-weight      Bias multiplier for high-curvature regions.
  --thin-bias-weight      Bias multiplier for thin regions.
  --size-sigma            Log-normal sigma for shard size variation (0=uniform size).
  --power-weight-scale    Global scale for power-diagram weights (default: 0.15).

4. FRACTURE SURFACE RELIEF & TEXTURE
  --height-amp-scale      Amplitude of FBM relief as fraction of local thickness (default: 0.18).
  --height-H              Hurst exponent for FBM (0.0 to 1.0, higher = smoother).
  --height-octaves        Number of FBM octaves (default: 5).
  --grain-amp-scale       Amplitude of high-frequency Gaussian grain (default: 0.03).
  --height-fade-fraction  Fraction of wall thickness where relief fades to 0 at edges.
  --chip-prob             Base probability for micro-chipping at the fracture rim.
  --chip-width-scale      Width of the micro-chipping erosion zone in pixels.
  --fill-color-model      ['constant', 'depth_core']. 'depth_core' blends surface to core.
  --surface-color R G B   RGB color for the outer oxidized margin (default: CLAY).
  --core-color R G B      RGB color for the inner reduced core (default: CORE).

5. ASSEMBLY VISUALS & FRAGMENTATION
  --assembly-color-mode   ['shaded', 'flat']. 'shaded' modulates palette by source luminance.
  --palette-sat           Saturation for the golden-angle fragment palette (default: 0.75).
  --palette-val           Value/brightness for the fragment palette (default: 0.95).
  -n, --fragments N       Target number of fragments/shards (default: 16).
  --explode-distance      Absolute world-unit explode distance (overrides scale).
  --explode-scale         Explode distance as fraction of bounding box diagonal (default: 0.12).
  --min-faces             Minimum faces required to keep a fragment (default: 20).
  --seed                  Random seed for reproducibility (default: random).
```