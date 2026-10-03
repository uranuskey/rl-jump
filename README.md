# RL Jump

Portable research training for the V7 serial wheel-leg robot: an estimated RS02 24 V motor envelope, fixed 62.5% attitude assistance, and flat-ground cushioning and stable landing. The executor uses MuJoCo, MuJoCo Warp, and PyTorch. It runs 256 environments for 128 PPO updates after a fresh smoke gate. Simulation and offline checks do not qualify hardware or unassisted operation.

## Install on Windows

Use Python 3.11 (source environment: 3.11.7), Git, an NVIDIA GPU, and a compatible driver. Create the environment in this checkout. For a CUDA 12.6 driver, use the official PyTorch CUDA 12.6 wheel:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe verify_portable.py
```

Use an ASCII checkout path such as `D:\RL_JUMP`. The Windows bootstrap sets process-local `TEMP`/`TMP` to `.tmp` and `WARP_CACHE_PATH` to `.warp_cache` in the checkout because NVRTC can fail on Chinese temporary paths. These generated directories are ignored. It does not change system environment settings.

The source environment used torch 2.8.0+cu128. CUDA wheel compatibility and physical baseline equivalence must be measured on each target machine. CPU verification does not validate GPU physics. `patches/manifest.json` records the official mujoco-warp 3.14.0 wheel and the frozen `forward.py` hash. The local frozen file is byte-identical to the official wheel; no patch is needed. Runtime verification fails if its version or hash changes. The reused RSL-RL modules are vendored with their existing license.

## Run

From the checkout root, first run the CPU checks, then a new 45-environment, 2-update smoke. GPU free-memory, process-lock, native geometry/readback, selective-reset, baseline, and frozen-file gates remain enabled. A passing smoke receipt from this exported manifest is required for training.

```powershell
$py = '.\.venv\Scripts\python.exe'
$task = 'experiments\v7_rs02_24v_flat_landing_reward'
& $py verify_portable.py
.\run_checked.ps1 -Stage smoke -NumEnvs 45 -RunId smoke_01
.\run_checked.ps1 -Stage train -NumEnvs 256 -RunId train_01 -SmokeResult "$task\runs\smoke_01\result.json"
.\run_checked.ps1 -Stage evaluate -RunId eval_01 -TrainResult "$task\runs\train_01\result.json"
& $py "$task\audit_training.py" --training "$task\runs\train_01" --evaluation "$task\runs\eval_01"
```

Use a fresh run ID each time. `run_checked.ps1` waits for the real Python process to exit and records its actual exit code in that run's ignored `exit_receipt.json`. Audit requires successful train and evaluation exit receipts matching the run IDs and frozen manifest, in addition to run statuses. A `STOP` file in the task folder or `experiments/v7_jump_in_place` requests bounded termination. Output goes under ignored `runs/` folders. Plotting is optional; install `matplotlib` to run `plot_training.py` after audit. This repository includes no historical run receipts, trajectories, logs, images, videos, or optimizer output from the landing experiment.

If the virtual environment is outside the checkout, add `-PythonPath` to each wrapper command, for example `-PythonPath '..\.venv\Scripts\python.exe'` when the checkout is a `repo` folder beside the environment. Use that same interpreter for CPU verification and audit.

## Assets and provenance

`SOURCE_PROVENANCE.json` records original source hashes, the original frozen-manifest hash, and the SHA-256 and source of each necessary base checkpoint. `FROZEN.json` under the task records the exported source/resources/models after path relocation. It is deliberately a new manifest; old machine-specific manifests and training results are excluded.

The five checkpoints in `models/` are the actual CPU construction chain: standing policy, voltage policy, learned curve, height-only policy, and the frozen 24 V launch policy at update 64. Each exported checkpoint retains only `model_state_dict`; optimizer states, RNG states, run metadata, and receipts are removed. Provenance records both original and exported hashes. CPU outputs on 32 fixed random inputs match the original standing and launch/landing actors exactly. The standing-policy constructor loads its intermediate architecture and weights before the retained height-only weights, so these dependencies are preserved. No landing-trained model is included.

The full collision XML embeds its mesh assets (approximately 63.7 MB); it is required static robot geometry. It is preserved without changing collisions. No entire virtual environment is included.

The user supplied the custom robot code/assets/checkpoints and authorized this public upload. No additional license is asserted for these custom materials. Third-party vendored licenses remain beside their code or under `patches/`. Review the applicable ownership and licensing before further redistribution.
