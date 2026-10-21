# Lean upload note

- This submission contains **no model weights** (no `*.safetensors`, `*.bin`, `*.pt`). The 2B embedding model and the optional 27B teacher/LoRA are provisioned by the evaluation environment.
- To keep the upload small, `code/offline_runtime/wheels/` (91 wheels, 329 MB) is **not** included. It was preserved at `~/track4/legacy/offline_wheels_20261020/wheels`.
- Runtime entries `code/run.py` and `code/run.sh` never read that directory. They use an existing Python that already provides `torch, transformers, yaml, sklearn, numpy, PIL`.
- `code/setup_runtime.py` is the optional full-release recovery helper. It needs the wheel set above; it is not part of the scoring entry.
- The learned assets required at inference remain in the package: `data/bridge_all_view/head_all.npz` (2.8 MB), `data/rail_head/rail_head_v1.npz` (179 KB), `data/training_support/support_bank.npz` (12 MB).
- The vendored `tongagents` / `qwen_vl_utils` wheels stay in `code/code/vendor/` because the runtime installs them only when the SDK is missing.

## 2026-10-20 runtime/backup policy

- **Runtime uses `~/track4_current` (local NVMe) for speed.** OSS `~/track4/assets/p2_runtime_assets` is the cold backup only.
- All teacher configs (`teacher`, `teacher_offline`, `teacher_small`, `preflight`, `unified`) point back to `~/track4_current/...` (teacher config SHA `00ac974e`, the 72.73 submission value).
- After a reboot run `bash ~/track4/reports/restore_models_to_root.sh` to copy 2B/27B/LoRA back to `/root` and verify SHAs. `--verify-only` checks without copying; a valid destination is skipped automatically.
- Verified backup: 2B `c73fa9ca`, 27B config/index `191e0af2`/`77042094` (18/18 shards, 51.75 GB), LoRA `458c2160`/`02df6134`.
- `result/result.json` is unchanged (`a633e7fb`), so the 72.73 score binding is intact. See `code/RUNTIME_POLICY_20261020.json`.
