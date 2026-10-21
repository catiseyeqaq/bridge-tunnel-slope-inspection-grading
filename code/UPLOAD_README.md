# P2 code-only submission (system packages ~/result contents)

- The system package contains only `code/`, `design/`, `result/`; it does **not** include 2B/27B weights.
- Default entry: `code/run.py` with no arguments; it loads `submission_risk_gate_teacher.yaml`.
- 2B resolution in the evaluation environment (in order): package `model_assets/embedding_2b` (absent in this code-only package), `~/track4_current/code/model_assets/bases/embedding_2b`, `~/track4_current/code/model_assets/bases/embedding_2b`, `~/models/Qwen3-VL-Embedding-2B`, `/data/model/...`, `/model/...`.
- 27B resolution: `code/configs/submission_risk_gate_teacher.yaml` uses `~/track4_current/code/model_assets/bases/teacher_27b` and `teacher_clean_full_20261011` for local-NVMe speed; SHA is checked before launch. OSS keeps a cold backup under `~/track4/assets/p2_runtime_assets`; after a reboot run `~/track4/reports/restore_models_to_root.sh` to copy it back to `/root`.
- If 27B is missing/SHA-mismatched, preflight keeps the legal 2B result. If 2B is also missing, startup fails explicitly; no other model is substituted.
- Actual capability without 27B: **2B main chain + optional 27B fallback**.
- `result/result.json` is the online330 deployment regression sample, **not** the offline final result. The onsite entry must regenerate `result.json` and `infer_time.json` from the mounted input.
- No SGLang, no new model service, no retraining, no gate/LoRA/acceptance-rule changes.

## Post-cleanup recovery

See START_HERE.md. OSS `~/track4/assets/p2_runtime_assets` is the cold backup; `~/track4_current` is the fast runtime path. After a reboot run `~/track4/reports/restore_models_to_root.sh`, then start the batch. The three small learned NPZ files are required package assets; foundation weights and LoRA remain external.
