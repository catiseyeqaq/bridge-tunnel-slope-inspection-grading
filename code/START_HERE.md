# Current P2 deployment and recovery

For a GitHub checkout, start with [the portable entry and setup](../REPRODUCIBILITY.md).
The instructions below describe the original contest machine and its pre-provisioned assets.

The authoritative inspection entry is code/run.py in the submission root. It uses
the frozen risk_gate_teacher configuration: all-image 2B inspection, severe-risk
selection, budgeted trained-27B review, and final seven-field validation.

The lean upload contains three required small learned NPZ assets (15,339,192 bytes)
and code. Offline wheels are not included. It does not contain 2B/27B foundation
weights or LoRA.
Separately provision the matching model files before restoring a new machine.

The OSS copy under ~/track4/assets/p2_runtime_assets is the cold backup.
For inference speed the runtime expects the weights under ~/track4_current.
After a reboot run:

```bash
bash ~/track4/reports/restore_models_to_root.sh
```

It copies 2B/27B/LoRA back to /root and verifies the SHAs before the batch.
Use --verify-only to check without copying.

On the official Python 3.12 PPU image:

```bash
source /etc/profile.d/dockerenv.sh
/usr/local/bin/python3 ~/result/code/setup_runtime.py
```

An existing virtual environment is never overwritten. The official PPU image
Python already provides torch/transformers; after a reboot use it directly.

```bash
/usr/local/bin/python3 ~/result/code/run.py --input-dir INPUT --metadata METADATA --output-dir NEW_OUTPUT
```

Use trustworthy current metadata for bridge identity. The output directory must
be separate from the input. Never reuse a failed scratch directory. Explicit
input/output arguments are useful for rehearsal and do not change the model or
gate configuration. No arguments are accepted only when input discovery is
unambiguous. The final input must be inferred again on site.

RELEASE_MANIFEST.json describes the complete current runtime, including the P2
gate sources and all matching external assets. PACKAGE_MANIFEST.json covers the
actual submission files, excluding itself. P2_RELEASE_MANIFEST.json binds the
frozen entry, configuration and sample prediction. It contains no final-test
accuracy claim. The historical sample's timing is not replaced by rehearsal time.
