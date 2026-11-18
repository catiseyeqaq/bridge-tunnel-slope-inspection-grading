"""Run the public entry point without weights and check the output contract."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from prepare_assets import sha256
from run import competition_entry, prepare_metadata
from track4_contract_agent.runtime.full_visual import SEVEN


class PortableEntryTest(unittest.TestCase):
    def test_demo_publishes_outputs_and_preserves_initial_predictions(self):
        def launch(path, mode):
            return subprocess.run(
                [sys.executable, "-X", "utf8", "run.py", "--demo", "--mode", mode,
                 "--output-dir", str(path)],
                cwd=ROOT, capture_output=True, encoding="utf-8", timeout=60,
            )

        with tempfile.TemporaryDirectory() as directory:
            first, gate = (Path(directory) / name for name in ("initial", "gate"))
            for target, mode in ((first, "2b_only"), (gate, "risk_gate")):
                result = launch(target, mode)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((first / "result.json").read_bytes(), (gate / "result.json").read_bytes())
            rows = json.loads((gate / "result.json").read_text(encoding="utf-8"))
            self.assertEqual([row["questionCategory"] for row in rows], ["桥梁", "轨道"])
            self.assertTrue(all(tuple(row) == SEVEN for row in rows))
            manifest = json.loads((gate / "run_manifest.json").read_text(encoding="utf-8"))
            timing = json.loads((gate / "infer_time.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["image_backend"], "synthetic_pixel_encoder")
            self.assertEqual(manifest["n_input_images"], manifest["n_embedded"])
            self.assertEqual(manifest["risk_gate"]["n_teacher_selected"], 1)
            self.assertIsNone(manifest["selective_review"])
            self.assertEqual(manifest["run_id"], timing["run_id"])
            self.assertGreater(timing["infer_time"], 0)
            self.assertEqual(manifest["result_sha256"], sha256(gate / "result.json"))
            self.assertNotEqual(launch(gate, "risk_gate").returncode, 0)

    def test_flat_bridge_requires_asset_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bridge = root / "桥梁"
            bridge.mkdir()
            (bridge / "photo.png").touch()
            with self.assertRaisesRegex(ValueError, "可信桥名"):
                prepare_metadata(competition_entry(), root, None)


if __name__ == "__main__":
    unittest.main()
