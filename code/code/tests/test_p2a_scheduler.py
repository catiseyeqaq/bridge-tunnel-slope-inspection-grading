"""P2-A scheduler/preflight tests. No real 27B is loaded or started."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from track4_contract_agent.runtime.selective_review import (
    apply_selective_review, checked_config, preflight_teacher,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "submission_risk_gate_preflight.yaml"


def rail_row():
    return {"questionCategory": "轨道", "defectType": "裂缝",
            "defectDescription": "", "ratingScale(1-5)": ""}


def rail_evidence():
    return {"question_category": "轨道", "embedding_ok": True,
            "probability_margin": 0.90, "entropy": 0.10}


class LocalTeacherFixture(unittest.TestCase):
    def setUp(self):
        super().setUp()
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["selective_review"]
        root = Path(scratch.name)
        for key in ("base_model", "adapter"):
            self.raw[key] = str(root / key)
            Path(self.raw[key]).mkdir()
        for group, filename, sha_key in (
            ("base_model", "config.json", "base_config_sha256"),
            ("base_model", "model.safetensors.index.json", "base_index_sha256"),
            ("adapter", "adapter_model.safetensors", "adapter_sha256"),
            ("adapter", "adapter_config.json", "adapter_config_sha256"),
        ):
            payload = f"fixture {filename}".encode()
            (Path(self.raw[group]) / filename).write_bytes(payload)
            self.raw[sha_key] = hashlib.sha256(payload).hexdigest()
        self.cfg = checked_config(self.raw)


class P2APreflightTest(LocalTeacherFixture):

    def test_four_selected_passes_time_budget(self):
        result = preflight_teacher(self.cfg, 4, gpu_free_gib=90.0)
        self.assertTrue(result["assets_ok"])
        self.assertTrue(result["sha_ok"])
        self.assertTrue(result["gpu_ok"])
        self.assertEqual(result["time_estimate_s"], 106.0)
        self.assertTrue(result["resource_ok"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["teacher_started"])

    def test_six_selected_fails_conservative_time_budget(self):
        result = preflight_teacher(self.cfg, 6, gpu_free_gib=90.0)
        self.assertEqual(result["time_estimate_s"], 130.0)
        self.assertFalse(result["time_ok"])
        self.assertIn("time_budget_insufficient", result["reasons"])
        self.assertFalse(result["resource_ok"])

    def test_missing_assets_are_reported(self):
        cfg = dict(self.cfg)
        cfg["base_model"] = "/nonexistent/teacher_27b"
        cfg["adapter"] = "/nonexistent/teacher_lora"
        result = preflight_teacher(cfg, 2, gpu_free_gib=90.0)
        self.assertFalse(result["assets_ok"])
        self.assertIn("teacher_assets_missing", result["reasons"])
        self.assertFalse(result["resource_ok"])

    def test_sha_mismatch_is_reported(self):
        cfg = dict(self.cfg)
        cfg["adapter_sha256"] = "0" * 64
        result = preflight_teacher(cfg, 2, gpu_free_gib=90.0)
        self.assertTrue(result["assets_ok"])
        self.assertFalse(result["sha_ok"])
        self.assertIn("teacher_sha_mismatch", result["reasons"])
        self.assertFalse(result["resource_ok"])

    def test_gpu_insufficient_is_reported(self):
        result = preflight_teacher(self.cfg, 2, gpu_free_gib=10.0)
        self.assertFalse(result["gpu_ok"])
        self.assertIn("gpu_free_insufficient", result["reasons"])
        self.assertFalse(result["resource_ok"])


class P2ASchedulerTest(LocalTeacherFixture):
    def setUp(self):
        super().setUp()
        self.raw["mode"] = "preflight"
        device = patch(
            "track4_contract_agent.runtime.selective_review.preflight_teacher",
            side_effect=lambda config, count: preflight_teacher(config, count, gpu_free_gib=90.0),
        )
        device.start()
        self.addCleanup(device.stop)

    def test_only_gate_selected_rows_are_scheduled(self):
        rows = [rail_row() for _ in range(4)]
        evidence = [rail_evidence() for _ in range(4)]
        paths = [f"image_{i}.jpg" for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            merged, summary = apply_selective_review(
                rows, evidence, paths, {"轨道": {"裂缝", "剥落"}}, None,
                self.raw, Path(tmp), selected_rows=[2, 0], run_mode="preflight")
            root = Path(tmp) / "selective_review"
            self.assertEqual(summary["routing_source"], "risk_gate")
            self.assertEqual(summary["teacher_selected"], 2)
            self.assertFalse(summary["teacher_started"])
            self.assertEqual(summary["teacher_completed"], 0)
            self.assertEqual(summary["teacher_accepted"], 0)
            self.assertEqual(summary["status"], "preflight_passed_not_started")
            routing = json.loads((root / "routing.json").read_text(encoding="utf-8"))
            self.assertEqual(routing["source"], "risk_gate")
            self.assertEqual([d["row_index"] for d in routing["selected"]], [2, 0])
            self.assertFalse((root / "request.json").exists())
            self.assertFalse((root / "responses.jsonl").exists())
            self.assertFalse((root / "responses.log").exists())
            decisions = json.loads((root / "decisions.json").read_text(encoding="utf-8"))
            self.assertTrue(all(d["status"] == "not_started_preflight" for d in decisions))
            self.assertTrue(all(d["teacher_started"] is False for d in decisions))
            self.assertEqual(merged, rows)

    def test_zero_selected_skips_preflight_and_assets(self):
        rows = [rail_row()]
        with tempfile.TemporaryDirectory() as tmp:
            merged, summary = apply_selective_review(
                rows, [rail_evidence()], ["image_0.jpg"], {"轨道": {"裂缝"}}, None,
                self.raw, Path(tmp), selected_rows=[], run_mode="preflight")
            root = Path(tmp) / "selective_review"
            self.assertEqual(summary["status"], "no_teacher_selected")
            self.assertEqual(summary["teacher_selected"], 0)
            self.assertFalse(summary["teacher_started"])
            self.assertFalse((root / "preflight.json").exists())
            self.assertFalse((root / "request.json").exists())
            self.assertEqual(merged, rows)

    def test_out_of_range_selected_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                apply_selective_review(
                    [rail_row()], [rail_evidence()], ["image_0.jpg"],
                    {"轨道": {"裂缝"}}, None, self.raw, Path(tmp),
                    selected_rows=[9], run_mode="preflight")

    def test_missing_assets_preserve_initial_prediction(self):
        rows = [rail_row()]
        worker = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = {**self.raw, "adapter": str(root / "missing")}
            merged, summary = apply_selective_review(
                rows, [rail_evidence()], ["image_0.jpg"], {"轨道": {"裂缝"}}, None,
                raw, root, selected_rows=[0], run_mode="real", worker_runner=worker,
            )
            self.assertEqual(merged, rows)
            self.assertFalse(summary["teacher_started"])
            self.assertIn("teacher_assets_missing", summary["skip_reason"])
            worker.assert_not_called()

    def test_real_interface_with_simulated_worker(self):
        """Simulated worker only; not a real TongAgents call or audit log."""
        rows = [rail_row()]
        evidence = [rail_evidence()]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_runner(request_path, response_path, cfg, worker_path):
                payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
                response = {
                    "row_index": 0, "ok": True,
                    "request_sha256": payload["request_sha256"],
                    "raw": json.dumps({"type": "裂缝", "rating": "",
                                       "evidence": "图中可见裂缝", "uncertain": False},
                                      ensure_ascii=False),
                    "elapsed_s": 1.25,
                }
                Path(response_path).write_text(json.dumps(response, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
                return {"status": "completed", "returncode": 0, "elapsed_s": 1.25,
                        "log": "simulated.log", "simulated": True}

            merged, summary = apply_selective_review(
                rows, evidence, ["image_0.jpg"], {"轨道": {"裂缝"}}, None,
                self.raw, root, selected_rows=[0], run_mode="real",
                worker_runner=fake_runner)
            self.assertTrue(summary["teacher_started"])
            self.assertEqual(summary["teacher_completed"], 1)
            self.assertEqual(summary["teacher_accepted"], 1)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["statuses"], {"accepted_unchanged": 1})
            self.assertEqual(merged[0]["defectType"], "裂缝")


if __name__ == "__main__":
    unittest.main()
