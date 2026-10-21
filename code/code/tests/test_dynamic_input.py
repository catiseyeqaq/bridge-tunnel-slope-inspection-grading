"""Dynamic input discovery and metadata identity tests (no model)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

from track4_contract_agent.runtime.risk_gate import DEFAULTS, evaluate

OUTER = Path(__file__).resolve().parents[2] / "run.py"
SPEC = importlib.util.spec_from_file_location("entry_run_dynamic", OUTER)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


class DynamicInputTest(unittest.TestCase):
    def test_single_candidate_and_training_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "single" / "任意目录名"
            touch(case / "桥梁" / "a.jpg")
            touch(case / "轨道" / "b.jpg")
            train = root / "single" / "训练集" / "桥梁"
            touch(train / "c.jpg")
            touch(root / "single" / "训练集" / "轨道" / "d.jpg")
            self.assertEqual(ENTRY.find_input_dir("", roots=[root / "single"]), case.resolve())

    def test_multiple_candidates_refuse_to_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "multi"
            for name in ("线上测试集", "线下测试集"):
                touch(root / name / "桥梁" / "a.jpg")
                touch(root / name / "轨道" / "b.jpg")
            with self.assertRaises(RuntimeError):
                ENTRY.find_input_dir("", roots=[root])

    def test_unique_metadata_marker_breaks_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "multi"
            for name in ("A", "B"):
                touch(root / name / "桥梁" / "a.jpg")
                touch(root / name / "轨道" / "b.jpg")
            (root / "B" / "metadata.json").write_text("[]")
            self.assertEqual(ENTRY.find_input_dir("", roots=[root]), (root / "B").resolve())

    def test_explicit_input_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit = root / "explicit"
            touch(explicit / "桥梁" / "a.jpg")
            touch(explicit / "轨道" / "b.jpg")
            self.assertEqual(ENTRY.find_input_dir(str(explicit), roots=[root]), explicit.resolve())

    def test_bridge_names_require_exact_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            touch(input_dir / "桥梁" / "same.jpg")
            touch(input_dir / "轨道" / "rail.jpg")
            source = root / "old_metadata.json"
            source.write_text('[{"questionCategory":"桥梁",'
                              '"imageRelativePath":"old/桥梁/same.jpg",'
                              '"filename":"same.jpg","bridgeName":"旧桥"}]')
            self.assertIsNone(ENTRY._discover_bridge_names(
                input_dir, input_dir / "桥梁", sources=[source]))
            source.write_text('[{"questionCategory":"桥梁",'
                              '"imageRelativePath":"桥梁/same.jpg",'
                              '"filename":"same.jpg","bridgeName":"新桥"}]')
            self.assertEqual(ENTRY._discover_bridge_names(
                input_dir, input_dir / "桥梁", sources=[source]),
                {"桥梁/same.jpg": "新桥"})

    def test_metadata_must_cover_every_input_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            touch(input_dir / "桥梁" / "a.jpg")
            touch(input_dir / "轨道" / "b.jpg")
            rows = [{"imageRelativePath": "桥梁/a.jpg"}]
            with self.assertRaises(ValueError):
                ENTRY._validate_metadata_coverage(input_dir, rows)

    def test_330_metadata_rejected_for_491_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            for i in range(491):
                touch(input_dir / "桥梁" / f"{i:04d}.jpg")
            rows_330 = [{"imageRelativePath": f"桥梁/{i:04d}.jpg"} for i in range(330)]
            with self.assertRaises(ValueError):
                ENTRY._validate_metadata_coverage(input_dir, rows_330)
            rows_491 = [{"imageRelativePath": f"桥梁/{i:04d}.jpg"} for i in range(491)]
            ENTRY._validate_metadata_coverage(input_dir, rows_491)

    def test_491_rows_still_cap_at_six(self):
        rows = [{"question_category": "桥梁", "embedding_ok": True,
                 "p_disease": 0.805, "threshold": 0.8, "type_margin": 0.9,
                 "nearest_similarity": 0.99, "training_support_similarity": 0.95,
                 "training_support_threshold": 0.7252, "low_training_support": False,
                 "head_label": "渗水/泛碱", "vector_sha256": f"{i:064d}"}
                for i in range(491)]
        report = evaluate(rows, DEFAULTS)
        self.assertEqual(report["batch"]["allocation"]["limit"], 6)
        self.assertEqual(report["batch"]["n_teacher_selected"], 6)
        self.assertEqual(report["batch"]["n_budget_overflow"], 485)


if __name__ == "__main__":
    unittest.main()
