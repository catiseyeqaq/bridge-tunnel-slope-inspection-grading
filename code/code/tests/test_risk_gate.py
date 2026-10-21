"""P1.1 gate tests: three-level semantics, category quotas, P2 admission cases."""
import unittest

from track4_contract_agent.runtime.risk_gate import DEFAULTS, evaluate


def bridge(p=0.10, margin=0.90, sim=0.99, support=0.90, low=False,
           label=None, digest="0"):
    return {
        "question_category": "桥梁", "embedding_ok": True,
        "p_disease": p, "threshold": 0.8,
        "type_margin": margin, "nearest_similarity": sim,
        "training_support_similarity": support, "training_support_threshold": 0.7252,
        "low_training_support": low,
        "head_label": label if label is not None else ("完好" if p < 0.8 else "渗水/泛碱"),
        "vector_sha256": digest * 64,
    }


def rail(margin=0.90, entropy=0.10, support=0.90, low=False, digest="0"):
    return {
        "question_category": "轨道", "embedding_ok": True,
        "probability_margin": margin, "entropy": entropy,
        "training_support_similarity": support, "training_support_threshold": 0.5886,
        "low_training_support": low, "head_label": "裂缝",
        "vector_sha256": digest * 64,
    }


class RiskGateV11Test(unittest.TestCase):
    def test_normal_high_confidence_support_sufficient(self):
        report = evaluate([bridge(p=0.10), rail(margin=0.90, entropy=0.10)], DEFAULTS)
        batch = report["batch"]
        self.assertEqual(batch["n_risk_detected"], 0)
        self.assertEqual(batch["n_teacher_triggered"], 0)
        self.assertEqual(batch["n_teacher_selected"], 0)
        self.assertFalse(batch["teacher_started"])
        self.assertEqual(batch["teacher_skip_reason"], "no_teacher_triggered")

    def test_ordinary_boundary_detected_but_not_triggered(self):
        report = evaluate([bridge(p=0.85)], DEFAULTS)
        row = report["candidates"][0]
        self.assertTrue(row["risk_detected"])
        self.assertFalse(row["teacher_triggered"])
        self.assertFalse(row["teacher_selected"])
        self.assertEqual(row["selection_status"], "detected_not_triggered")

    def test_severe_bridge_boundary_triggers(self):
        report = evaluate([bridge(p=0.805)], DEFAULTS)
        row = report["candidates"][0]
        self.assertIn("bridge_presence_boundary_severe", row["severe_reasons"])
        self.assertTrue(row["teacher_triggered"])
        self.assertTrue(row["teacher_selected"])

    def test_severe_bridge_margin_and_similarity(self):
        report = evaluate([bridge(p=0.95, margin=0.001, sim=0.30)], DEFAULTS)
        reasons = set(report["candidates"][0]["severe_reasons"])
        self.assertIn("bridge_case_vote_disagreement_severe", reasons)
        self.assertIn("bridge_low_disease_case_similarity_severe", reasons)

    def test_severe_rail_margin_and_entropy(self):
        report = evaluate([rail(margin=0.001, entropy=2.50)], DEFAULTS)
        reasons = set(report["candidates"][0]["severe_reasons"])
        self.assertIn("rail_low_class_margin_severe", reasons)
        self.assertIn("rail_high_class_entropy_severe", reasons)

    def test_high_confidence_with_severe_support_shortfall(self):
        report = evaluate([bridge(p=0.10, margin=0.90, sim=0.99,
                                  support=0.7252 - 0.15, low=True)], DEFAULTS)
        row = report["candidates"][0]
        self.assertTrue(row["high_confidence"])
        self.assertIn("high_confidence_low_training_support", row["severe_reasons"])
        self.assertTrue(row["teacher_triggered"])

    def test_both_categories_get_at_least_one_slot(self):
        rows = [bridge(p=0.805, digest=f"{i:02d}") for i in range(100)]
        rows += [rail(margin=0.001, digest="r") for _ in range(100)]
        rows = [dict(row, vector_sha256=f"{i:064d}") for i, row in enumerate(rows)]
        report = evaluate(rows, DEFAULTS)
        allocation = report["batch"]["allocation"]
        self.assertGreaterEqual(allocation["bridge"], 1)
        self.assertGreaterEqual(allocation["rail"], 1)
        self.assertLessEqual(report["batch"]["n_teacher_selected"], 6)
        self.assertEqual(report["batch"]["n_teacher_selected"], allocation["bridge"] + allocation["rail"])

    def test_only_bridge_triggered_does_not_force_rail(self):
        rows = [bridge(p=0.805, digest=f"{i:02d}") for i in range(3)]
        rows += [rail(margin=0.90, entropy=0.10) for _ in range(197)]
        rows = [dict(row, vector_sha256=f"{i:064d}") for i, row in enumerate(rows)]
        report = evaluate(rows, DEFAULTS)
        allocation = report["batch"]["allocation"]
        self.assertEqual(allocation["rail"], 0)
        self.assertEqual(allocation["bridge"], 3)
        self.assertEqual(report["batch"]["n_teacher_selected"], 3)

    def test_large_batch_respects_budget(self):
        rows = [dict(bridge(p=0.805), vector_sha256=f"{i:064d}") for i in range(200)]
        report = evaluate(rows, DEFAULTS)
        self.assertEqual(report["batch"]["n_teacher_selected"], 6)
        self.assertEqual(report["batch"]["n_budget_overflow"], 194)

    def test_zero_candidates_skips_teacher(self):
        report = evaluate([], DEFAULTS)
        self.assertEqual(report["batch"]["n_teacher_triggered"], 0)
        self.assertEqual(report["batch"]["n_teacher_selected"], 0)
        self.assertFalse(report["batch"]["teacher_started"])
        self.assertEqual(report["batch"]["teacher_skip_reason"], "no_teacher_triggered")

    def test_disabled_gate(self):
        report = evaluate([bridge(p=0.805)], {"enabled": False})
        self.assertFalse(report["enabled"])
        self.assertFalse(report["batch"]["teacher_started"])
        self.assertEqual(report["batch"]["teacher_skip_reason"], "gate_disabled")
        self.assertEqual(report["candidates"], [])

    def test_dual_risk_groups_are_counted(self):
        report = evaluate([bridge(p=0.85, support=0.50, low=True)], DEFAULTS)
        self.assertEqual(report["batch"]["n_dual_triggered"], 1)
        self.assertEqual(report["batch"]["n_low_training_support"], 1)

    def test_selected_and_triggered_are_independent(self):
        rows = [dict(bridge(p=0.805), vector_sha256=f"{i:064d}") for i in range(200)]
        report = evaluate(rows, DEFAULTS)
        selected = [c for c in report["candidates"] if c["teacher_selected"]]
        overflow = [c for c in report["candidates"] if c["budget_overflow"]]
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(overflow), 194)
        self.assertTrue(all(c["review_status"] == "not_started_p1" for c in selected))
        self.assertTrue(all(c["review_status"] == "not_started_budget" for c in overflow))


if __name__ == "__main__":
    unittest.main()
