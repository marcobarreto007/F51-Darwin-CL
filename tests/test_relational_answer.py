"""Real donor excerpts and adversarial controls for relational scoring."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darwin_cl.eval.relational_answer import verdict


COMP = {
    "kind": "composition",
    "answer": "Lina and Vera",
    "query": {"subject_a": "Nex-41", "relation_a": "maker", "subject_b": "Kite-3", "relation_b": "maker"},
}
FALSE = {"kind": "false_claim", "answer": "false", "query": {"subject": "Nex-41", "relation": "maker", "claimed": "Omar"}}


class RelationalAnswerTest(unittest.TestCase):
    def test_real_donor_composition_sentences(self):
        self.assertEqual(verdict(COMP, "Nex-41 maker is Lina and Kite-3 maker is Vera."), "correct")
        self.assertEqual(verdict(COMP, "Nex-41 maker is Lina, Kite-3 maker is Vera."), "correct")
        self.assertEqual(verdict(COMP, "Kite-3 maker is Vera, and Nex-41 maker is Lina."), "correct")

    def test_wrong_pair_and_wrong_binding(self):
        self.assertEqual(verdict(COMP, "Lina and Omar"), "incorrect")
        self.assertEqual(verdict(COMP, "Lina and Vera"), "correct")
        self.assertEqual(verdict(COMP, "Nex-41 maker is Lina and Kite-3 maker is Omar."), "incorrect")
        self.assertEqual(verdict(COMP, "Nex-41 maker is Vera and Kite-3 maker is Lina."), "incorrect")
        self.assertEqual(verdict(COMP, "Nex-41 maker is Lina and Orb-12 maker is Vera."), "incorrect")

    def test_ambiguous_mentions_are_not_credited(self):
        self.assertEqual(verdict(COMP, "Lina and Vera might be wrong."), "review")
        self.assertEqual(verdict(COMP, "Not Lina and Vera."), "review")

    def test_false_claim_has_separate_yes_no_semantics(self):
        self.assertEqual(verdict(FALSE, "No"), "correct")
        self.assertEqual(verdict(FALSE, "False."), "correct")
        self.assertEqual(verdict(FALSE, "Yes"), "incorrect")
        self.assertEqual(verdict(FALSE, "No, the maker of Nex-41 is Lina."), "review")
        self.assertEqual(verdict(FALSE, "No, the maker of Nex-41 is Omar."), "review")

    def test_missing_is_not_a_false_claim(self):
        missing = {"kind": "missing", "answer": "INDETERMINADO", "query": {"kind": "missing"}}
        self.assertEqual(verdict(missing, "INDETERMINADO"), "correct")
        self.assertEqual(verdict(missing, "false"), "incorrect")
        self.assertEqual(verdict(missing, "No motto is recorded."), "review")

    def test_inverse_wrong_entity_is_not_credited(self):
        inverse = {"kind": "inverse", "answer": "Nex-41", "query": {"kind": "inverse"}}
        self.assertEqual(verdict(inverse, "Nex-41"), "correct")
        self.assertEqual(verdict(inverse, "Lina"), "incorrect")
        self.assertEqual(verdict(inverse, "Nex-41 maker is Lina."), "review")


if __name__ == "__main__":
    unittest.main()
