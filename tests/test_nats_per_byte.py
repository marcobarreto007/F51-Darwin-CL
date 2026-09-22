import math
import unittest

from darwin_cl.eval.boundaries import (
    CONTEXT,
    PAD,
    SCORED,
    SPECIAL,
    evaluated_utf8_bytes,
    total_scored_nll,
)
from darwin_cl.eval.metrics import bits_per_byte, nats_per_byte, nats_per_token


class NatsPerByteTests(unittest.TestCase):
    def test_formula_is_sum_nll_over_utf8_bytes(self) -> None:
        total = 1.0 + 2.0 + 3.0
        self.assertEqual(nats_per_byte(total, 6), 1.0)
        self.assertAlmostEqual(bits_per_byte(total, 6), 1.0 / math.log(2))

    def test_padding_and_boundary_bytes_are_excluded(self) -> None:
        text = "áx"
        # á is U+00E1, two UTF-8 bytes. x is one byte.
        # token 0 covers "á" and is context, so those two bytes are not evaluated.
        # token 1 covers "x" and is scored.
        # token 2 is an artificial pad with an empty span.
        offsets = [(0, 1), (1, 2), (0, 0)]
        roles = [CONTEXT, SCORED, PAD]
        self.assertEqual(evaluated_utf8_bytes(text, offsets, roles), 1)
        nll = total_scored_nll([2.5, 9.0], [SCORED, PAD])
        self.assertEqual(nll, 2.5)
        self.assertEqual(nats_per_byte(nll, evaluated_utf8_bytes(text, offsets, roles)), 2.5)

    def test_special_token_does_not_add_bytes(self) -> None:
        text = "ab"
        offsets = [(0, 1), (1, 2), (0, 0)]
        roles = [CONTEXT, SCORED, SPECIAL]
        self.assertEqual(evaluated_utf8_bytes(text, offsets, roles), 1)
        self.assertEqual(total_scored_nll([4.0, 8.0], [SCORED, SPECIAL]), 4.0)

    def test_internal_token_metric_is_separate(self) -> None:
        self.assertEqual(nats_per_token(4.0, 2), 2.0)

    def test_zero_bytes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            nats_per_byte(1.0, 0)

    def test_manual_multibyte_span_uses_bytes_not_characters(self) -> None:
        text = "café ação 日本"
        # Character index: c a f é space a ç ã o space 日 本
        # 0 1 2 3 4     5 6 7 8 9     10 11
        # Manual UTF-8 sizes: c a f = 1 each, é = 2, space = 1, a = 1,
        # ç = 2, ã = 2, o = 1, space = 1, 日 = 3, 本 = 3.
        manual_raw_bytes = 1 + 1 + 1 + 2 + 1 + 1 + 2 + 2 + 1 + 1 + 3 + 3
        self.assertEqual(len(text), 12)
        self.assertEqual(len(text.encode("utf-8")), 19)
        self.assertEqual(manual_raw_bytes, 19)
        self.assertNotEqual(len(text), manual_raw_bytes)
        # Causal shift. t0 = "caf" is context and is not in the denominator.
        # Scored pieces: "é a" = 4 bytes, "ção " = 6 bytes, "日本" = 6 bytes.
        offsets = [(0, 3), (3, 6), (6, 10), (10, 12)]
        roles = [CONTEXT, SCORED, SCORED, SCORED]
        individual = [0.5, 1.5, 2.0]
        reported_total = total_scored_nll(individual, [SCORED, SCORED, SCORED])
        self.assertEqual(reported_total, sum(individual))
        scored_bytes = evaluated_utf8_bytes(text, offsets, roles)
        self.assertEqual(scored_bytes, 4 + 6 + 6)
        self.assertEqual(scored_bytes, 16)
        self.assertNotEqual(scored_bytes, len(text[3:]))
        self.assertEqual(
            nats_per_byte(reported_total, scored_bytes),
            reported_total / scored_bytes,
        )
        self.assertEqual(nats_per_byte(reported_total, scored_bytes), 0.25)


if __name__ == "__main__":
    unittest.main()
