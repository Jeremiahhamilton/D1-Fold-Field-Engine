import hashlib
import unittest
from fractions import Fraction

from translator import DecoderKernel, create_wheel


class TranslatorTests(unittest.TestCase):
    def test_create_wheel_parses_entries(self):
        wheel = create_wheel()

        self.assertEqual(len(wheel), 13)
        self.assertIn("A", wheel)
        self.assertIsNone(wheel["A"].value)
        self.assertEqual(wheel["A"].triplet, "agm")

        self.assertEqual(wheel["H"].value, Fraction(64, 1))
        self.assertEqual(wheel["H"].triplet, "hua")

    def test_ingest_filters_to_wheel_symbols(self):
        wheel = create_wheel()
        kernel = DecoderKernel(wheel)

        kernel.ingest(b"abc")

        digest = hashlib.sha256(b"abc").hexdigest().upper()
        valid_symbols = set(wheel.keys())
        expected_state = [c for c in digest if c in valid_symbols]

        self.assertEqual(kernel.state, expected_state)
        self.assertGreater(len(kernel.state), 0)

    def test_fold_once_rotates_and_accumulates_energy(self):
        wheel = create_wheel()
        kernel = DecoderKernel(wheel)
        kernel.state = ["A", "B", "C"]

        keys = sorted(wheel.keys())
        expected_projected = []
        for symbol in kernel.state:
            triplet = wheel[symbol].triplet
            rotated_char = triplet[1] if len(triplet) > 1 else triplet[0]
            expected_projected.append(keys[ord(rotated_char) % len(keys)])

        expected_energy = sum(
            (wheel[c].value or Fraction(0, 1)) for c in expected_projected
        )

        metrics = kernel.fold_once()

        self.assertEqual(metrics["symbols"], len(expected_projected))
        self.assertEqual(metrics["energy"], expected_energy)
        self.assertEqual(kernel.state, expected_projected)
        self.assertEqual(kernel.history[-1], metrics)


if __name__ == "__main__":
    unittest.main()
