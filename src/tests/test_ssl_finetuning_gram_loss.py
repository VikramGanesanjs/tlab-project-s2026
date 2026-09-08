from __future__ import annotations

import unittest

import torch

from ssl_finetuning.gram_loss import GramLoss


class TestSpatialGramLoss(unittest.TestCase):
    def test_disabled_window_matches_full_gram_loss(self) -> None:
        student = torch.tensor([[[1.0, 0.0], [0.5, 1.0], [-1.0, 0.0], [0.0, -1.0]]])
        teacher = torch.tensor([[[0.8, 0.2], [0.0, 1.0], [-0.8, 0.2], [0.2, -0.8]]])
        disabled = GramLoss(spatial_window_size=0)
        reference = GramLoss()
        self.assertTrue(torch.allclose(disabled(student, teacher), reference(student, teacher)))

    def test_three_by_three_window_uses_patch_distance(self) -> None:
        loss = GramLoss(spatial_window_size=3)
        mask = loss._spatial_window_mask(16, torch.device("cpu"))

        # A corner has itself plus a 2x2 clipped neighborhood; an interior
        # patch retains its full 3x3 neighborhood, including diagonal patches
        # at sqrt(2) patch-size distance.
        self.assertEqual(int(mask[0].sum()), 4)
        self.assertEqual(int(mask[5].sum()), 9)
        self.assertTrue(mask[5, 0])
        self.assertFalse(mask[5, 15])
        self.assertTrue(torch.equal(mask, mask.T))

    def test_window_zeros_distant_entries_in_both_gram_matrices(self) -> None:
        # The only student/teacher discrepancy is a correlation between the
        # diagonally opposite patches of a 2x2 grid. A 1x1 window removes it.
        teacher = torch.eye(4).unsqueeze(0)
        student = teacher.clone()
        student[0, 0] = student[0, 3]

        full_loss = GramLoss(spatial_window_size=0)(student, teacher)
        local_loss = GramLoss(spatial_window_size=1)(student, teacher)
        self.assertGreater(full_loss.item(), 0.0)
        self.assertEqual(local_loss.item(), 0.0)

    def test_rejects_even_window_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive odd"):
            GramLoss(spatial_window_size=2)


if __name__ == "__main__":
    unittest.main()
