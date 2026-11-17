"""Nontrivial reference-frame and normalization round trips for the EEF codec."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from cosmos_framework.data.generator.action.utils.robodojo_eef import (
    denormalize,
    fit_normalizer,
    matrix_rot6d,
    normalize,
    pose_errors,
    recover16,
    relative20,
)


class EEFCodecTests(unittest.TestCase):
    def test_nonidentity_reference_and_quaternion_sign_roundtrip(self):
        rng = np.random.default_rng(42)
        pose = rng.normal(size=(17, 2, 8))
        q = Rotation.random(34, random_state=rng).as_quat()[:, [3, 0, 1, 2]].reshape(17, 2, 4)
        pose[..., 3:7] = q
        pose = pose.reshape(17, 16)
        action = relative20(pose[:1], pose[1:])
        recovered = recover16(pose[:1], action)
        position, angle = pose_errors(recovered, pose[1:])
        self.assertLess(position.max(), 1e-12)
        self.assertLess(angle.max(), 1e-12)
        negative = pose.copy().reshape(17, 2, 8)
        negative[..., 3:7] *= -1
        np.testing.assert_allclose(
            relative20(negative[:1].reshape(1, 16), negative[1:].reshape(16, 16)), action, atol=1e-12
        )

    def test_rot6d_is_columns_and_heldout_is_not_clipped(self):
        matrix = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        np.testing.assert_allclose(matrix_rot6d(matrix), [0, 1, 0, -1, 0, 0], atol=1e-12)
        train = np.arange(2000, dtype=float).reshape(100, 20)
        stats = fit_normalizer(train)
        heldout = train[-1:] * 2
        self.assertGreater(normalize(heldout, stats).max(), 1)
        np.testing.assert_allclose(denormalize(normalize(heldout, stats), stats), heldout, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
