"""Lossless20D dual-arm EEF codec: environment-frame offsets from chunk start.

Each arm is [xyz, first rotation-matrix column, second column, gripper].
Relative rotations are R_target @ R_reference.T; grippers remain absolute.
This matches the column-based rot6d convention in pose_utils.convert_rotation.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if not np.isfinite(q).all() or np.any(np.linalg.norm(q, axis=-1) < 1e-8):
        raise ValueError("Invalid wxyz quaternion")
    return Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(*q.shape[:-1], 3, 3)


def matrix_quaternion(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    q = Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat().reshape(*m.shape[:-2], 4)
    return q[..., [3, 0, 1, 2]]


def matrix_rot6d(matrix):
    m = np.asarray(matrix)
    return m[..., :, :2].swapaxes(-1, -2).reshape(*m.shape[:-2], 6)


def rot6d_matrix(rotation):
    r = np.asarray(rotation, dtype=np.float64)
    a, b = r[..., :3], r[..., 3:6]
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    if np.any(np.linalg.norm(a, axis=-1) < 0.5) or np.any(np.linalg.norm(b, axis=-1) < 0.5):
        raise ValueError("Degenerate6D rotation")
    return np.stack((a, b, np.cross(a, b, axis=-1)), axis=-1)


def absolute20(pose16):
    p = np.asarray(pose16, dtype=np.float64).reshape(*np.shape(pose16)[:-1], 2, 8)
    return np.concatenate((p[..., :3], matrix_rot6d(quaternion_matrix(p[..., 3:7])), p[..., 7:8]), axis=-1).reshape(
        *p.shape[:-2], 20
    )


def relative20(reference16, target16):
    ref = np.asarray(reference16, dtype=np.float64).reshape(*np.shape(reference16)[:-1], 2, 8)
    target = np.asarray(target16, dtype=np.float64).reshape(*np.shape(target16)[:-1], 2, 8)
    delta = quaternion_matrix(target[..., 3:7]) @ quaternion_matrix(ref[..., 3:7]).swapaxes(-1, -2)
    return np.concatenate((target[..., :3] - ref[..., :3], matrix_rot6d(delta), target[..., 7:8]), axis=-1).reshape(
        *target.shape[:-2], 20
    )


def recover16(reference16, relative):
    ref = np.asarray(reference16, dtype=np.float64).reshape(*np.shape(reference16)[:-1], 2, 8)
    action = np.asarray(relative, dtype=np.float64).reshape(*np.shape(relative)[:-1], 2, 10)
    target_rotation = rot6d_matrix(action[..., 3:9]) @ quaternion_matrix(ref[..., 3:7])
    return np.concatenate(
        (action[..., :3] + ref[..., :3], matrix_quaternion(target_rotation), action[..., 9:10]), axis=-1
    ).reshape(*action.shape[:-2], 16)


def fit_normalizer(values):
    values = np.asarray(values)
    if values.shape[-1] != 20 or not np.isfinite(values).all():
        raise ValueError("Expected finite20D values")
    q01, q99 = np.quantile(values.reshape(-1, 20), [0.01, 0.99], axis=0)
    scale = np.maximum((q99 - q01) / 2, 1e-6)
    return dict(
        kind="q01_q99_symmetric_no_clip",
        center=((q01 + q99) / 2).tolist(),
        scale=scale.tolist(),
        q01=q01.tolist(),
        q99=q99.tolist(),
        scale_floor=1e-6,
        clip=False,
    )


def normalize(values, stats):
    return ((np.asarray(values) - np.asarray(stats["center"])) / np.asarray(stats["scale"])).astype(np.float32)


def denormalize(values, stats):
    return np.asarray(values, dtype=np.float64) * np.asarray(stats["scale"]) + np.asarray(stats["center"])


def pose_errors(actual16, target16):
    a = np.asarray(actual16).reshape(*np.shape(actual16)[:-1], 2, 8)
    b = np.asarray(target16).reshape(*np.shape(target16)[:-1], 2, 8)
    position = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1)
    delta = quaternion_matrix(a[..., 3:7]) @ quaternion_matrix(b[..., 3:7]).swapaxes(-1, -2)
    skew = np.stack(
        (delta[..., 2, 1] - delta[..., 1, 2], delta[..., 0, 2] - delta[..., 2, 0], delta[..., 1, 0] - delta[..., 0, 1]),
        axis=-1,
    )
    angle = np.arctan2(np.linalg.norm(skew, axis=-1) / 2, (np.trace(delta, axis1=-2, axis2=-1) - 1) / 2)
    return position, angle
