from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.asarray(
        [
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0.0, sin_alpha, cos_alpha, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def offline_dobot_fk_matrix(
    joint_rad: np.ndarray,
    *,
    tool_x_m: float = 0.0,
    tool_y_m: float = 0.0,
    tool_z_m: float = 0.197,
) -> np.ndarray:
    """Local Dobot FK, returning T_base_from_tcp in meters/radians."""
    q = np.asarray(joint_rad, dtype=np.float64).reshape(6)
    dh_params = [
        (q[0], 0.2234, 0.0, np.pi / 2.0),
        (q[1] - np.pi / 2.0, 0.0, -0.280, 0.0),
        (q[2], 0.0, -0.225, 0.0),
        (q[3] - np.pi / 2.0, 0.1175, 0.0, np.pi / 2.0),
        (q[4], 0.120, 0.0, -np.pi / 2.0),
        (q[5], 0.088, 0.0, 0.0),
    ]
    transform = np.eye(4, dtype=np.float64)
    for params in dh_params:
        transform = transform @ dh_transform(*params)

    tool = np.eye(4, dtype=np.float64)
    tool[:3, 3] = np.asarray([tool_x_m, tool_y_m, tool_z_m], dtype=np.float64)
    return transform @ tool


def pose6_to_matrix(pose6: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose6, dtype=np.float64).reshape(6)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_rotvec(pose[3:6]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose6(transform: np.ndarray) -> np.ndarray:
    mat = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return np.concatenate([mat[:3, 3], R.from_matrix(mat[:3, :3]).as_rotvec()]).astype(np.float64)


def rotation_vector_to_sixd(rot_vec: np.ndarray) -> np.ndarray:
    rot_mat = R.from_rotvec(np.asarray(rot_vec, dtype=np.float64).reshape(3)).as_matrix()
    return rot_mat[:, :2].flatten().astype(np.float64)


def sixd_to_rotation_vector(sixd: np.ndarray) -> np.ndarray:
    arr = np.asarray(sixd, dtype=np.float64).reshape(3, 2)
    a1, a2 = arr[:, 0], arr[:, 1]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-12:
        raise ValueError(f"Invalid 6D rotation first basis vector: {sixd!r}")
    b1 = a1 / n1
    b2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(b2)
    if n2 < 1e-12:
        raise ValueError(f"Invalid 6D rotation second basis vector: {sixd!r}")
    b2 = b2 / n2
    b3 = np.cross(b1, b2)
    return R.from_matrix(np.column_stack([b1, b2, b3])).as_rotvec().astype(np.float64)


def transform_pose6(pose6: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return matrix_to_pose6(np.asarray(transform, dtype=np.float64).reshape(4, 4) @ pose6_to_matrix(pose6))


def joint14_to_eef14(
    joint_vector: np.ndarray,
    t_world_from_left_base: np.ndarray | None = None,
    t_world_from_right_base: np.ndarray | None = None,
    *,
    tool_x_m: float = 0.0,
    tool_y_m: float = 0.0,
    tool_z_m: float = 0.197,
) -> np.ndarray:
    joints = np.asarray(joint_vector, dtype=np.float64).reshape(14)
    left_tf = np.eye(4, dtype=np.float64) if t_world_from_left_base is None else np.asarray(t_world_from_left_base, dtype=np.float64).reshape(4, 4)
    right_tf = np.eye(4, dtype=np.float64) if t_world_from_right_base is None else np.asarray(t_world_from_right_base, dtype=np.float64).reshape(4, 4)

    left_pose = matrix_to_pose6(
        left_tf
        @ offline_dobot_fk_matrix(
            joints[:6],
            tool_x_m=tool_x_m,
            tool_y_m=tool_y_m,
            tool_z_m=tool_z_m,
        )
    )
    right_pose = matrix_to_pose6(
        right_tf
        @ offline_dobot_fk_matrix(
            joints[7:13],
            tool_x_m=tool_x_m,
            tool_y_m=tool_y_m,
            tool_z_m=tool_z_m,
        )
    )
    return np.concatenate([left_pose, [joints[6]], right_pose, [joints[13]]]).astype(np.float32)


def eef_pose12_base_to_eef14(
    eef_pose_base: np.ndarray,
    gripper_source: np.ndarray,
    t_world_from_left_base: np.ndarray | None = None,
    t_world_from_right_base: np.ndarray | None = None,
) -> np.ndarray:
    eef_pose = np.asarray(eef_pose_base, dtype=np.float64).reshape(12)
    grippers = np.asarray(gripper_source, dtype=np.float64).reshape(14)
    left_tf = np.eye(4, dtype=np.float64) if t_world_from_left_base is None else np.asarray(t_world_from_left_base, dtype=np.float64).reshape(4, 4)
    right_tf = np.eye(4, dtype=np.float64) if t_world_from_right_base is None else np.asarray(t_world_from_right_base, dtype=np.float64).reshape(4, 4)
    left_pose = transform_pose6(eef_pose[:6], left_tf)
    right_pose = transform_pose6(eef_pose[6:12], right_tf)
    return np.concatenate([left_pose, [grippers[6]], right_pose, [grippers[13]]]).astype(np.float32)


def eef14_to_action20(eef14: np.ndarray) -> np.ndarray:
    eef = np.asarray(eef14, dtype=np.float64).reshape(14)
    return np.concatenate(
        [
            eef[:3],
            rotation_vector_to_sixd(eef[3:6]),
            [eef[6]],
            eef[7:10],
            rotation_vector_to_sixd(eef[10:13]),
            [eef[13]],
        ]
    ).astype(np.float32)


def action20_to_eef14(action20: np.ndarray) -> np.ndarray:
    action = np.asarray(action20, dtype=np.float64).reshape(20)
    return np.concatenate(
        [
            action[:3],
            sixd_to_rotation_vector(action[3:9]),
            [action[9]],
            action[10:13],
            sixd_to_rotation_vector(action[13:19]),
            [action[19]],
        ]
    ).astype(np.float32)


def eef14_world_to_base(
    eef14_world: np.ndarray,
    t_world_from_left_base: np.ndarray | None = None,
    t_world_from_right_base: np.ndarray | None = None,
) -> np.ndarray:
    eef = np.asarray(eef14_world, dtype=np.float64).reshape(14)
    left_tf = np.eye(4, dtype=np.float64) if t_world_from_left_base is None else np.asarray(t_world_from_left_base, dtype=np.float64).reshape(4, 4)
    right_tf = np.eye(4, dtype=np.float64) if t_world_from_right_base is None else np.asarray(t_world_from_right_base, dtype=np.float64).reshape(4, 4)
    t_left_base_from_world = np.linalg.inv(left_tf)
    t_right_base_from_world = np.linalg.inv(right_tf)
    left_pose = transform_pose6(eef[:6], t_left_base_from_world)
    right_pose = transform_pose6(eef[7:13], t_right_base_from_world)
    return np.concatenate([left_pose, [eef[6]], right_pose, [eef[13]]]).astype(np.float32)


def episode_eef_state_action_arrays(
    joint_vectors: np.ndarray,
    *,
    control_vectors: np.ndarray | None = None,
    eef_pose_base: np.ndarray | None = None,
    t_world_from_left_base: np.ndarray | None = None,
    t_world_from_right_base: np.ndarray | None = None,
    tool_x_m: float = 0.0,
    tool_y_m: float = 0.0,
    tool_z_m: float = 0.197,
) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(joint_vectors, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 14:
        raise ValueError(f"Expected joint_vectors shape (T,14), got {joints.shape}")
    if joints.shape[0] < 2:
        return np.zeros((0, 14), dtype=np.float32), np.zeros((0, 20), dtype=np.float32)
    controls = joints if control_vectors is None else np.asarray(control_vectors, dtype=np.float64)
    if controls.shape != joints.shape:
        raise ValueError(f"Expected control_vectors shape {joints.shape}, got {controls.shape}")

    if eef_pose_base is not None:
        measured_eef = np.asarray(eef_pose_base, dtype=np.float64)
        if measured_eef.shape != (joints.shape[0], 12):
            raise ValueError(f"Expected eef_pose_base shape ({joints.shape[0]},12), got {measured_eef.shape}")
        states = [
            eef_pose12_base_to_eef14(
                measured_eef[idx],
                joints[idx],
                t_world_from_left_base=t_world_from_left_base,
                t_world_from_right_base=t_world_from_right_base,
            )
            for idx in range(joints.shape[0] - 1)
        ]
        actions = [
            eef14_to_action20(
                eef_pose12_base_to_eef14(
                    measured_eef[idx],
                    controls[idx],
                    t_world_from_left_base=t_world_from_left_base,
                    t_world_from_right_base=t_world_from_right_base,
                )
            )
            for idx in range(1, measured_eef.shape[0])
        ]
        return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)

    states = [
        joint14_to_eef14(
            joints[idx],
            t_world_from_left_base=t_world_from_left_base,
            t_world_from_right_base=t_world_from_right_base,
            tool_x_m=tool_x_m,
            tool_y_m=tool_y_m,
            tool_z_m=tool_z_m,
        )
        for idx in range(joints.shape[0] - 1)
    ]
    actions = [
        eef14_to_action20(
            joint14_to_eef14(
                controls[idx],
                t_world_from_left_base=t_world_from_left_base,
                t_world_from_right_base=t_world_from_right_base,
                tool_x_m=tool_x_m,
                tool_y_m=tool_y_m,
                tool_z_m=tool_z_m,
            )
        )
        for idx in range(1, controls.shape[0])
    ]
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def _load_yaml_mapping(path: str | os.PathLike[str]) -> dict[str, Any]:
    import yaml

    resolved = _resolve_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {resolved}")
    data["_path"] = str(resolved)
    return data


def _matrix4(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4,4), got {arr.shape}")
    return arr


def _load_robot_camera_calibration(path: str | os.PathLike[str]) -> dict[str, Any]:
    data = _load_yaml_mapping(path)
    if "camera_label" not in data:
        raise ValueError(f"Robot-camera calibration missing camera_label: {path}")
    if "t_camera_from_base" not in data:
        raise ValueError(f"Robot-camera calibration missing t_camera_from_base: {path}")
    return data


def load_world_from_base_transforms(
    *,
    calibration_path: str | os.PathLike[str] | None = None,
    frame_mode: str = "workspace",
    left_robot_camera_calibration_path: str | os.PathLike[str] | None = None,
    right_robot_camera_calibration_path: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load T_target_from_left_base and T_target_from_right_base.

    Empty calibration paths intentionally return identity transforms so tests
    and simulator-only conversions can run without real calibration files.
    """
    if not calibration_path or not left_robot_camera_calibration_path or not right_robot_camera_calibration_path:
        return np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)

    from script.real_zed_collection.real_zed_utils import load_three_zed_calibration  # noqa: PLC0415

    target_frame = str(frame_mode)
    if target_frame not in {"reference_camera", "workspace", "left_base", "right_base"}:
        raise ValueError(
            "eef_frame_mode must be one of: reference_camera, workspace, left_base, right_base; "
            f"got {target_frame!r}"
        )
    camera_frame_mode = "workspace" if target_frame in {"left_base", "right_base"} else target_frame
    camera_calib = load_three_zed_calibration(_resolve_path(calibration_path), frame_mode=camera_frame_mode)

    def camera_frame_from_base(robot_camera_path: str | os.PathLike[str], side: str) -> np.ndarray:
        robot_camera = _load_robot_camera_calibration(robot_camera_path)
        camera_label = str(robot_camera["camera_label"])
        if camera_label not in camera_calib:
            raise ValueError(
                f"{side} robot-camera calibration camera_label={camera_label!r} "
                f"is missing from {calibration_path}"
            )
        t_camera_frame_from_camera = np.asarray(
            camera_calib[camera_label].t_world_from_cam,
            dtype=np.float64,
        ).reshape(4, 4)
        t_camera_from_base = _matrix4(robot_camera["t_camera_from_base"], f"{side}.t_camera_from_base")
        return t_camera_frame_from_camera @ t_camera_from_base

    t_camera_frame_from_left_base = camera_frame_from_base(left_robot_camera_calibration_path, "left")
    t_camera_frame_from_right_base = camera_frame_from_base(right_robot_camera_calibration_path, "right")

    if target_frame == "left_base":
        t_target_from_camera_frame = np.linalg.inv(t_camera_frame_from_left_base)
    elif target_frame == "right_base":
        t_target_from_camera_frame = np.linalg.inv(t_camera_frame_from_right_base)
    else:
        t_target_from_camera_frame = np.eye(4, dtype=np.float64)

    return (
        t_target_from_camera_frame @ t_camera_frame_from_left_base,
        t_target_from_camera_frame @ t_camera_frame_from_right_base,
    )


def add_eef_preprocess_args(parser) -> None:
    parser.add_argument("--action_mode", choices=["joint", "eef_absolute6d"], default="joint")
    parser.add_argument("--eef_calibration_path", default="")
    parser.add_argument(
        "--eef_frame_mode",
        choices=["reference_camera", "workspace", "left_base", "right_base"],
        default="right_base",
    )
    parser.add_argument("--left_robot_camera_calibration_path", default="")
    parser.add_argument("--right_robot_camera_calibration_path", default="")
    parser.add_argument("--eef_tool_x_m", type=float, default=0.0)
    parser.add_argument("--eef_tool_y_m", type=float, default=0.0)
    parser.add_argument("--eef_tool_z_m", type=float, default=0.197)


def validate_eef_dataset_frame(
    *,
    action_mode: str,
    eef_frame_mode: str,
    load_dir: str | os.PathLike[str],
) -> None:
    if str(action_mode) != "eef_absolute6d":
        return
    meta_path = Path(load_dir) / "real_zed_sam2_objpc_meta.json"
    if not meta_path.is_file():
        return
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    output_frame = str(meta.get("output_frame") or "").strip()
    if not output_frame:
        return
    expected_by_eef_frame = {
        "reference_camera": "source",
        "workspace": "workspace",
        "left_base": "left_base",
        "right_base": "right_base",
    }
    expected_output_frame = expected_by_eef_frame.get(str(eef_frame_mode))
    if expected_output_frame is None:
        raise ValueError(f"Unsupported eef_frame_mode={eef_frame_mode!r}")
    if output_frame != expected_output_frame:
        raise RuntimeError(
            "EEF absolute-6D preprocessing requires point clouds and EEF actions to use the same frame, "
            f"but {meta_path} has output_frame={output_frame} and eef_frame_mode={eef_frame_mode}. "
            f"Regenerate the real-ZED HDF5 data with --output_frame {expected_output_frame}, "
            "or rerun preprocessing with the matching --eef_frame_mode."
        )


def eef_arrays_for_episode(args: Any, episode: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    if str(getattr(args, "action_mode", "joint")) != "eef_absolute6d":
        return None
    t_world_from_left_base, t_world_from_right_base = load_world_from_base_transforms(
        calibration_path=getattr(args, "eef_calibration_path", ""),
        frame_mode=getattr(args, "eef_frame_mode", "right_base"),
        left_robot_camera_calibration_path=getattr(args, "left_robot_camera_calibration_path", ""),
        right_robot_camera_calibration_path=getattr(args, "right_robot_camera_calibration_path", ""),
    )
    return episode_eef_state_action_arrays(
        episode["vector"],
        control_vectors=episode.get("control"),
        eef_pose_base=episode.get("eef_pose_base"),
        t_world_from_left_base=t_world_from_left_base,
        t_world_from_right_base=t_world_from_right_base,
        tool_x_m=float(getattr(args, "eef_tool_x_m", 0.0)),
        tool_y_m=float(getattr(args, "eef_tool_y_m", 0.0)),
        tool_z_m=float(getattr(args, "eef_tool_z_m", 0.197)),
    )
