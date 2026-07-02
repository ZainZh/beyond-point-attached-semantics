from __future__ import annotations

import argparse
import colorsys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from my_datasets.partnext_canonical_field import (
    IGNORE_LABEL,
    _build_global_face_labels,
    _concatenate_meshes,
    _load_geometry_meshes,
    _sample_surface_points,
)
from my_datasets.partnext_universal_field import (
    PartNextUniversalFieldDataset,
    _normalize_multiple_arrays,
)


DEFAULT_DATASET_ROOT = Path("data/PartNext_mesh")
DEFAULT_SUPPORT_COLOR = np.asarray([168, 168, 168], dtype=np.uint8)
DEFAULT_IGNORE_COLOR = np.asarray([110, 110, 110], dtype=np.uint8)


@dataclass(slots=True)
class PreparedSample:
    category: str
    model_id: str
    mesh_path: str
    support_points_raw: np.ndarray
    support_points_normalized: np.ndarray
    query_points_raw: np.ndarray
    query_points_normalized: np.ndarray
    query_labels: np.ndarray
    canonical_label_names: list[str]


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize PartNext support points and semantic query points using the same "
            "sampling logic as universal-field training."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--category", type=str, default="Computer_Mouse")
    parser.add_argument("--sample-index", type=int, default=14)
    parser.add_argument("--mesh-path", type=Path, default=None)
    parser.add_argument("--alias-config", type=Path, default=Path("configs/computer_mouse.json"))
    # parser.add_argument("--alias-config", type=Path, default=None)
    parser.add_argument(
        "--label-level",
        type=str,
        choices=["leaf", "parent", "path", "config"],
        default="config",
    )
    parser.add_argument("--ignore-labels", nargs="*", default=["Other"])
    parser.add_argument("--num-support-points", type=int, default=5000)
    parser.add_argument("--num-query-points", type=int, default=5000)
    parser.add_argument("--balanced-sampling-ratio", type=float, default=0.8)
    parser.add_argument("--query-sampling-ratio", type=float, default=0.75)
    parser.add_argument("--coord-scale", type=float, default=1.0)
    parser.add_argument(
        "--disable-z-shift",
        action="store_true",
        help="Disable the object-centric z-shift used in training normalization.",
    )
    parser.add_argument("--backend", choices=["html", "open3d"], default="html")
    parser.add_argument("--coord-frame", choices=["raw", "normalized"], default="normalized")
    parser.add_argument("--show", choices=["support", "query", "both"], default="both")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations_partnext_support_query"),
    )
    parser.add_argument("--output-name", type=str, default=None)
    return parser


def _build_palette(num_classes: int) -> np.ndarray:
    colors: list[list[int]] = []
    for index in range(max(num_classes, 1)):
        hue = (index * 0.6180339887498949) % 1.0
        saturation = 0.55 + 0.25 * ((index % 3) / 2.0)
        value = 0.80 + 0.15 * ((index % 4) / 3.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([int(channel * 255) for channel in rgb])
    return np.asarray(colors, dtype=np.uint8)


def _labels_to_colors(
    labels: np.ndarray,
    palette: np.ndarray,
    ignore_color: np.ndarray = DEFAULT_IGNORE_COLOR,
) -> np.ndarray:
    colors = np.repeat(ignore_color[None, :], labels.shape[0], axis=0)
    valid_mask = (labels >= 0) & (labels < len(palette))
    colors[valid_mask] = palette[labels[valid_mask]]
    return colors.astype(np.uint8, copy=False)


def _rgb_strings(colors: np.ndarray) -> list[str]:
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors]


def _visibility_mask(preset: str) -> list[bool]:
    presets = {
        "raw_both": [True, True, False, False],
        "raw_support": [True, False, False, False],
        "raw_query": [False, True, False, False],
        "normalized_both": [False, False, True, True],
        "normalized_support": [False, False, True, False],
        "normalized_query": [False, False, False, True],
    }
    return presets[preset]


def _camera_eye(x: float, y: float, z: float) -> dict[str, dict[str, float]]:
    return {"eye": {"x": x, "y": y, "z": z}}


def _camera_presets() -> list[tuple[str, dict[str, dict[str, float]]]]:
    return [
        ("Front", _camera_eye(0.0, -2.2, 0.15)),
        ("Back", _camera_eye(0.0, 2.2, 0.15)),
        ("Left", _camera_eye(-2.2, 0.0, 0.15)),
        ("Right", _camera_eye(2.2, 0.0, 0.15)),
        ("Top", _camera_eye(0.0, 0.0, 2.6)),
        ("Iso", _camera_eye(1.6, -1.6, 1.25)),
    ]


def _scene_axis_visibility(visible: bool) -> dict[str, bool]:
    return {
        "scene.xaxis.visible": visible,
        "scene.yaxis.visible": visible,
        "scene.zaxis.visible": visible,
    }


def build_plotly_figure(bundle: PreparedSample):
    try:
        import plotly.graph_objects as go
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Plotly is required for the HTML backend.") from error

    palette = _build_palette(len(bundle.canonical_label_names))
    query_colors = _labels_to_colors(bundle.query_labels, palette)
    support_colors = np.repeat(DEFAULT_SUPPORT_COLOR[None, :], bundle.support_points_raw.shape[0], axis=0)

    traces = [
        go.Scatter3d(
            x=bundle.support_points_raw[:, 0],
            y=bundle.support_points_raw[:, 1],
            z=bundle.support_points_raw[:, 2],
            mode="markers",
            name="support_raw",
            marker={"size": 2.8, "color": _rgb_strings(support_colors), "opacity": 0.9},
            visible=False,
        ),
        go.Scatter3d(
            x=bundle.query_points_raw[:, 0],
            y=bundle.query_points_raw[:, 1],
            z=bundle.query_points_raw[:, 2],
            mode="markers",
            name="query_raw",
            marker={"size": 2.6, "color": _rgb_strings(query_colors), "opacity": 0.95},
            visible=False,
        ),
        go.Scatter3d(
            x=bundle.support_points_normalized[:, 0],
            y=bundle.support_points_normalized[:, 1],
            z=bundle.support_points_normalized[:, 2],
            mode="markers",
            name="support_normalized",
            marker={"size": 2.8, "color": _rgb_strings(support_colors), "opacity": 0.9},
            visible=True,
        ),
        go.Scatter3d(
            x=bundle.query_points_normalized[:, 0],
            y=bundle.query_points_normalized[:, 1],
            z=bundle.query_points_normalized[:, 2],
            mode="markers",
            name="query_normalized",
            marker={"size": 2.6, "color": _rgb_strings(query_colors), "opacity": 0.95},
            visible=True,
        ),
    ]

    buttons = []
    for label, preset in [
        ("Raw / Both", "raw_both"),
        ("Raw / Support", "raw_support"),
        ("Raw / Query", "raw_query"),
        ("Normalized / Both", "normalized_both"),
        ("Normalized / Support", "normalized_support"),
        ("Normalized / Query", "normalized_query"),
    ]:
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": _visibility_mask(preset)},
                    {"title": f"{bundle.category} / {bundle.model_id} / {label}"},
                ],
            }
        )

    camera_buttons = [
        {
            "label": label,
            "method": "relayout",
            "args": [{"scene.camera": camera}],
        }
        for label, camera in _camera_presets()
    ]
    axes_buttons = [
        {
            "label": "Axes On",
            "method": "relayout",
            "args": [_scene_axis_visibility(True)],
        },
        {
            "label": "Axes Off",
            "method": "relayout",
            "args": [_scene_axis_visibility(False)],
        },
    ]

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=f"{bundle.category} / {bundle.model_id} / Normalized / Both",
        template="plotly_white",
        showlegend=True,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
            "xaxis": {"visible": True},
            "yaxis": {"visible": True},
            "zaxis": {"visible": True},
            "camera": _camera_presets()[-1][1],
        },
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        updatemenus=[
            {
                "type": "dropdown",
                "x": 0.01,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": buttons,
            },
            {
                "type": "dropdown",
                "x": 0.33,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": camera_buttons,
            },
            {
                "type": "dropdown",
                "x": 0.56,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": axes_buttons,
            },
        ],
    )
    return figure


def write_plotly_html(bundle: PreparedSample, output_path: Path) -> Path:
    figure = build_plotly_figure(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return output_path


def copy_source_mesh(bundle: PreparedSample, output_dir: str | Path) -> Path:
    source_path = Path(bundle.mesh_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source mesh does not exist: {source_path}")
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_path = target_dir / source_path.name
    shutil.copy2(source_path, copied_path)
    return copied_path


def _points_and_colors_for_view(
    bundle: PreparedSample,
    coord_frame: str,
    show: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if coord_frame == "raw":
        support_points = bundle.support_points_raw
        query_points = bundle.query_points_raw
    else:
        support_points = bundle.support_points_normalized
        query_points = bundle.query_points_normalized

    support_colors = np.repeat(DEFAULT_SUPPORT_COLOR[None, :], support_points.shape[0], axis=0)
    query_colors = _labels_to_colors(bundle.query_labels, _build_palette(len(bundle.canonical_label_names)))

    groups: list[tuple[np.ndarray, np.ndarray]] = []
    if show in {"support", "both"}:
        groups.append((support_points, support_colors))
    if show in {"query", "both"}:
        groups.append((query_points, query_colors))
    return groups


def show_open3d(bundle: PreparedSample, coord_frame: str, show: str) -> None:
    try:
        import open3d as o3d
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Open3D is required for the open3d backend.") from error

    geometries = []
    for points, colors in _points_and_colors_for_view(bundle, coord_frame=coord_frame, show=show):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector((colors.astype(np.float64) / 255.0))
        geometries.append(cloud)

    if not geometries:
        raise ValueError("No geometries selected for visualization.")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{bundle.category} / {bundle.model_id} / {coord_frame} / {show}",
    )


def _build_dataset(args: argparse.Namespace) -> PartNextUniversalFieldDataset:
    categories: Iterable[str] | None = [args.category] if args.category else None
    return PartNextUniversalFieldDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        categories=categories,
        num_support_points=args.num_support_points,
        num_query_surface_points=args.num_query_points,
        num_query_occ_points=32,
        num_query_probe_points=32,
        balanced_sampling_ratio=args.balanced_sampling_ratio,
        query_sampling_ratio=args.query_sampling_ratio,
        coord_scale=args.coord_scale,
        center_shift_z=not args.disable_z_shift,
        label_level=args.label_level,
        alias_config_path=args.alias_config,
        ignore_labels=args.ignore_labels,
        rotation_mode="none",
        jitter_std=0.0,
        second_view=False,
        max_retries=1,
        val_ratio=0.1,
        test_ratio=0.0,
    )


def _find_record_by_mesh_path(dataset: PartNextUniversalFieldDataset, mesh_path: Path):
    requested = mesh_path.resolve()
    for record in dataset.records:
        if Path(record.mesh_path).resolve() == requested:
            return record
    raise ValueError(f"Could not find mesh path in dataset split: {requested}")


def _select_record(dataset: PartNextUniversalFieldDataset, args: argparse.Namespace):
    if args.mesh_path is not None:
        return _find_record_by_mesh_path(dataset, args.mesh_path)
    if args.category is None:
        raise ValueError("Please provide --category when selecting a sample by index.")
    if not (0 <= args.sample_index < len(dataset.records)):
        raise IndexError(
            f"sample-index {args.sample_index} is out of range for dataset size {len(dataset.records)}"
        )
    return dataset.records[args.sample_index]


def prepare_sample(dataset: PartNextUniversalFieldDataset, record) -> PreparedSample:
    meshes = _load_geometry_meshes(record.mesh_path)
    mesh = _concatenate_meshes(meshes)
    face_labels = _build_global_face_labels(
        record=record,
        num_meshes=len(meshes),
        mesh_face_counts=[len(item.faces) for item in meshes],
        total_faces=len(mesh.faces),
        canonical_label_to_id=dataset.canonical_label_to_id,
        label_level=dataset.label_level,
        alias_lookup=dataset.alias_lookup,
        ignore_names=dataset.ignore_names,
    )

    support_points_raw, _, _, _ = _sample_surface_points(
        mesh=mesh,
        face_labels=face_labels,
        num_points=dataset.num_support_points,
        balanced_sampling_ratio=dataset.balanced_sampling_ratio,
    )
    query_points_raw, _, _, query_labels = _sample_surface_points(
        mesh=mesh,
        face_labels=face_labels,
        num_points=dataset.num_query_surface_points,
        balanced_sampling_ratio=dataset.query_sampling_ratio,
    )

    support_points_normalized, query_points_normalized = _normalize_multiple_arrays(
        support_points_raw,
        query_points_raw,
        coord_scale=dataset.coord_scale,
        center_shift_z=dataset.center_shift_z,
    )

    return PreparedSample(
        category=record.category,
        model_id=record.model_id,
        mesh_path=str(record.mesh_path),
        support_points_raw=support_points_raw,
        support_points_normalized=support_points_normalized,
        query_points_raw=query_points_raw,
        query_points_normalized=query_points_normalized,
        query_labels=query_labels,
        canonical_label_names=list(dataset.canonical_label_names),
    )


def _default_output_name(bundle: PreparedSample) -> str:
    return f"{bundle.category}_{bundle.model_id}_support_query.html"


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    dataset = _build_dataset(args)
    record = _select_record(dataset, args)
    bundle = prepare_sample(dataset, record)

    if args.backend == "html":
        output_name = args.output_name or _default_output_name(bundle)
        output_path = args.output_dir / output_name
        write_plotly_html(bundle, output_path)
        copied_mesh_path = copy_source_mesh(bundle, args.output_dir)
        print(f"[visualize] wrote HTML: {output_path.resolve()}")
        print(f"[visualize] copied source mesh: {copied_mesh_path.resolve()}")
        return

    show_open3d(bundle, coord_frame=args.coord_frame, show=args.show)


if __name__ == "__main__":
    main()
