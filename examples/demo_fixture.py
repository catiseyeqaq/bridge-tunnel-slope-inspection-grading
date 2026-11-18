"""Generate two synthetic images and small, synthetic head assets."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_demo(directory: Path):
    data_dir = directory / "data"
    input_dir = directory / "images"
    bridge_label, healthy = "裂缝", "完好"
    specs = [
        ("桥梁", "示例桥", "梁底示例.png", (195, 128, 70)),
        ("轨道", "", "轨道示例.png", (45, 105, 170)),
    ]
    metadata = []
    for category, bridge, filename, rgb in specs:
        path = input_dir / category / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (192, 128), rgb)
        draw = ImageDraw.Draw(image)
        draw.line([(20, 25), (87, 62), (135, 48), (172, 100)], fill=(30, 30, 30), width=4)
        image.save(path)
        metadata.append({
            "questionCategory": category, "bridgeName": bridge,
            "filename": filename, "imageRelativePath": path.relative_to(input_dir).as_posix(),
        })

    path = input_dir / "metadata.json"
    write_json(path, metadata)
    for name in ("rail_head", "bridge_all_view", "training_support", "ontology"):
        (data_dir / name).mkdir(parents=True)
    axis = np.eye(2, 2048, dtype=np.float32)
    zeros = np.zeros((1, 2048), dtype=np.float32)
    np.savez(
        data_dir / "rail_head/rail_head_v1.npz",
        coef=np.zeros_like(axis), intercept=[4.0, 0.0], classes=[healthy, bridge_label],
    )
    np.savez(
        data_dir / "bridge_all_view/head_all.npz",
        hd_coef=zeros, hd_intercept=[1.39], dis_vectors=axis[:1],
        dis_types=[bridge_label], dis_ratings=["2"],
        view_coef=zeros, view_intercept=[0.0], view_classes=["other"],
    )
    write_json(data_dir / "bridge_all_view/head_all.json", {
        "threshold": 0.8, "k": 1, "type_medoid": {bridge_label: "合成示例描述"},
        "healthy_medoid": "合成示例完好", "healthy_by_view": {"other": "合成示例完好"},
        "training_scope": "synthetic_fixture",
    })
    np.savez(data_dir / "training_support/support_bank.npz", vectors=axis, categories=["桥梁", "轨道"])
    write_json(data_dir / "training_support/support_bank.json", {
        "source_scope": "synthetic_fixture", "thresholds": {"桥梁": 0.5, "轨道": 0.5},
    })
    for branch in ("bridge", "rail"):
        write_json(data_dir / f"ontology/{branch}_allowed_labels.json", {
            "allowed_labels": [healthy, bridge_label],
        })
    return input_dir, path, data_dir


class PixelFixtureEncoder:
    """Return deterministic fixture vectors based on dominant image color."""

    def embed_image_paths(self, paths):
        vectors = np.zeros((len(paths), 2048), dtype=np.float32)
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB")).mean(axis=(0, 1))
            vectors[index, int(rgb[2] > rgb[0])] = 1.0
        return vectors
