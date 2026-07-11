"""Generate the audited Paper 1 Tier A three-module model YAMLs."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "26"
SCALES = {
    "n": [0.50, 0.25, 1024],
    "s": [0.50, 0.50, 1024],
    "m": [0.50, 1.00, 512],
    "l": [1.00, 1.00, 512],
    "x": [1.00, 1.50, 512],
}

BASELINE_LAYERS = {
    0: "p1",
    1: "p2_down",
    2: "backbone_p2",
    3: "p3_down",
    4: "backbone_p3",
    5: "p4_down",
    6: "backbone_p4",
    7: "p5_down",
    8: "backbone_p5",
    9: "sppf",
    10: "p5_context",
    13: "top_p4",
    16: "top_p3",
    17: "down_p3_p4",
    19: "detect_p4",
    20: "down_p4_p5",
    22: "detect_p5",
}

SPECS = [
    ("yolo26n-Paper1-TierA01-P2-SPDConv-EMA-P3f8.yaml", {"p2": True, "detail": "spd", "attention": "ema"}),
    ("yolo26n-Paper1-TierA02-P2-LaplacianConv-EMA-P3f8.yaml", {"p2": True, "detail": "lap", "attention": "ema"}),
    ("yolo26n-Paper1-TierA03-P2-EMA-P3f8-BiFPN.yaml", {"p2": True, "attention": "ema", "fusion": "bifpn"}),
    ("yolo26n-Paper1-TierA04-SPDConv-EMA-P3f8-FFAFusion.yaml", {"detail": "spd", "attention": "ema", "fusion": "ffa"}),
    ("yolo26n-Paper1-TierA05-LaplacianConv-EMA-P3f8-BiFPN.yaml", {"detail": "lap", "attention": "ema", "fusion": "bifpn"}),
    ("yolo26n-Paper1-TierA06-P2-SEAttention-P3-FFAFusion.yaml", {"p2": True, "attention": "se", "fusion": "ffa"}),
    ("yolo26n-Paper1-TierA07-P2-CBAM-P3-BiFPN.yaml", {"p2": True, "attention": "cbam", "fusion": "bifpn"}),
    ("yolo26n-Paper1-TierA08-P2-EMA-P3f8-slimneck.yaml", {"p2": True, "attention": "ema", "slim": True}),
    ("yolo26n-Paper1-TierA09-SPDConv-EMA-P3f8-slimneck.yaml", {"detail": "spd", "attention": "ema", "slim": True}),
    ("yolo26n-Paper1-TierA10-FDConv-EMA-P3f8-FFAFusion.yaml", {"detail": "fd", "attention": "ema", "fusion": "ffa"}),
    ("yolo26n-Paper1-TierA11-P2-LaplacianConv-CARAFE.yaml", {"p2": True, "detail": "lap", "carafe": True}),
    ("yolo26n-Paper1-TierA12-FDConv-GSConv-CARAFE.yaml", {"detail": "fd", "gs": True, "carafe": True}),
]


class Graph:
    def __init__(self):
        self.layers = []
        self.names = {}

    def add(self, name, source, repeats, module, args):
        def resolve(value):
            return self.names[value] if isinstance(value, str) else value

        resolved = [resolve(x) for x in source] if isinstance(source, list) else resolve(source)
        self.layers.append([resolved, repeats, module, args])
        self.names[name] = len(self.layers) - 1


def fusion_layer(mode, point):
    if mode == "bifpn":
        return "Concat_bifpn", [1]
    if mode == "ffa" and point == "fuse_p3_top":
        return "FFAFusionConcat", [1, 7, 32, 0.0]
    return "Concat", [1]


def add_fusion(graph, name, sources, mode):
    module, args = fusion_layer(mode, name)
    graph.add(name, sources, 1, module, args)


def pretrained_map(graph, has_p2):
    """Map baseline semantic layers to their target indices without accidental index collisions."""
    mapping = {f"model.{source}": f"model.{graph.names[name]}" for source, name in BASELINE_LAYERS.items()}
    detect = graph.names["detect"]
    if has_p2:
        for family in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            for source_level in range(3):
                mapping[f"model.23.{family}.{source_level}"] = f"model.{detect}.{family}.{source_level + 1}"
    else:
        mapping["model.23"] = f"model.{detect}"
    return mapping


def build_model(options):
    graph = Graph()
    graph.add("p1", -1, 1, "Conv", [64, 3, 2])
    graph.add("p2_down", -1, 1, "GSConv" if options.get("gs") else "Conv", [128, 3, 2])
    graph.add("backbone_p2", -1, 2, "C3k2", [256, False, 0.25])

    detail = options.get("detail")
    if detail == "spd":
        graph.add("p3_down", -1, 1, "SPDConv", [256])
    else:
        down_module = {"lap": "LaplacianConv", "fd": "FDConv"}.get(detail, "Conv")
        down_args = [256, 3, 2, 1] if detail == "fd" else [256, 3, 2]
        graph.add("p3_down", -1, 1, down_module, down_args)
    graph.add("backbone_p3", -1, 2, "C3k2", [512, False, 0.25])
    graph.add("p4_down", -1, 1, "Conv", [512, 3, 2])
    graph.add("backbone_p4", -1, 2, "C3k2", [512, True])
    graph.add("p5_down", -1, 1, "Conv", [1024, 3, 2])
    graph.add("backbone_p5", -1, 2, "C3k2", [1024, True])
    graph.add("sppf", -1, 1, "SPPF", [1024, 5, 3, True])
    graph.add("p5_context", -1, 2, "C2PSA", [1024])
    backbone_length = len(graph.layers)

    fusion = options.get("fusion")

    graph.add("up_p5_p4", -1, 1, "nn.Upsample", [None, 2, "nearest"])
    add_fusion(graph, "fuse_p4_top", ["up_p5_p4", "backbone_p4"], fusion)
    graph.add("top_p4", -1, 2, "C3k2", [512, True])

    if options.get("carafe"):
        graph.add("up_p4_p3", -1, 1, "CARAFE", [512, 3, 2])
    else:
        graph.add("up_p4_p3", -1, 1, "nn.Upsample", [None, 2, "nearest"])
    add_fusion(graph, "fuse_p3_top", ["up_p4_p3", "backbone_p3"], fusion)
    graph.add("top_p3", -1, 2, "C3k2", [256, True])

    if options.get("p2"):
        graph.add("up_p3_p2", -1, 1, "nn.Upsample", [None, 2, "nearest"])
        add_fusion(graph, "fuse_p2", ["up_p3_p2", "backbone_p2"], fusion)
        graph.add("detect_p2", -1, 2, "C3k2", [128, True])
        graph.add("down_p2_p3", -1, 1, "Conv", [128, 3, 2])
        add_fusion(graph, "fuse_p3_bottom", ["down_p2_p3", "top_p3"], fusion)
        graph.add("detect_p3", -1, 2, "C3k2", [256, True])
    else:
        graph.names["detect_p3"] = graph.names["top_p3"]

    graph.add("down_p3_p4", "detect_p3", 1, "Conv", [256, 3, 2])
    add_fusion(graph, "fuse_p4_bottom", ["down_p3_p4", "top_p4"], fusion)
    graph.add("detect_p4", -1, 2, "C3k2", [512, True])
    graph.add("down_p4_p5", -1, 1, "Conv", [512, 3, 2])
    add_fusion(graph, "fuse_p5_bottom", ["down_p4_p5", "p5_context"], fusion)
    if options.get("slim"):
        graph.add("detect_p5", -1, 1, "VoVGSCSP", [1024, 1, 0.5])
    else:
        graph.add("detect_p5", -1, 1, "C3k2", [1024, True, 0.5, True])

    attention = options.get("attention")
    if attention:
        module, args = {
            "ema": ("EMA_attention", [256, 8]),
            "se": ("SEAttention", [256]),
            "cbam": ("CBAM", [256]),
        }[attention]
        graph.add("detect_p3_attention", "detect_p3", 1, module, args)
        graph.names["detect_p3"] = graph.names["detect_p3_attention"]

    detect_sources = ["detect_p3", "detect_p4", "detect_p5"]
    if options.get("p2"):
        detect_sources.insert(0, "detect_p2")
    graph.add("detect", detect_sources, 1, "Detect", ["nc"])

    model = {
        "nc": 80,
        "end2end": True,
        "reg_max": 1,
        "scale": "n",
        "scales": SCALES,
        "backbone": graph.layers[:backbone_length],
        "head": graph.layers[backbone_length:],
    }
    model["pretrained_map"] = pretrained_map(graph, bool(options.get("p2")))
    return model


def validate(path, expect_p2):
    with path.open(encoding="utf-8") as handle:
        model = yaml.safe_load(handle)
    layers = model["backbone"] + model["head"]
    detect = layers[-1]
    assert detect[2] == "Detect"
    assert len(detect[0]) == (4 if expect_p2 else 3)
    assert all(0 <= index < len(layers) - 1 for index in detect[0])
    mapping = model.get("pretrained_map")
    assert isinstance(mapping, dict) and mapping
    assert len(mapping) == len(set(mapping)) == len(set(mapping.values()))


def main():
    assert len(SPECS) == 12 and len({name for name, _ in SPECS}) == 12
    for filename, options in SPECS:
        path = MODEL_DIR / filename
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("# Generated by scripts/generate_paper1_tiera_composites.py\n")
            handle.write("# Build-only candidate; train one YAML at a time after audit.\n\n")
            yaml.safe_dump(build_model(options), handle, sort_keys=False, allow_unicode=False)
        validate(path, bool(options.get("p2")))
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
