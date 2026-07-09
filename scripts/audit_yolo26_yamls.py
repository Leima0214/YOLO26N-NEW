#!/usr/bin/env python3
"""
audit_yolo26_yamls.py — Static audit of all YAML model configs in ultralytics/cfg/models/26/

Reads every .yaml, extracts structural info, detects modules, and outputs:
  docs/yolo26_yaml_inventory/
    yaml_file_inventory.csv
    yaml_structure_summary.csv
    module_usage_summary.csv
    paper1_candidate_recommendations.md

Read-only. No model construction. No file modification.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict, Counter

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
YAML_DIR = Path("ultralytics/cfg/models/26")
OUT_DIR = Path("docs/yolo26_yaml_inventory")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard Ultralytics modules — always available
STANDARD_MODULES = {
    "Conv", "C3k2", "C2PSA", "SPPF", "Concat", "Detect", "Segment", "Pose", "OBB",
    "Classify", "nn.Upsample", "Upsample",
}

# Custom modules that need ultralytics/nn/*.py or custom registration
CUSTOM_MODULES = {
    "BoTNet", "CBAM", "SEAttention", "EMA_attention", "SimA", "StokenAttention",
    "SwinTransformer", "BiForm", "C2f_Faster", "C3_Faster", "C2f_DySnakeConv",
    "ContextAggregation", "EfficientNetv2", "PatchExpand", "RepLKNet", "RepViT",
    "ShuffleNetV2", "vanillanet", "Moblileone", "RFAConv", "AKConv", "AConv",
    "ADown", "CARAFE", "DSConv", "DualConv", "DWConv", "GhostConv", "GSConv",
    "Involution", "OREPA", "RepConv", "SPDConv", "LDConv", "DASI", "MDCR",
    "MSFN", "slimneck", "BiFPN", "BiFPN1", "GOLDYOLO", "Glod", "4D", "2D",
    "HorBlock", "v10D",
}


def safe_load_yaml(path: Path) -> dict:
    """Load YAML, return None on failure."""
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        return {"_parse_error": str(e)}


def extract_modules(cfg: dict) -> list:
    """Extract all module names from backbone and head sections."""
    modules = set()
    for section in ["backbone", "head"]:
        layers = cfg.get(section, [])
        if isinstance(layers, list):
            for layer in layers:
                if isinstance(layer, list) and len(layer) >= 3:
                    mod_name = str(layer[2])
                    # Keep nn.Upsample as-is (it's in STANDARD_MODULES)
                    modules.add(mod_name)
    return sorted(modules)


def detect_detection_scales(head_layers: list) -> dict:
    """Identify which scales (P2-P6) are used for detection."""
    scales = {"P2": False, "P3": False, "P4": False, "P5": False, "P6": False}
    detect_indices = set()

    # Find Detect/Segment/Pose/OBB layer
    for i, layer in enumerate(head_layers):
        if isinstance(layer, list) and len(layer) >= 3:
            module = str(layer[2])
            if module in ("Detect", "Segment", "Pose", "OBB"):
                # First arg is list of input indices
                if isinstance(layer[0], list):
                    detect_indices = set(layer[0])

    # Now map indices to scales by scanning backbone for P-annotated conv layers
    # Backbone convs with stride=2 indicate scale changes:
    #   P1/2, P2/4, P3/8, P4/16, P5/32
    # We need to correlate detect input indices with backbone scale annotations.

    # Simplified: look at the concatenation points referenced by Detect inputs
    # In standard YOLO26: Detect input indices → {16:P3, 19:P4, 22:P5}
    # We check if there are out-of-range indices or 4 inputs
    num_detect_layers = len(detect_indices)

    if num_detect_layers >= 3:
        scales["P3"] = True
        scales["P4"] = True
        scales["P5"] = True
    elif num_detect_layers == 2:
        scales["P3"] = True
        scales["P4"] = True
    elif num_detect_layers == 1:
        scales["P3"] = True
    if num_detect_layers >= 4:
        scales["P2"] = True
    if num_detect_layers >= 5:
        scales["P6"] = True

    return {
        "num_detect_layers": num_detect_layers,
        "has_p2": scales["P2"],
        "has_p3": scales["P3"],
        "has_p4": scales["P4"],
        "has_p5": scales["P5"],
        "has_p6": scales["P6"],
    }


def detect_head_type(head_layers: list) -> str:
    """Detect output head type."""
    for layer in head_layers:
        if isinstance(layer, list) and len(layer) >= 3:
            mod = str(layer[2])
            if mod in ("Detect", "Segment", "Pose", "OBB", "Classify"):
                return mod
    return "unknown"


def count_layers(section_layers: list) -> int:
    """Count number of layers in a section."""
    if isinstance(section_layers, list):
        return len(section_layers)
    return 0


def main():
    yaml_files = sorted(YAML_DIR.glob("*.yaml"))
    print(f"Scanning {len(yaml_files)} YAML files in {YAML_DIR}")
    print()

    # ── Collect data ───────────────────────────────────────────────────────────
    inventory_rows = []
    structure_rows = []
    module_counter = Counter()
    module_locations = defaultdict(list)
    parse_errors = []
    parsed_ok = 0

    for yf in yaml_files:
        cfg = safe_load_yaml(yf)

        if "_parse_error" in cfg:
            parse_errors.append((yf.name, cfg["_parse_error"]))
            inventory_rows.append({
                "File": yf.name, "Path": str(yf.relative_to(Path.cwd())),
                "Size_Bytes": yf.stat().st_size, "Task_Type": "?",
                "Scale": "?", "Has_Scales_Field": "?", "nc": "?", "ch": "?",
                "Backbone_Layers": "?", "Head_Layers": "?", "Output_Head": "?",
                "Detect_Layers": "?", "Has_P2": "?", "Has_P3_P4_P5": "?",
                "Has_P6": "?", "Key_Modules": "?", "Custom_Modules": "?",
                "Parse_Status": "FAILED", "Notes": cfg["_parse_error"],
            })
            continue

        parsed_ok += 1
        bb = cfg.get("backbone", [])
        hd = cfg.get("head", [])
        scales_info = detect_detection_scales(hd)
        head_type = detect_head_type(hd)
        modules = extract_modules(cfg)
        custom_mods = [m for m in modules if m not in STANDARD_MODULES]
        standard_mods = [m for m in modules if m in STANDARD_MODULES]

        for m in modules:
            module_counter[m] += 1
            module_locations[m].append(yf.name)

        # Guess task type
        task = "detect"
        if head_type == "Segment":
            task = "segment"
        elif head_type == "Pose":
            task = "pose"
        elif head_type == "OBB":
            task = "obb"
        elif head_type == "Classify":
            task = "classify"
        elif head_type == "unknown":
            task = "?"  # some yamls might not have a head

        # Has scales?
        has_scales = "scales" in cfg

        inventory_rows.append({
            "File": yf.name,
            "Path": str(yf),
            "Size_Bytes": yf.stat().st_size,
            "Task_Type": task,
            "Scale": "n/s/m/l/x" if has_scales else "custom",
            "Has_Scales_Field": "yes" if has_scales else "no",
            "nc": cfg.get("nc", "?"),
            "ch": cfg.get("ch", "?"),
            "Backbone_Layers": count_layers(bb),
            "Head_Layers": count_layers(hd),
            "Output_Head": head_type,
            "Detect_Layers": scales_info["num_detect_layers"],
            "Has_P2": "yes" if scales_info["has_p2"] else "no",
            "Has_P3_P4_P5": "yes" if scales_info["has_p3"] else "no",
            "Has_P6": "yes" if scales_info["has_p6"] else "no",
            "Key_Modules": ", ".join(sorted(set(standard_mods + custom_mods))),
            "Custom_Modules": ", ".join(custom_mods) if custom_mods else "none",
            "Parse_Status": "OK",
            "Notes": "",
        })

        structure_rows.append({
            "File": yf.name,
            "Backbone_Summary": f"{count_layers(bb)} layers",
            "Head_Summary": f"{count_layers(hd)} layers, {head_type} head",
            "Detection_Scales": f"P3,P4,P5" if scales_info["has_p3"] else
                                f"{scales_info['num_detect_layers']} detection layers",
            "Small_Object_Relevance": "high" if scales_info["has_p2"] else
                                      "medium" if scales_info["has_p3"] else "low",
            "Lightweight_Relevance": "high" if has_scales and "n" in str(cfg.get("scales", {}))
                                     else "medium",
            "Paper1_Relevance": "high" if (has_scales and not custom_mods) else
                                "medium" if custom_mods else "low",
            "Risk": "low" if not custom_mods else
                    "medium" if len(custom_mods) <= 1 else "high",
        })

    # ── Write CSV files ────────────────────────────────────────────────────────
    def write_csv(filename, rows, fieldnames):
        path = OUT_DIR / filename
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  Wrote {path} ({len(rows)} rows)")

    inv_fields = ["File", "Path", "Size_Bytes", "Task_Type", "Scale",
                  "Has_Scales_Field", "nc", "ch", "Backbone_Layers", "Head_Layers",
                  "Output_Head", "Detect_Layers", "Has_P2", "Has_P3_P4_P5",
                  "Has_P6", "Key_Modules", "Custom_Modules", "Parse_Status", "Notes"]
    write_csv("yaml_file_inventory.csv", inventory_rows, inv_fields)

    str_fields = ["File", "Backbone_Summary", "Head_Summary", "Detection_Scales",
                  "Small_Object_Relevance", "Lightweight_Relevance",
                  "Paper1_Relevance", "Risk"]
    write_csv("yaml_structure_summary.csv", structure_rows, str_fields)

    # Module usage CSV
    mod_rows = []
    for mod, count in module_counter.most_common():
        is_custom = mod not in STANDARD_MODULES
        mod_rows.append({
            "Module": mod,
            "Count": count,
            "Appears_In": ", ".join(module_locations[mod][:5]) +
                          ("..." if len(module_locations[mod]) > 5 else ""),
            "Role": "custom" if is_custom else "standard",
            "Need_Custom_Code": "yes" if is_custom else "no",
            "Risk": "high" if is_custom else "low",
        })
    write_csv("module_usage_summary.csv", mod_rows,
              ["Module", "Count", "Appears_In", "Role", "Need_Custom_Code", "Risk"])

    # ── Write Markdown reports ─────────────────────────────────────────────────

    # --- paper1_candidate_recommendations.md ---
    rec_lines = [
        "# Paper 1 Candidate Recommendations",
        "",
        f"Generated from {len(yaml_files)} YAML files. {parsed_ok} parsed OK, {len(parse_errors)} failed.",
        "",
        "## 1. Clean Baselines (recommended for formal comparison)",
        "",
        "| YAML | Detect Layers | Scales | Custom Modules | Risk |",
        "| --- | ---: | --- | --- | --- |",
    ]
    clean_baselines = [r for r in inventory_rows
                       if r["Custom_Modules"] == "none" and r["Has_Scales_Field"] == "yes"
                       and r["Parse_Status"] == "OK"]
    for r in clean_baselines:
        rec_lines.append(f"| {r['File']} | {r['Detect_Layers']} | {r['Scale']} | none | low |")

    rec_lines.extend([
        "",
        "## 2. P2 / Small Object Candidates",
        "",
        "| YAML | Detect Layers | P2 | Custom Modules | Risk |",
        "| --- | ---: | --- | --- | --- |",
    ])
    p2_candidates = [r for r in inventory_rows
                     if r["Has_P2"] == "yes" and r["Parse_Status"] == "OK"]
    if not p2_candidates:
        rec_lines.append("| — | — | — | No P2 detection layers found in any YAML. | — |")
    for r in p2_candidates:
        rec_lines.append(f"| {r['File']} | {r['Detect_Layers']} | {r['Has_P2']} | {r['Custom_Modules']} | {r['Notes']} |")

    rec_lines.extend([
        "",
        "## 3. Attention Module Candidates",
        "",
        "| YAML | Attention Module | Other Custom | Risk |",
        "| --- | --- | --- | --- |",
    ])
    attn_yamls = [r for r in inventory_rows
                  if any(a in r["Custom_Modules"]
                         for a in ["CBAM", "SEAttention", "EMA_attention", "SimA",
                                   "StokenAttention", "BiForm"])]
    for r in attn_yamls:
        rec_lines.append(f"| {r['File']} | {r['Custom_Modules']} | {r['Detect_Layers']} heads | medium |")

    rec_lines.extend([
        "",
        "## 4. Not Recommended for Formal Baseline",
        "",
        "| YAML | Reason |",
        "| --- | --- |",
    ])
    not_rec = [r for r in inventory_rows
               if r["Custom_Modules"] != "none" or r["Has_Scales_Field"] == "no"
               or r["Parse_Status"] != "OK"]
    for r in not_rec:
        reason = []
        if r["Parse_Status"] != "OK":
            reason.append("parse failed")
        if r["Custom_Modules"] != "none":
            reason.append(f"custom modules: {r['Custom_Modules']}")
        if r["Has_Scales_Field"] == "no":
            reason.append("no scales field (cannot use n/s/m/l/x)")
        rec_lines.append(f"| {r['File']} | {'; '.join(reason)} |")

    rec_lines.extend([
        "",
        "## 5. Top 3 Recommended YAMLs for Paper 1",
        "",
        "1. **yolo26.yaml** — Official clean baseline. P3/P4/P5, scales n/s/m/l/x, zero custom modules.",
    ])
    # Find the best candidates
    if clean_baselines:
        rec_lines.append(f"2. **{clean_baselines[0]['File']}** — Clean variant, standard structure, no custom code needed.")
        if len(clean_baselines) > 1:
            rec_lines.append(f"3. **{clean_baselines[1]['File']}** — Another clean baseline option.")
    rec_lines.append("")

    with open(OUT_DIR / "paper1_candidate_recommendations.md", "w") as f:
        f.write("\n".join(rec_lines))

    # --- yaml_risk_audit.md ---
    risk_lines = [
        "# YAML Risk Audit",
        "",
        "| YAML | Parse OK | Custom Modules | Scales | Num Detect | Can Parse | Fair Compare | Notes |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for r in inventory_rows:
        parse_ok = r["Parse_Status"] == "OK"
        custom = r["Custom_Modules"] if r["Custom_Modules"] != "none" else "none"
        can_parse = "✅" if parse_ok else "❌"
        fair = "✅" if (parse_ok and r["Has_Scales_Field"] == "yes" and custom == "none") else "⚠️"
        risk_lines.append(
            f"| {r['File']} | {can_parse} | {custom} | "
            f"{r['Has_Scales_Field']} | {r['Detect_Layers']} | "
            f"{can_parse} | {fair} | {r['Notes']} |"
        )

    if parse_errors:
        risk_lines.append("")
        risk_lines.append("## Parse Failures")
        for name, err in parse_errors:
            risk_lines.append(f"- **{name}**: {err}")

    risk_lines.append("")
    risk_lines.append("## Key Risks")
    risk_lines.append(f"- **{len([r for r in inventory_rows if r['Custom_Modules'] != 'none'])}** YAMLs require custom Python modules")
    risk_lines.append(f"- **{len([r for r in inventory_rows if r['Has_Scales_Field'] == 'no'])}** YAMLs have no scales field (cannot use n/s/m/l/x)")
    risk_lines.append(f"- **{len(parse_errors)}** YAMLs failed to parse")

    with open(OUT_DIR / "yaml_risk_audit.md", "w") as f:
        f.write("\n".join(risk_lines))

    # --- parsing_notes.md ---
    notes_lines = [
        "# Parsing Notes",
        "",
        f"**Method**: `yaml.safe_load()` on each file, then static structural inspection.",
        f"**Total files**: {len(yaml_files)}",
        f"**Parsed OK**: {parsed_ok}",
        f"**Parse failures**: {len(parse_errors)}",
        "",
    ]
    if parse_errors:
        notes_lines.append("## Failures")
        for name, err in parse_errors:
            notes_lines.append(f"- `{name}`: {err}")
    else:
        notes_lines.append("All YAMLs parsed successfully.")

    notes_lines.extend([
        "",
        "## Detection scale mapping method",
        "",
        "P2-P6 detection scales are inferred from the number of detection layer inputs.",
        "This is a heuristic — actual scale labels come from backbone Conv stride comments.",
        "Manual verification recommended for any YAML with >3 or <3 detection layers.",
        "",
        "## Module classification",
        "",
        f"Standard modules: {sorted(STANDARD_MODULES)}",
        "",
        "All other modules are flagged as custom and may require additional Python files in `ultralytics/nn/`.",
        "",
        "## Unconfirmed items",
        "",
        "- Whether custom modules are actually importable at runtime",
        "- Whether custom modules change FLOPs/params in ways that affect fair comparison",
        "- Whether any YAML has hidden dependencies not captured by static parsing",
    ])

    with open(OUT_DIR / "parsing_notes.md", "w") as f:
        f.write("\n".join(notes_lines))

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Total YAMLs:        {len(yaml_files)}")
    print(f"  Parsed OK:          {parsed_ok}")
    print(f"  Parse failures:     {len(parse_errors)}")
    print(f"  Unique modules:     {len(module_counter)}")
    print(f"  Custom modules:     {sum(1 for m in module_counter if m not in STANDARD_MODULES)}")
    print(f"  Clean baselines:    {len(clean_baselines)}")
    print(f"  With custom code:   {len([r for r in inventory_rows if r['Custom_Modules'] != 'none'])}")
    print(f"  P2 detected:        {len(p2_candidates)}")
    print(f"  Output directory:   {OUT_DIR.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
