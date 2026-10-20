import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .segment_drawing import build_colors, draw_segment_map, get_drawable_cache, get_records, get_segment_order, load_bundle
from .unit_plotting import attribution_axis_label


def contribution_note(prop=None, task=""):
    return "+: increases predicted output\n−: decreases predicted output"

def safe_name(value, max_length=90):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return (text or "item")[:max_length]


def parse_filter(text):
    if not text or str(text).strip().lower() in {"all", "*"}: return None
    return [x.strip() for x in str(text).split(",") if x.strip()]


def as_float(value):
    try:
        if pd.isna(value): return np.nan
        return float(value)
    except Exception:
        return np.nan


def threshold_rank(output_name, output_value):
    value = as_float(output_value)
    if np.isfinite(value): return value
    text = str(output_name).lower().replace("fu_ge", "ge")
    match = re.search(r"ge([0-9]+(?:p[0-9]+)?(?:\.[0-9]+)?)(?:h)?", text)
    if match: return float(match.group(1).replace("p", "."))
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else np.nan


def output_from_base(base):
    if str(base).startswith("prediction_"): return str(base).removeprefix("prediction_")
    if str(base).startswith("prob_"): return str(base).removeprefix("prob_")
    return str(base)


def metric_bases(row):
    keys = set(row.index)
    suffixes = ("_std", "_min", "_max", "_n_models", "_raw_mean", "_lower_1sd", "_upper_1sd")
    bases = []
    for col in row.index:
        if not (str(col).startswith("prediction_") or str(col).startswith("prob_")): continue
        if str(col).endswith(suffixes): continue
        output = output_from_base(col)
        if str(col).startswith("prob_") and "prediction_" + output in keys: continue
        if not np.isfinite(as_float(row.get(col))): continue
        bases.append(col)
    return bases


def predicted_class(row, output, mean, cutoff):
    for col in ("class_" + output, "pred_" + output):
        value = row.get(col)
        if pd.isna(value): continue
        text = str(value).strip()
        if text: return text in {"1", "1.0", "True", "true", "TRUE"}
    return mean >= cutoff


def find_prediction_row(predictions, keys, key_row):
    if predictions is None or predictions.empty: return None
    usable = [x for x in keys if x in predictions.columns and x in key_row.index]
    if not usable: return None
    mask = np.ones(len(predictions), dtype=bool)
    for column in usable:
        value = key_row[column]
        if pd.isna(value):
            mask &= predictions[column].isna().to_numpy()
        else:
            mask &= predictions[column].astype(str).eq(str(value)).to_numpy()
    matched = predictions.loc[mask]
    return matched.iloc[0] if not matched.empty else None


def compact_outputs_from_predictions(molecule_data, outputs, cutoff, prediction_row):
    if prediction_row is None: return compact_outputs(molecule_data, outputs, cutoff)
    available = set(molecule_data["output_name"].dropna().astype(str))
    ordered = [x for x in outputs if str(x).startswith("species_threshold_")]
    rows = []
    local_threshold_i = 0

    for base in metric_bases(prediction_row):
        output = output_from_base(base)
        mean = as_float(prediction_row.get(base))
        rank = threshold_rank(output, prediction_row.get("threshold_" + output))
        if not np.isfinite(rank): continue

        plot_output = output if output in available else None
        if plot_output is None and local_threshold_i < len(ordered):
            plot_output = ordered[local_threshold_i]
        local_threshold_i += 1
        if plot_output not in available: continue

        rows.append({
            "name": plot_output,
            "rank": rank,
            "positive": predicted_class(prediction_row, output, mean, cutoff),
        })

    if not rows: return compact_outputs(molecule_data, outputs, cutoff)
    positives = [x for x in rows if x["positive"]]
    chosen = max(positives, key=lambda x: x["rank"]) if positives else min(rows, key=lambda x: x["rank"])
    return [chosen["name"]]


def compact_outputs(molecule_data, outputs, cutoff):
    rows = []
    for output_name in outputs:
        subset = molecule_data[molecule_data["output_name"].astype(str) == str(output_name)]
        if subset.empty: continue
        first = subset.iloc[0]
        rows.append({
            "name": output_name,
            "rank": threshold_rank(output_name, first.get("output_value")),
            "mean": as_float(first.get("original_prediction_mean")),
        })

    ranked = [x for x in rows if np.isfinite(x["rank"]) and np.isfinite(x["mean"])]
    if len(ranked) <= 1: return [x["name"] for x in rows]
    positives = [x for x in ranked if x["mean"] >= cutoff]
    chosen = max(positives, key=lambda x: x["rank"]) if positives else min(ranked, key=lambda x: x["rank"])
    return [chosen["name"]]


def task_key(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def filter_tasks(data, tasks, name):
    if tasks is None: return data
    if "task" not in data.columns: raise KeyError(f"{name} CSV has no task column")
    requested = {task_key(x) for x in tasks}
    output = data[data["task"].map(task_key).isin(requested)].copy()
    if output.empty:
        available = data["task"].dropna().astype(str).drop_duplicates().tolist()
        raise ValueError(f"No rows found for tasks={tasks}; available={available}")
    return output


def normalize_columns(data):
    aliases = {
        "threshold_name": "output_name",
        "original_probability_mean": "original_prediction_mean",
        "original_probability_std": "original_prediction_std",
        "contribution_probability_mean": "contribution_mean",
        "contribution_probability_std": "contribution_std",
    }
    data = data.copy()
    for old, new in aliases.items():
        if new not in data.columns and old in data.columns: data[new] = data[old]
    return data


def select_molecules(data, top_n, select_by):
    keys = [x for x in ["row_id", "mol_id", "task"] if x in data.columns]
    rows = []
    for order, (key, subset) in enumerate(data.groupby(keys, sort=False, dropna=False)):
        key = key if isinstance(key, tuple) else (key,)
        values = pd.to_numeric(subset["contribution_mean"], errors="coerce").to_numpy(float)
        max_abs = np.nanmax(np.abs(values)) if np.isfinite(values).any() else np.nan
        rows.append({**dict(zip(keys, key)), "_order": order, "_max_abs": max_abs})
    selected = pd.DataFrame(rows)
    if select_by == "max_abs":
        selected = selected.sort_values(["_max_abs", "_order"], ascending=[False, True])
    else:
        selected = selected.sort_values("_order")
    if top_n > 0:
        selected = selected.head(top_n)
    return keys, selected


def subset_for_key(data, keys, row):
    mask = np.ones(len(data), dtype=bool)
    for column in keys:
        value = row[column]
        mask &= data[column].isna().to_numpy() if pd.isna(value) else data[column].astype(str).eq(str(value)).to_numpy()
    return data.loc[mask].copy()


def create_contribution_figure(n_segments):
    height = 4.0 if n_segments <= 12 else max(4.8, 0.30 * n_segments + 1.3)
    figure, axis = plt.subplots(figsize=(10.5, height))
    return figure, axis


def clean_order(values, n_records):
    out, used = [], set()
    for value in values or []:
        try:
            value = int(value)
        except Exception:
            continue
        if 0 <= value < n_records and value not in used:
            out.append(value)
            used.add(value)
    return out


def seq_entry_len(entry):
    if not isinstance(entry, dict): return None
    for key in ["segment_ids", "segment_idx"]:
        value = entry.get(key)
        if value is not None: return len(value)
    value = entry.get("X", entry.get("seq_X"))
    return None if value is None else len(value)


def segment_labels_from_attr(molecule_data):
    label_column = "segment_smiles" if "segment_smiles" in molecule_data.columns else "segment_label"
    if label_column not in molecule_data.columns: return []
    output = str(molecule_data["output_name"].dropna().astype(str).iloc[0])
    subset = molecule_data[molecule_data["output_name"].astype(str) == output].copy()
    subset["segment_pos"] = pd.to_numeric(subset["segment_pos"], errors="coerce")
    subset = subset.dropna(subset=["segment_pos"]).sort_values("segment_pos")
    return [str(x) for x in subset[label_column]]


def source_order_from_attr(molecule_data):
    for column in ["source_segment_pos", "record_pos", "record_index"]:
        if column not in molecule_data.columns: continue
        output = str(molecule_data["output_name"].dropna().astype(str).iloc[0])
        subset = molecule_data[molecule_data["output_name"].astype(str) == output].copy()
        subset["segment_pos"] = pd.to_numeric(subset["segment_pos"], errors="coerce")
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
        subset = subset.dropna(subset=["segment_pos", column]).sort_values("segment_pos")
        if len(subset):
            return [int(x) for x in subset[column]]
    return []


def record_labels(payload):
    return [str(x.get("segment_smiles", "")) for x in get_records(payload)]


def labels_match(labels, order, expected):
    if not expected or len(order) < len(expected): return False
    return [labels[i] for i in order[:len(expected)]] == expected


def infer_order_from_labels(labels, expected):
    used, order = set(), []
    for label in expected:
        choices = [i for i, value in enumerate(labels) if value == label and i not in used]
        if not choices: return []
        order.append(choices[0])
        used.add(choices[0])
    return order


def has_unknown_labels(labels):
    return any(str(x).strip() in {"", "__UNK__"} for x in labels)


def is_group_attribution(data):
    return "group_pos" in data.columns and "model_segment_pos_list" in data.columns


def model_order_for_mol(bundle, mol_id, payload, molecule_data):
    records = get_records(payload)
    if is_group_attribution(molecule_data):
        order = get_segment_order(payload, len(records))
        seq_cache = bundle.get("seq_tensor_cache", {}) if isinstance(bundle, dict) else {}
        entry = seq_cache.get(str(mol_id)) if isinstance(seq_cache, dict) else None
        if isinstance(entry, dict):
            seq_order = clean_order(entry.get("segment_order"), len(records))
            if seq_order:
                return seq_order
        return order

    labels = record_labels(payload)
    attr_order = source_order_from_attr(molecule_data)
    expected = segment_labels_from_attr(molecule_data)
    if attr_order:
        cleaned = clean_order(attr_order, len(records))
        if len(cleaned) != len(attr_order):
            raise RuntimeError(f"{mol_id}: invalid source segment positions in attribution CSV")
        if expected and not has_unknown_labels(expected) and not labels_match(labels, cleaned, expected):
            raise RuntimeError(f"{mol_id}: attribution source positions do not match segment bundle")
        return cleaned

    order = get_segment_order(payload, len(records))
    seq_cache = bundle.get("seq_tensor_cache", {}) if isinstance(bundle, dict) else {}
    entry = seq_cache.get(str(mol_id)) if isinstance(seq_cache, dict) else None
    if not isinstance(entry, dict):
        inferred = infer_order_from_labels(labels, expected)
        if inferred:
            print(f"[PLOT][WARN] {mol_id}: inferred segment order from token strings; exact mapping needs source_segment_pos or seq_tensor_cache")
        return inferred or order

    seq_len = seq_entry_len(entry)
    seq_order = clean_order(entry.get("segment_order"), len(records))
    if seq_order:
        if seq_order != order[:len(seq_order)]:
            raise RuntimeError(f"{mol_id}: segment_cache and seq_tensor_cache segment_order disagree")
        order = seq_order
    if seq_len is not None:
        order = order[:int(seq_len)]

    if expected and not has_unknown_labels(expected) and not labels_match(labels, order, expected):
        raise RuntimeError(f"{mol_id}: attribution CSV does not match segment bundle sequence order")
    return order


def segment_map_rows(molecule_data, payload, model_order):
    if "segment_pos" not in molecule_data.columns:
        return []

    first = molecule_data.iloc[0]
    records = get_records(payload)
    attr = molecule_data.copy()
    attr["segment_pos"] = pd.to_numeric(attr["segment_pos"], errors="coerce")
    attr = attr.dropna(subset=["segment_pos"]).sort_values("segment_pos").drop_duplicates("segment_pos")
    attr_by_pos = {int(row["segment_pos"]): row for _, row in attr.iterrows()}
    rows = []

    for model_pos, raw_pos in enumerate(model_order):
        if not 0 <= raw_pos < len(records): continue
        record = records[raw_pos]
        attr_row = attr_by_pos.get(model_pos, {})
        rows.append({
            "mol_id": str(first["mol_id"]),
            "task": str(first.get("task", "default")),
            "model_segment_number": model_pos + 1,
            "source_segment_number": raw_pos + 1,
            "source_segment_label": record.get("segment_label", f"S{raw_pos + 1}"),
            "segment_smiles": record.get("segment_smiles", ""),
            "attribution_segment_smiles": attr_row.get("segment_smiles", attr_row.get("segment_label", "")),
            "atom_ids": " ".join(str(int(x)) for x in record.get("atom_ids", []) or []),
        })
    return rows


def parse_position_list(value):
    out = []
    for part in re.split(r"[,+;|\s]+", str(value or "")):
        part = part.strip()
        if not part:
            continue
        if part.upper().startswith("S"):
            part = part[1:]
        try:
            out.append(int(part) - 1)
        except Exception:
            pass
    return out


def parse_zero_based_position_list(value):
    out = []
    for part in re.split(r"[,;|\s]+", str(value or "")):
        try:
            out.append(int(part.strip()))
        except Exception:
            pass
    return out


def plot_group_contribution(subset, payload, output_name, path, args):
    subset["group_pos"] = pd.to_numeric(subset["group_pos"], errors="raise").astype(int)
    subset = subset.sort_values("group_pos")
    first = subset.iloc[0]
    task, n_groups = str(first.get("task", "default")), len(subset)
    contributions = pd.to_numeric(subset["contribution_mean"], errors="coerce").to_numpy(float)
    if not np.isfinite(contributions).any():
        return False

    n_records = len(get_records(payload))
    palette = build_colors(max(n_records, 1))
    labels, colors = [], []
    for _, row in subset.iterrows():
        label = str(row.get("group_label") or row.get("source_segment_numbers") or f"R{int(row['group_pos']) + 1}").strip()
        source_numbers = str(row.get("source_segment_numbers") or "").strip()
        if source_numbers and source_numbers != label and not label.startswith("S"):
            label = f"{label} ({source_numbers})"
        source_positions = parse_position_list(row.get("source_segment_numbers"))
        if not source_positions:
            source_positions = parse_zero_based_position_list(row.get("source_segment_pos_list"))
        source_pos = source_positions[0] if source_positions else 0
        labels.append(label)
        colors.append(palette[source_pos % len(palette)])

    figure, attribution_ax = create_contribution_figure(n_groups)
    y = np.arange(n_groups)
    attribution_ax.barh(y, contributions, color=colors, alpha=0.9)
    attribution_ax.axvline(0, color="black", linewidth=1)
    attribution_ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel=attribution_axis_label(args.property, task).replace("Segment", "Residue/group"),
    )

    note = contribution_note(args.property, task)

    attribution_ax.text(
        0.98, 0.02, note,
        transform=attribution_ax.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="gray")
    )
    attribution_ax.invert_yaxis()
    attribution_ax.grid(axis="x", alpha=0.25)
    figure.tight_layout(pad=1.0)
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_segmentation(molecule_data, payload, model_order, path):
    first = molecule_data.iloc[0]
    mol_id = str(first["mol_id"])
    n_segments = len(get_records(payload))
    height = 650 if n_segments <= 12 else 750 if n_segments <= 24 else 900
    image = draw_segment_map(
        mol_id, payload,
        fallback_smiles=str(first.get("canonical_smiles", "")),
        width=1500, height=height,
    )
    image.save(path)


def plot_contribution(molecule_data, payload, model_order, output_name, path, args):
    subset = molecule_data[molecule_data["output_name"].astype(str) == str(output_name)].copy()
    if subset.empty: return False
    if is_group_attribution(subset):
        return plot_group_contribution(subset, payload, output_name, path, args)

    subset["segment_pos"] = pd.to_numeric(subset["segment_pos"], errors="raise").astype(int)
    subset = subset.sort_values("segment_pos")
    observed = sorted(subset["segment_pos"].unique())
    if observed != list(range(len(observed))) or max(observed, default=-1) >= len(model_order):
        raise RuntimeError(f"{subset.iloc[0]['mol_id']}: attribution positions {observed} do not match model segment order")

    if "source_segment_pos" in subset.columns:
        subset["plot_segment_pos"] = pd.to_numeric(subset["source_segment_pos"], errors="raise").astype(int)
    else:
        subset["plot_segment_pos"] = subset["segment_pos"]
    subset = subset.sort_values("plot_segment_pos")

    first = subset.iloc[0]
    task, n_segments = str(first.get("task", "default")), len(subset)
    contributions = pd.to_numeric(subset["contribution_mean"], errors="coerce").to_numpy(float)
    if not np.isfinite(contributions).any():
        return False

    n_records = max(len(get_records(payload)), int(subset["plot_segment_pos"].max()) + 1)
    palette = build_colors(n_records)
    plot_positions = [int(x) for x in subset["plot_segment_pos"]]
    colors = [palette[i] for i in plot_positions]
    labels = [f"S{i + 1}" for i in plot_positions]
    figure, attribution_ax = create_contribution_figure(n_segments)

    y = np.arange(n_segments)
    attribution_ax.barh(y, contributions, color=colors, alpha=0.9)
    attribution_ax.axvline(0, color="black", linewidth=1)
    attribution_ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel=attribution_axis_label(args.property, task),
    )

    note = contribution_note(args.property, task)

    attribution_ax.text(
        0.98, 0.02, note,
        transform=attribution_ax.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="gray")
    )
    attribution_ax.invert_yaxis()
    attribution_ax.grid(axis="x", alpha=0.25)
    figure.tight_layout(pad=1.0)
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    return True


def run_attribution_plots(args):
    tasks = parse_filter(getattr(args, "tasks", "all"))
    data = filter_tasks(normalize_columns(pd.read_csv(args.attr_csv)), tasks, "Attribution")
    predictions = filter_tasks(pd.read_csv(args.pred_csv, low_memory=False), tasks, "Prediction") if getattr(args, "pred_csv", None) else None
    position_column = "group_pos" if "group_pos" in data.columns else "segment_pos"
    required = ["mol_id", position_column, "output_name", "original_prediction_mean", "contribution_mean"]
    missing = [x for x in required if x not in data.columns]
    if missing: raise KeyError(f"Attribution CSV missing columns: {missing}")

    available = data["task"].dropna().astype(str).drop_duplicates().tolist() if "task" in data else ["default"]
    print(f"[PLOT] tasks={available}")

    bundle = load_bundle(args.segment_bundle)
    cache = get_drawable_cache(bundle)
    outputs = parse_filter(args.outputs)
    auto_outputs = outputs is None
    outputs = outputs or data["output_name"].dropna().astype(str).drop_duplicates().tolist()
    keys, selected = select_molecules(data, args.top_n_mols, args.select_by)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = Path(args.results_dir).resolve() / "_shared" / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    flat = task_key(args.property) in {"pampa", "solubility"}
    plot_count = 0
    segmented = set()

    for _, key_row in selected.iterrows():
        molecule_data = subset_for_key(data, keys, key_row)
        if molecule_data.empty: continue
        mol_id = str(molecule_data.iloc[0]["mol_id"])
        task = str(molecule_data.iloc[0].get("task", "default"))
        payload = cache.get(mol_id)
        if payload is None:
            print(f"[PLOT][WARN] no segment payload for {mol_id}")
            continue
        model_order = model_order_for_mol(bundle, mol_id, payload, molecule_data)
        segment_path = segment_dir / f"{safe_name(mol_id)}__segmentation.{args.format}"
        if mol_id not in segmented:
            plot_segmentation(molecule_data, payload, model_order, segment_path)
            plot_count += 1
            segmented.add(mol_id)

        plot_dir = out_dir if flat else out_dir / safe_name(task)
        plot_dir.mkdir(parents=True, exist_ok=True)

        prediction_row = find_prediction_row(predictions, keys, key_row)
        molecule_outputs = compact_outputs_from_predictions(
            molecule_data, outputs, getattr(args, "prob_cutoff", 0.5), prediction_row
        ) if auto_outputs else outputs
        for output_name in molecule_outputs:
            path = plot_dir / f"{safe_name(mol_id)}__{safe_name(output_name)}.{args.format}"
            if plot_contribution(molecule_data, payload, model_order, output_name, path, args):
                plot_count += 1

    print(f"[SAVE] {out_dir} | plots={plot_count}")


