import numpy as np
import pandas as pd

from .ensemble import finalize_accumulator, new_accumulator, update_accumulator


def initialize_attribution(items, n_molecules, n_outputs):
    n_molecules = min(max(int(n_molecules), 0), len(items))
    accumulators = []
    for item in items[:n_molecules]:
        shape = (item["seq_len"], n_outputs)
        accumulators.append({
            "original": new_accumulator(shape),
            "masked": new_accumulator(shape),
            "contribution": new_accumulator(shape),
        })
    return accumulators


def initialize_residue_attribution(items, n_molecules, n_outputs):
    n_molecules = min(max(int(n_molecules), 0), len(items))
    accumulators = []
    for item in items[:n_molecules]:
        groups = item.get("attribution_groups") or []
        shape = (len(groups), n_outputs)
        accumulators.append({
            "original": new_accumulator(shape),
            "masked": new_accumulator(shape),
            "contribution": new_accumulator(shape),
        })
    return accumulators


def update_attribution(accumulators, predictions, masked_predictions):
    if len(accumulators) != len(masked_predictions):
        raise RuntimeError("Attribution molecule count changed during inference")

    for row_index, (accumulator, masked) in enumerate(zip(accumulators, masked_predictions)):
        if accumulator["masked"]["sum"].shape != np.asarray(masked).shape:
            raise RuntimeError("Attribution mask shape does not match accumulator")
        original = np.broadcast_to(predictions[row_index][None, :], masked.shape).copy()
        contribution = original - masked
        update_accumulator(accumulator["original"], original)
        update_accumulator(accumulator["masked"], masked)
        update_accumulator(accumulator["contribution"], contribution)


def update_residue_attribution(accumulators, predictions, masked_predictions):
    update_attribution(accumulators, predictions, masked_predictions)


def build_attribution_dataframe(items,accumulators,norm,segment_vocab, property_name,mask_mode,unk_segment_id):
    id_to_segment = {}
    for token, token_id in segment_vocab.items():
        try:
            id_to_segment.setdefault(int(token_id), str(token))
        except Exception:
            pass

    rows = []
    for item, accumulator in zip(items, accumulators):
        original = finalize_accumulator(accumulator["original"])
        masked = finalize_accumulator(accumulator["masked"])
        contribution = finalize_accumulator(accumulator["contribution"])
        segment_ids = item["segment_idx"].detach().cpu().numpy().astype(int)
        segment_order = item.get("segment_order", np.arange(len(segment_ids)))
        segment_order = np.asarray(segment_order, dtype=int)
        input_row = item.get("input_row", {})

        for segment_pos, segment_id in enumerate(segment_ids):
            source_pos = int(segment_order[segment_pos])
            segment_smiles = id_to_segment.get(int(segment_id), str(segment_id))
            for output_index, output_name in enumerate(norm["output_names"]):
                output_value = norm["output_values"][output_index]
                row = {
                    "input_index": item["input_index"],
                    "row_id": item["row_id"],
                    "mol_id": item["mol_id"],
                    "name": input_row.get("name", input_row.get("Name", "")),
                    "task": item["task"],
                    "property": property_name,
                    "canonical_smiles": item.get("smiles", ""),
                    "segment_pos": segment_pos,
                    "segment_number": segment_pos + 1,
                    "model_segment_pos": segment_pos,
                    "model_segment_number": segment_pos + 1,
                    "source_segment_pos": source_pos,
                    "source_segment_number": source_pos + 1,
                    "segment_id": int(segment_id),
                    "segment_smiles": segment_smiles,
                    "segment_label": segment_smiles,
                    "output_name": str(output_name),
                    "output_value": float(output_value),
                    "mask_mode": mask_mode,
                    "unk_segment_id": int(unk_segment_id),
                    "original_prediction_mean": float(original["mean"][segment_pos, output_index]),
                    "original_prediction_std": float(original["std"][segment_pos, output_index]),
                    "masked_prediction_mean": float(masked["mean"][segment_pos, output_index]),
                    "masked_prediction_std": float(masked["std"][segment_pos, output_index]),
                    "contribution_mean": float(contribution["mean"][segment_pos, output_index]),
                    "contribution_std": float(contribution["std"][segment_pos, output_index]),
                    "contribution_min": float(contribution["min"][segment_pos, output_index]),
                    "contribution_max": float(contribution["max"][segment_pos, output_index]),
                    "n_models": int(contribution["count"][segment_pos, output_index]),
                }

                # Compatibility with the existing bioavailability plot files.
                row["threshold_name"] = row["output_name"]
                row["threshold_value"] = row["output_value"]
                row["original_probability_mean"] = row["original_prediction_mean"]
                row["original_probability_std"] = row["original_prediction_std"]
                row["masked_probability_mean"] = row["masked_prediction_mean"]
                row["masked_probability_std"] = row["masked_prediction_std"]
                row["contribution_probability_mean"] = row["contribution_mean"]
                row["contribution_probability_std"] = row["contribution_std"]
                row["contribution_probability_min"] = row["contribution_min"]
                row["contribution_probability_max"] = row["contribution_max"]
                rows.append(row)

    return pd.DataFrame(rows)


def build_residue_attribution_dataframe(items,accumulators, norm,property_name,mask_mode,unk_segment_id):
    rows = []
    for item, accumulator in zip(items, accumulators):
        groups = item.get("attribution_groups") or []
        if not groups:
            continue

        original = finalize_accumulator(accumulator["original"])
        masked = finalize_accumulator(accumulator["masked"])
        contribution = finalize_accumulator(accumulator["contribution"])
        input_row = item.get("input_row", {})

        for group_pos, group in enumerate(groups):
            source_numbers = [int(x) + 1 for x in group.get("source_positions", [])]
            model_numbers = [int(x) + 1 for x in group.get("model_positions", [])]
            group_source = group.get("source", "")
            attribution_level = "segment" if group_source in {"segment_fallback", "singleton"} else "residue"
            for output_index, output_name in enumerate(norm["output_names"]):
                output_value = norm["output_values"][output_index]
                row = {
                    "input_index": item["input_index"],
                    "row_id": item["row_id"],
                    "mol_id": item["mol_id"],
                    "name": input_row.get("name", input_row.get("Name", "")),
                    "task": item["task"],
                    "property": property_name,
                    "canonical_smiles": item.get("smiles", ""),
                    "attribution_level": attribution_level,
                    "group_pos": group_pos,
                    "group_number": group_pos + 1,
                    "group_label": group.get("label", f"R{group_pos + 1}"),
                    "group_name": group.get("name", ""),
                    "group_source": group_source,
                    "n_group_segments": len(model_numbers),
                    "source_segment_numbers": "+".join(f"S{x}" for x in source_numbers),
                    "model_segment_numbers": "+".join(f"S{x}" for x in model_numbers),
                    "source_segment_pos_list": ",".join(str(int(x)) for x in group.get("source_positions", [])),
                    "model_segment_pos_list": ",".join(str(int(x)) for x in group.get("model_positions", [])),
                    "segment_smiles": " | ".join(str(x) for x in group.get("segment_smiles", [])),
                    "output_name": str(output_name),
                    "output_value": float(output_value),
                    "mask_mode": mask_mode,
                    "unk_segment_id": int(unk_segment_id),
                    "original_prediction_mean": float(original["mean"][group_pos, output_index]),
                    "original_prediction_std": float(original["std"][group_pos, output_index]),
                    "masked_prediction_mean": float(masked["mean"][group_pos, output_index]),
                    "masked_prediction_std": float(masked["std"][group_pos, output_index]),
                    "contribution_mean": float(contribution["mean"][group_pos, output_index]),
                    "contribution_std": float(contribution["std"][group_pos, output_index]),
                    "contribution_min": float(contribution["min"][group_pos, output_index]),
                    "contribution_max": float(contribution["max"][group_pos, output_index]),
                    "n_models": int(contribution["count"][group_pos, output_index]),
                }

                row["threshold_name"] = row["output_name"]
                row["threshold_value"] = row["output_value"]
                row["original_probability_mean"] = row["original_prediction_mean"]
                row["original_probability_std"] = row["original_prediction_std"]
                row["masked_probability_mean"] = row["masked_prediction_mean"]
                row["masked_probability_std"] = row["masked_prediction_std"]
                row["contribution_probability_mean"] = row["contribution_mean"]
                row["contribution_probability_std"] = row["contribution_std"]
                row["contribution_probability_min"] = row["contribution_min"]
                row["contribution_probability_max"] = row["contribution_max"]
                rows.append(row)

    return pd.DataFrame(rows)
