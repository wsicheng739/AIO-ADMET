import numpy as np
import torch
from tqdm import tqdm

from .data import normalize_item


def _transform_predictions(adapter, logits, n_items, n_outputs, model_name):
    predictions = adapter.transform_output(logits).detach().cpu().numpy()
    expected = n_items * n_outputs
    if predictions.size != expected:
        raise RuntimeError(f"{model_name}: output size={predictions.size}, expected={expected}")
    return predictions.reshape(n_items, n_outputs)


def _group_indices(items):
    groups = {}
    for index, item in enumerate(items):
        key = len(item["graphs_h2o"]), len(item["graphs_chcl3"])
        groups.setdefault(key, []).append(index)
    return groups.values()


def _make_masked_item(item, segment_pos, mask_mode, unk_segment_id):
    masked = dict(item)
    positions = np.asarray(segment_pos, dtype=int).reshape(-1)

    if mask_mode in {"both", "zero"}:
        masked["seq_X"] = item["seq_X"].clone()
        masked["seq_X"][positions].zero_()

    if mask_mode in {"both", "unk"}:
        masked["segment_idx"] = item["segment_idx"].clone()
        masked["segment_idx"][positions] = int(unk_segment_id)

    return masked


@torch.no_grad()
def predict_one_model(model, adapter, items, norm, device, model_name, batch_size,
                      attribution_n_mols=0, mask_mode="both", unk_segment_id=0,
                      show_progress=True):
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    n_items = len(items)
    n_outputs = len(norm["output_names"])
    n_attr = min(max(int(attribution_n_mols), 0), n_items)

    predictions = np.full((n_items, n_outputs), np.nan, dtype=np.float64)
    masked_predictions = []
    residue_masked_predictions = []

    with tqdm(
        total=n_items,
        desc=f"[INFER {model_name}]",
        leave=True,
        unit="mol",
        disable=not show_progress,
    ) as pbar:
        for start in range(0, n_items, batch_size):
            stop = min(start + batch_size, n_items)
            batch = [normalize_item(item, norm) for item in items[start:stop]]
            static_features = [None] * len(batch)

            for indices in _group_indices(batch):
                grouped = [batch[index] for index in indices]
                tasks = [item["task"] for item in grouped]

                static = model.encode_static_feature_batch(grouped, device)
                sequence = model.encode_sequence_feature_batch(grouped, device)
                logits = model.logits_from_features_batch(tasks, static, sequence)
                output = _transform_predictions(adapter, logits, len(grouped), n_outputs, model_name)

                for group_index, batch_index in enumerate(indices):
                    predictions[start + batch_index] = output[group_index]
                    static_features[batch_index] = static[group_index]

            for batch_index, item in enumerate(batch):
                row_index = start + batch_index
                if row_index >= n_attr:
                    break

                seq_len = int(item["seq_len"])
                masked_output = np.full((seq_len, n_outputs), np.nan, dtype=np.float64)

                for attr_start in range(0, seq_len, batch_size):
                    attr_stop = min(attr_start + batch_size, seq_len)
                    masked = [
                        _make_masked_item(item, position, mask_mode, unk_segment_id)
                        for position in range(attr_start, attr_stop)
                    ]

                    sequence = model.encode_sequence_feature_batch(masked, device)
                    static = static_features[batch_index].unsqueeze(0).expand(len(masked), -1)
                    tasks = [item["task"]] * len(masked)
                    logits = model.logits_from_features_batch(tasks, static, sequence)

                    masked_output[attr_start:attr_stop] = _transform_predictions(adapter, logits, len(masked), n_outputs, model_name)

                masked_predictions.append(masked_output)

                groups = item.get("attribution_groups") or []
                group_output = np.full((len(groups), n_outputs), np.nan, dtype=np.float64)

                for attr_start in range(0, len(groups), batch_size):
                    attr_stop = min(attr_start + batch_size, len(groups))
                    masked = [
                        _make_masked_item(item, groups[position]["model_positions"], mask_mode, unk_segment_id)
                        for position in range(attr_start, attr_stop)
                    ]

                    if not masked:
                        continue

                    sequence = model.encode_sequence_feature_batch(masked, device)
                    static = static_features[batch_index].unsqueeze(0).expand(len(masked), -1)
                    tasks = [item["task"]] * len(masked)
                    logits = model.logits_from_features_batch(tasks, static, sequence)
                    group_output[attr_start:attr_stop] = _transform_predictions(adapter, logits, len(masked), n_outputs, model_name)

                residue_masked_predictions.append(group_output)

            pbar.update(len(batch))

    return predictions, masked_predictions, residue_masked_predictions
