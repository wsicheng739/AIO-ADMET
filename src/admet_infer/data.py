import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import clean_topology, finite_or_nan, safe_torch_load

try:
    from rdkit import Chem
except Exception:
    Chem = None


CYCLIC_TOPOLOGIES = {
    "cyclo_simple", "simple_cyclo", "cyclo_branched", "branched_cyclo",
    "bridged", "bicycle", "mixed", "multicyclo",
}


def task_key(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")



def segment_identity(smiles):
    smiles = str(smiles)
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return Chem.MolToSmiles(mol, isomericSmiles=True)


def validate_segment_bundle_identity(bundle):
    segment_cache = bundle.get("segment_cache", {})
    if not isinstance(segment_cache, dict):
        return

    vocab = bundle["segment_vocab"]
    seq_cache = bundle["seq_tensor_cache"]
    id_to_smiles = {}
    for smiles, segment_id in vocab.items():
        id_to_smiles.setdefault(int(segment_id), str(smiles))

    errors = []
    for mol_id, entry in seq_cache.items():
        payload = segment_cache.get(str(mol_id))
        if not isinstance(entry, dict) or not isinstance(payload, dict):
            continue
        records = payload.get("records", [])
        ids = entry.get("segment_ids", entry.get("segment_idx", []))
        order = entry.get("segment_order", list(range(len(ids))))
        ids = np.asarray(ids, dtype=np.int64)

        if len(order) != len(ids):
            errors.append(f"{mol_id}: segment_order={len(order)} segment_ids={len(ids)}")
            continue

        for model_pos, (raw_pos, segment_id) in enumerate(zip(order, ids)):
            raw_pos, segment_id = int(raw_pos), int(segment_id)
            if raw_pos < 0 or raw_pos >= len(records):
                errors.append(f"{mol_id}: model S{model_pos + 1} raw_pos={raw_pos} out of range")
                continue

            rec_smiles = str(records[raw_pos].get("segment_smiles", ""))
            expected_id = int(vocab.get(rec_smiles, 0))
            if expected_id != segment_id:
                errors.append(f"{mol_id}: model S{model_pos + 1} id={segment_id}, vocab({rec_smiles})={expected_id}")
                continue

            token_smiles = id_to_smiles.get(segment_id, "")
            if segment_id and token_smiles and segment_identity(rec_smiles) != segment_identity(token_smiles):
                errors.append(f"{mol_id}: model S{model_pos + 1} smiles/inchikey mismatch")

            if len(errors) >= 8:
                raise RuntimeError("Segment bundle identity check failed:\n" + "\n".join(errors))

    if errors:
        raise RuntimeError("Segment bundle identity check failed:\n" + "\n".join(errors))

def load_segment_bundle(path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    missing = [x for x in ["segment_vocab", "seq_tensor_cache"] if x not in bundle]
    if missing:
        raise KeyError(f"Segment bundle missing keys: {missing}")

    vocab = bundle["segment_vocab"]
    cache = {str(k): v for k, v in bundle["seq_tensor_cache"].items()}
    validate_segment_bundle_identity(bundle)
    return vocab, cache, bundle.get("meta", {})


def unpack_sequence_entry(entry, mol_id, mono_dim):
    notation = ""
    segment_order = None

    if isinstance(entry, dict):
        features = entry.get("X", entry.get("seq_X"))
        segment_ids = entry.get("segment_ids", entry.get("segment_idx"))
        topology = entry.get("topology")
        notation = str(entry.get("notation", ""))
        segment_order = entry.get("segment_order")
    elif isinstance(entry, (tuple, list)) and len(entry) >= 3 and isinstance(entry[2], dict):
        features, segment_ids, topology = entry[:3]
    else:
        raise TypeError(f"{mol_id}: unsupported sequence-cache entry")

    if features is None or segment_ids is None or topology is None:
        raise ValueError(f"{mol_id}: sequence entry is missing features, IDs, or topology")
    if not isinstance(topology, dict) or "topology_type" not in topology:
        raise ValueError(f"{mol_id}: invalid topology metadata")

    topology_name = clean_topology(topology["topology_type"])
    if topology_name in CYCLIC_TOPOLOGIES and topology.get("main_cycle") is None:
        raise ValueError(f"{mol_id}: cyclic topology is missing main_cycle")

    features = np.asarray(features, dtype=np.float32)
    segment_ids = np.asarray(segment_ids, dtype=np.int64)

    if features.ndim != 2 or segment_ids.ndim != 1:
        raise ValueError(f"{mol_id}: invalid sequence tensor dimensions")
    if len(features) != len(segment_ids):
        raise ValueError(f"{mol_id}: sequence feature/ID length mismatch")
    if features.shape[1] != mono_dim:
        raise ValueError(f"{mol_id}: mono feature dimension={features.shape[1]}, expected={mono_dim}")

    if segment_order is None:
        segment_order = np.arange(len(segment_ids), dtype=np.int64)
    segment_order = np.asarray(segment_order, dtype=np.int64)
    if len(segment_order) != len(segment_ids):
        raise ValueError(f"{mol_id}: segment_order length mismatch")

    return features, segment_ids, topology, notation, segment_order


def graph_path(mol_id, graph_dir, selection_tag):
    return Path(graph_dir) / f"{mol_id}_graphs_{selection_tag}.pt"


class RawInferenceDataset(Dataset):
    def __init__(self, input_df, feature_df, seq_cache, norm, config, graph_dir, species_override=None):
        columns = config.get("columns", {})

        self.id_col = columns.get("id", "InChIKey")
        self.smiles_col = columns.get("smiles", "canonical_smiles")
        self.task_col = columns.get("task", "species_norm")
        self.value_col = columns.get("value", "value_parsed")
        self.source_col = columns.get("source", "source")
        self.selection_tag = config.get("selection_tag", "v1_5conf")
        self.graph_dir = Path(graph_dir)
        self.seq_cache = seq_cache
        self.norm = norm
        self.config = config

        self.task_names = [str(x) for x in norm["task_names"]]
        self.task_lookup = {task_key(x): x for x in self.task_names}

        if len(self.task_lookup) != len(self.task_names):
            raise ValueError(f"Task names collide after normalization: {self.task_names}")

        self.skipped_rows = []
        self.qc = {
            "n_input_rows": 0,
            "n_after_merge_feat": 0,
            "n_missing_feat": 0,
            "n_invalid_task": 0,
            "n_missing_graph": 0,
            "n_bad_graph_obj": 0,
            "n_empty_graph_list": 0,
            "n_missing_seqfeat": 0,
            "n_bad_seqfeat": 0,
            "n_nonfinite_desc": 0,
            "n_nonfinite_seq": 0,
            "n_kept": 0,
        }

        data = input_df.copy().reset_index(drop=True)

        if self.id_col not in data:
            raise KeyError(f"Input CSV missing required column: {self.id_col}")

        data[self.id_col] = data[self.id_col].astype(str)

        if "row_id" not in data:
            data["row_id"] = np.arange(len(data)).astype(str)
        else:
            data["row_id"] = data["row_id"].astype(str)

        if self.smiles_col not in data:
            data[self.smiles_col] = ""
        if self.source_col not in data:
            data[self.source_col] = "NA"

        self.input_columns = list(data.columns)

        if species_override:
            data["_task"] = str(species_override)
        elif self.task_col in data:
            data["_task"] = data[self.task_col].astype(str)
        else:
            default_task = config.get("default_task")
            if not default_task:
                raise KeyError(f"Input CSV has no {self.task_col}; supply --species or set default_task")
            if isinstance(default_task, (list, tuple)):
                raise TypeError("default_task must be a string; use default_tasks expansion in runner.py")
            data["_task"] = str(default_task)

        overlap = [x for x in norm["feat_cols"] if x in data]
        if overlap:
            data = data.drop(columns=overlap)

        needed = [self.id_col] + norm["feat_cols"]
        missing = [x for x in needed if x not in feature_df]
        if missing:
            raise KeyError(f"Feature CSV missing columns: {missing}")

        features = feature_df[needed].copy()
        features[self.id_col] = features[self.id_col].astype(str)
        features = features.drop_duplicates(self.id_col, keep="first")

        self.data = data.merge(features, on=self.id_col, how="left", indicator="_feature_merge")

        matched = int((self.data["_feature_merge"] == "both").sum())
        self.qc["n_input_rows"] = len(data)
        self.qc["n_after_merge_feat"] = matched
        self.qc["n_missing_feat"] = len(data) - matched

    def __len__(self):
        return len(self.data)

    def skip(self, index, mol_id, reason, detail=""):
        row = self.data.iloc[index]
        self.skipped_rows.append({
            "row_id": str(row.get("row_id", index)),
            self.id_col: mol_id,
            "task": str(row.get("_task", "")),
            "reason": reason,
            "detail": detail,
        })

        if len(self.skipped_rows) <= 20:
            print(f"[SKIP] {reason}: {mol_id}" + (f" | {detail}" if detail else ""))

    def __getitem__(self, index):
        row = self.data.iloc[index]
        mol_id = str(row[self.id_col])

        raw_task = row.get("_task", "")
        task = self.task_lookup.get(task_key(raw_task))

        if task is None:
            self.qc["n_invalid_task"] += 1
            self.skip(index, mol_id, "invalid_task", f"task={raw_task} | allowed={self.task_names}")
            return None

        if row.get("_feature_merge") != "both":
            self.skip(index, mol_id, "missing_feature_row")
            return None

        descriptors = pd.to_numeric(row[self.norm["feat_cols"]], errors="coerce").to_numpy(np.float32)

        if not np.isfinite(descriptors).all():
            self.qc["n_nonfinite_desc"] += 1
            self.skip(index, mol_id, "nonfinite_descriptor")
            return None

        path = graph_path(mol_id, self.graph_dir, self.selection_tag)

        if not path.exists():
            self.qc["n_missing_graph"] += 1
            self.skip(index, mol_id, "missing_graph", str(path))
            return None

        try:
            graph_data = safe_torch_load(path, map_location="cpu")
        except Exception as error:
            self.qc["n_bad_graph_obj"] += 1
            self.skip(index, mol_id, "bad_graph_load", repr(error))
            return None

        if not isinstance(graph_data, dict):
            self.qc["n_bad_graph_obj"] += 1
            self.skip(index, mol_id, "bad_graph_object", str(type(graph_data)))
            return None

        graphs_h2o = graph_data.get("graphs_h2o", [])
        graphs_chcl3 = graph_data.get("graphs_chcl3", [])

        if not graphs_h2o or not graphs_chcl3:
            self.qc["n_empty_graph_list"] += 1
            self.skip(index, mol_id, "empty_graph_list")
            return None

        entry = self.seq_cache.get(mol_id)

        if entry is None:
            self.qc["n_missing_seqfeat"] += 1
            self.skip(index, mol_id, "missing_seq_cache")
            return None

        try:
            seq_features, segment_ids, topology, notation, segment_order = unpack_sequence_entry(
                entry, mol_id, self.config.get("architecture", {}).get("mono_dim", 15)
            )
        except Exception as error:
            self.qc["n_bad_seqfeat"] += 1
            self.skip(index, mol_id, "bad_seq_topology_cache", repr(error))
            return None

        if not np.isfinite(seq_features).all():
            self.qc["n_nonfinite_seq"] += 1
            self.skip(index, mol_id, "nonfinite_sequence_features")
            return None

        topology_name = clean_topology(topology.get("topology_type", "unknown"))
        input_row = {name: row.get(name, "") for name in self.input_columns}
        self.qc["n_kept"] += 1

        return {
            "input_index": int(index),
            "input_row": input_row,
            "row_id": str(row.get("row_id", index)),
            "mol_id": mol_id,
            "task": task,
            "species_norm": task,
            "source": str(row.get(self.source_col, "NA")),
            "smiles": str(row.get(self.smiles_col, "")),
            "value_raw": finite_or_nan(row.get(self.value_col, float("nan"))),
            "notation": notation,
            "topology_type": str(topology.get("topology_type", "unknown")),
            "is_cyclic": int(topology_name in CYCLIC_TOPOLOGIES),
            "seq_len": int(len(seq_features)),
            "n_conf_h2o": len(graphs_h2o),
            "n_conf_chcl3": len(graphs_chcl3),
            "graphs_h2o": graphs_h2o,
            "graphs_chcl3": graphs_chcl3,
            "desc_raw": descriptors,
            "seq_X_raw": seq_features,
            "segment_idx": torch.tensor(segment_ids, dtype=torch.long),
            "segment_order": segment_order,
            "topology": topology,
        }


def collate_valid(batch):
    return [x for x in batch if x is not None]


def materialize(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_valid, num_workers=0)
    items = []

    for batch in tqdm(loader, desc="[LOAD CACHE]", leave=True):
        items.extend(batch)

    return items


def normalize_item(raw_item, norm):
    descriptors = (raw_item["desc_raw"] - norm["mu_desc"]) / (norm["sd_desc"] + 1e-12)
    sequence = (raw_item["seq_X_raw"] - norm["mu_mono"]) / (norm["sd_mono"] + 1e-12)

    descriptors = np.clip(descriptors, -10, 10).astype(np.float32)
    sequence = np.clip(sequence, -10, 10).astype(np.float32)

    if not np.isfinite(descriptors).all() or not np.isfinite(sequence).all():
        raise ValueError(f"{raw_item['mol_id']}: normalization produced non-finite values")

    return {
        "mol_id": raw_item["mol_id"],
        "task": raw_item["task"],
        "graphs_h2o": raw_item["graphs_h2o"],
        "graphs_chcl3": raw_item["graphs_chcl3"],
        "pep_feat": torch.tensor(descriptors, dtype=torch.float32),
        "seq_X": torch.tensor(sequence, dtype=torch.float32),
        "segment_idx": raw_item["segment_idx"],
        "seq_len": raw_item["seq_len"],
        "n_res": raw_item["seq_len"],
        "topology": raw_item["topology"],
        "is_cyclic": raw_item["is_cyclic"],
        "attribution_groups": raw_item.get("attribution_groups", []),
    }
