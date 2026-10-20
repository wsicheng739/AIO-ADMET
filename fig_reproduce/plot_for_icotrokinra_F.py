#!/usr/bin/env python3
"""Plot bioavailability residue attribution for Icotrokinra only."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from admet_infer.plotting.attribution_plot import run_attribution_plots
from admet_infer.plotting.segment_drawing import draw_segment_map,get_records,get_drawable_cache,load_bundle

MOLECULE = "Icotrokinra"
PROPERTY = "bioavailability"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--format", choices=("png", "pdf"), default="png")
    p.add_argument("--no-error-bars", action="store_true")
    cli = p.parse_args()

    results = ROOT / "run_test/external_cases"
    property_dir = results / PROPERTY
    cache = ROOT / "data/external_five_case/cache"
    attr_csv = property_dir / "icotrokinra_residue_attribution.csv"
    pred_csv = property_dir / "predictions.csv"
    segment_bundle = cache / "admet_segment_bundle_simplified.pkl"
    config_file = ROOT / "configs/properties/bioavailability.json"
    out_dir = SCRIPT_DIR / "plot_for_icotrokinra_F" / MOLECULE

    for path in (attr_csv, pred_csv, segment_bundle, config_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    df = pd.read_csv(attr_csv, low_memory=False)
    columns = [c for c in ("candidate_id", "name", "InChIKey", "mol_id") if c in df]
    if not columns:
        raise ValueError("Attribution CSV has no molecule identifier column")

    selected = df[pd.concat(
        [df[c].astype("string").str.strip().eq(MOLECULE) for c in columns], axis=1
    ).any(axis=1)].copy()
    if selected.empty:
        raise ValueError(f"{MOLECULE!r} was not found using columns {columns}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prediction_plot_") as tmp:
        filtered = Path(tmp) / "residue_attribution.csv"
        selected.to_csv(filtered, index=False)
        config = json.loads(config_file.read_text(encoding="utf-8"))
        args = argparse.Namespace(
            property=PROPERTY, results_dir=results, cache_dir=cache,
            attr_csv=filtered, attribution_level="residue", pred_csv=pred_csv,
            segment_bundle=segment_bundle, out_dir=out_dir, tasks="all",
            outputs="all", mols=MOLECULE, top_n_mols=0,
            select_by="input_order", dpi=cli.dpi, format=cli.format,
            no_error_bars=cli.no_error_bars,
            prob_cutoff=float(config.get("prediction_threshold", .5)),
        )
        print(f"Plotting {MOLECULE}: {len(selected)} residue rows")
        run_attribution_plots(args)

        identifiers = []
        for column in ("mol_id", "InChIKey", "candidate_id", "name"):
            if column in selected:
                identifiers.extend(selected[column].dropna().astype(str).str.strip().tolist())
        drawable = get_drawable_cache(load_bundle(segment_bundle))
        mol_id = next((value for value in dict.fromkeys(identifiers) if value in drawable), None)
        if mol_id is None:
            raise KeyError(f"No segment-cache entry matched {MOLECULE!r}; tried {sorted(set(identifiers))}")

        smiles = ""
        for column in ("canonical_smiles", "smiles", "Smiles_unify"):
            if column in selected and selected[column].notna().any():
                smiles = str(selected.loc[selected[column].notna(), column].iloc[0])
                break

        payload = drawable[mol_id]
        records = get_records(payload)
        source_positions = sorted({
            int(pos)
            for value in selected["source_segment_pos_list"].dropna().astype(str)
            for pos in value.split(",")
            if pos.strip()
        })
        missing_positions = [pos for pos in source_positions if not 0 <= pos < len(records)]
        if missing_positions:
            raise ValueError(
                f"Attribution source positions absent from segment payload: {missing_positions}; "
                f"payload has {len(records)} records"
            )
        segment_image = draw_segment_map(mol_id, payload, fallback_smiles=smiles)
        segment_path = out_dir / f"{MOLECULE}_segmentation.{cli.format}"
        segment_image.save(segment_path, dpi=(cli.dpi, cli.dpi))
        print(f"Saved segmentation: {segment_path}")

    print(f"Saved plots: {out_dir}")


if __name__ == "__main__":
    main()
