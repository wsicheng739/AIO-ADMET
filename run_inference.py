#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from admet_infer.config import apply_overrides, load_property_config
from admet_infer.properties import get_property_adapter
from admet_infer.runner import run_inference
from license_verifier import verify_license


def parse_args():
    p = argparse.ArgumentParser(description="Run multiple ADMET endpoints with one shared cache/QC pass")
    p.add_argument("--properties", required=True, help="Comma-separated property names")
    p.add_argument("--input", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config-dir", help="Directory containing <property>.json files")
    p.add_argument("--encrypted-model-dir", default="encrypted_models")
    p.add_argument("--key-dir", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--disable-attribution", action="store_true")
    p.set_defaults(
        model_source="encrypted",
        batch_size=16,
        seed=42,
        prob_cutoff=0.5,
        attribution_mask_mode="both",
        attribution_max_mols=0,
        residue_groups=None,
        unk_segment_id=0,
    )
    return p.parse_args()


def main():
    args = parse_args()
    properties = [x.strip() for x in args.properties.split(",") if x.strip()]
    if not properties:
        raise ValueError("--properties must contain at least one property")
    for prop in properties:
        verify_license(prop, path=args.license)

    for prop in properties:
        config_path = (
            str(Path(args.config_dir) / f"{prop}.json")
            if args.config_dir else None
        )
        config = apply_overrides(load_property_config(prop, config_path),args)
        adapter = get_property_adapter(prop, config)
        run_inference(args, config, adapter)


if __name__ == "__main__":
    main()
