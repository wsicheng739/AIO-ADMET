#!/usr/bin/env python3
"""Licensed entry point for the four-endpoint inference workflow.

ACCESS_DIR=/path/to/key/folder
python predict_five_external_cases.py \
  --license "$ACCESS_DIR/license.dat" \
  --key-dir "$ACCESS_DIR/endpoint_keys" \
  --encrypted-model-dir encrypted_models \
  --results-dir run_test/external_cases \
  --device cuda:0
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from license_verifier import LicenseError, verify_license

ENDPOINTS = ("caco2", "unbound_fraction", "terminal_t_half", "bioavailability")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/external_five_case/input.csv")
    parser.add_argument("--cache-dir", default="data/external_five_case/cache")
    parser.add_argument("--results-dir", default="run_test/results")
    parser.add_argument("--encrypted-model-dir", default="encrypted_models")
    parser.add_argument("--key-dir", required=True)
    parser.add_argument("--license", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--enable-attribution", action="store_true")
    args = parser.parse_args()
    license_path = args.license or os.environ.get("INFERENCE_LICENSE", "license.dat")

    try:
        for endpoint in ENDPOINTS:
            license_data = verify_license(endpoint, path=license_path)
    except LicenseError as exc:
        print(f"License error: {exc}", file=sys.stderr)
        return 3

    print(f"License OK: {license_data['license_id']} (expires {license_data['expires_utc']})")
    command = [
        sys.executable, str(ROOT / "run_inference.py"),
        "--properties", ",".join(ENDPOINTS),
        "--input", str(ROOT / args.input),
        "--cache-dir", str(ROOT / args.cache_dir),
        "--output-dir", str(ROOT / args.results_dir),
        "--config-dir", str(ROOT / "configs" / "properties"),
        "--encrypted-model-dir", str(ROOT / args.encrypted_model_dir),
        "--device", args.device,
        "--license", str(Path(license_path).expanduser().resolve()),
    ]
    if not args.enable_attribution:
        command.append("--disable-attribution")
    command.extend(["--key-dir", str(Path(args.key_dir).expanduser().resolve())])
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

