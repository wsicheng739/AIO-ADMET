#!/usr/bin/env python3
"""
python predict_small_mol.py \
  --license "$ACCESS_DIR/license.dat" \
  --key-dir "$ACCESS_DIR/endpoint_keys" \
  --encrypted-model-dir encrypted_models \
  --results-dir run_test/small_mol_PharmaBench \
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", default="data/small_mol_PharmaBench/test.csv")
    p.add_argument("--results-dir", default="run_test/results/small_mol_PharmaBench")
    p.add_argument("--encrypted-model-dir", default="encrypted_models")
    p.add_argument("--key-dir", required=True)
    p.add_argument("--license", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    license_path = args.license or os.environ.get("INFERENCE_LICENSE", "license.dat")

    try:
        license_data = verify_license("small_mol_PharmaBench", path=license_path)
    except LicenseError as exc:
        print(f"License error: {exc}", file=sys.stderr)
        return 3

    key = Path(args.key_dir).expanduser().resolve() / "small_mol_PharmaBench.key"
    encrypted = ROOT / args.encrypted_model_dir / "small_mol_PharmaBench.encpkg"
    for path, label in ((key, "endpoint key"), (encrypted, "encrypted model")):
        if not path.is_file():
            print(f"Missing {label}: {path}", file=sys.stderr)
            return 4

    print(f"License OK: {license_data['license_id']} (expires {license_data['expires_utc']})")
    command = [
        sys.executable, "-c",
        "from small_mol_infer.infer_human_mic_cl_5cv import main; main()",
        "--root", str(ROOT),
        "--key-file", str(key),
        "--bundle", str(encrypted),
        "--test-csv", str(ROOT / args.test_csv),
        "--output-dir", str(ROOT / args.results_dir),
        "--device", args.device,
        "--license", str(Path(license_path).expanduser().resolve()),
    ]
    env = os.environ.copy()
    search_paths = [str(ROOT / "src"), str(ROOT / "src/pretrain_model")]
    if env.get("PYTHONPATH"):
        search_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(search_paths)
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
