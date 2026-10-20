#!/usr/bin/env python3
"""python predict_pampa.py \
  --license "$ACCESS_DIR/license.dat" \
  --key-dir "$ACCESS_DIR/endpoint_keys" \
  --encrypted-model-dir encrypted_models \
  --results-dir run_test/pampa \
  --device cuda:0
  """

import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from license_verifier import LicenseError, verify_license


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--license")
    p.add_argument("--key-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--test-csv", default="data/permeability_data/dataset/pampa_fixed_test.csv")
    p.add_argument("--cache-dir", default="data/permeability_data/pampa_cache")
    p.add_argument("--results-dir", default="run_test/results/pampa")
    p.add_argument("--encrypted-model-dir", default="encrypted_models")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    a = p.parse_args()
    license_path = a.license or os.environ.get("INFERENCE_LICENSE", "license.dat")

    try:
        license_data = verify_license("pampa", path=license_path)
    except LicenseError as exc:
        print(f"License error: {exc}", file=sys.stderr)
        return 3

    key = a.key_dir.expanduser().resolve() / "pampa.key"
    encrypted = (ROOT / a.encrypted_model_dir / "pampa.encpkg").resolve()
    test_csv = (ROOT / a.test_csv).resolve()
    cache_dir = (ROOT / a.cache_dir).resolve()
    required = {
        "PAMPA key": key,
        "encrypted PAMPA model": encrypted,
        "test CSV": test_csv,
        "test-only segment bundle": cache_dir / "admet_segment_bundle_test_only.pkl",
        "test-only feature CSV": cache_dir / "peptide_features_model_aligned.csv",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        print("Missing required files:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 4

    print(f"License OK: {license_data['license_id']} (expires {license_data['expires_utc']})")
    command = [
        sys.executable, "-c",
        "from pampa_reproduce.infer_pampa_10cv import main; main()",
        "--root", str(ROOT), "--key-file", str(key), "--bundle", str(encrypted),
        "--test-csv", str(test_csv), "--cache-dir", str(cache_dir),
        "--output-dir", str((ROOT / a.results_dir).resolve()), "--device", a.device,
        "--batch-size", str(a.batch_size), "--workers", str(a.workers),
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
