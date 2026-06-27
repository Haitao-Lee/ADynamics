"""
Prepare longitudinal pairs JSON from ADNI-style directory structure.

Expected directory layout:
    data/longitudinal/
    ├── sub-001/
    │   ├── ses-baseline_T1w.nii.gz
    │   ├── ses-followup1_T1w.nii.gz
    │   └── sessions.json       # {"ses-baseline": {"stage": 0, "age": 70, "years": 0}, ...}
    ├── sub-002/
    │   └── ...

Output: longitudinal_pairs.json with all valid (baseline, followup) pairs.

Usage:
    python scripts/prepare_longitudinal_pairs.py \
        --data_dir ./data/longitudinal \
        --output ./core_data/longitudinal_pairs.json
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib


def find_subject_pairs(data_dir: str) -> List[Dict[str, Any]]:
    """Scan directory for subjects with multiple sessions."""
    data_path = Path(data_dir)
    pairs = []

    for subject_dir in sorted(data_path.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name.startswith("."):
            continue

        subject_id = subject_dir.name

        # Look for sessions.json
        sessions_file = subject_dir / "sessions.json"
        if not sessions_file.exists():
            print(f"  [WARN] {subject_id}: no sessions.json, skipping")
            continue

        with open(sessions_file) as f:
            sessions = json.load(f)

        # Find baseline session (years=0 or earliest)
        session_list = []
        for ses_name, ses_info in sessions.items():
            t1_candidates = [
                subject_dir / f"{ses_name}_T1w.nii.gz",
                subject_dir / f"{ses_name}_T1w.nii",
                subject_dir / f"{ses_name}.nii.gz",
            ]
            t1_path = None
            for c in t1_candidates:
                if c.exists():
                    t1_path = str(c)
                    break

            if t1_path is None:
                print(f"  [WARN] {subject_id}/{ses_name}: T1 not found")
                continue

            session_list.append({
                "session": ses_name,
                "t1_path": t1_path,
                "stage": ses_info.get("stage", ses_info.get("label", -1)),
                "age": ses_info.get("age", None),
                "sex": ses_info.get("sex", None),
                "years_from_baseline": ses_info.get("years", ses_info.get("time_years", 0)),
            })

        # Sort by years_from_baseline
        session_list.sort(key=lambda s: s["years_from_baseline"])

        # Find baseline (first session)
        baseline = session_list[0]

        # Create pairs: baseline → each followup
        for followup in session_list[1:]:
            time_years = followup["years_from_baseline"] - baseline["years_from_baseline"]
            if time_years <= 0:
                continue

            pair = {
                "patient_id": subject_id,
                "baseline_t1": baseline["t1_path"],
                "followup_t1": followup["t1_path"],
                "baseline_stage": baseline["stage"],
                "followup_stage": followup["stage"],
                "time_years": time_years,
                "age_baseline": baseline["age"],
                "sex": baseline["sex"],
                "baseline_session": baseline["session"],
                "followup_session": followup["session"],
            }
            pairs.append(pair)

    return pairs


def validate_pairs(pairs: List[Dict], check_exists: bool = True) -> List[Dict]:
    """Validate pairs: check files exist, shapes match."""
    valid = []
    for i, pair in enumerate(pairs):
        if check_exists:
            if not os.path.exists(pair["baseline_t1"]):
                print(f"  [SKIP] {pair['patient_id']}: baseline not found")
                continue
            if not os.path.exists(pair["followup_t1"]):
                print(f"  [SKIP] {pair['patient_id']}: followup not found")

        # Check shapes match
        try:
            shape_base = nib.load(pair["baseline_t1"]).shape
            shape_follow = nib.load(pair["followup_t1"]).shape
            if shape_base != shape_follow:
                print(f"  [WARN] {pair['patient_id']}: shape mismatch "
                      f"{shape_base} vs {shape_follow}")
        except Exception as e:
            print(f"  [WARN] {pair['patient_id']}: {e}")

        valid.append(pair)

    return valid


def main():
    parser = argparse.ArgumentParser(description="Prepare longitudinal pairs JSON")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing subject folders")
    parser.add_argument("--output", type=str, default="./core_data/longitudinal_pairs.json",
                        help="Output JSON path")
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip file existence validation")
    args = parser.parse_args()

    print(f"Scanning: {args.data_dir}")
    pairs = find_subject_pairs(args.data_dir)
    print(f"Found {len(pairs)} longitudinal pairs")

    if not args.no_validate:
        pairs = validate_pairs(pairs)
        print(f"After validation: {len(pairs)} pairs")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved: {args.output}")

    # Print stage transition summary
    transitions = {}
    for p in pairs:
        key = f"{p['baseline_stage']}->{p['followup_stage']}"
        transitions[key] = transitions.get(key, 0) + 1
    print("Stage transitions:")
    for k, v in sorted(transitions.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
