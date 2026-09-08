"""Create the read-only ADNI patient-to-NIfTI manifest before a training run."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from .dataset import DEFAULT_MANIFEST_NAME, DEFAULT_ROOT, _normalize_image_id

logger = logging.getLogger(__name__)


def write_nii_manifest(root: Path, manifest_path: Path) -> None:
    """Walk ADNI once and persist NIfTI paths grouped by patient directory."""
    patients: dict[str, list[dict[str, str]]] = {}
    for volume_path in sorted((*root.rglob("*.nii"), *root.rglob("*.nii.gz"))):
        image_id = _normalize_image_id(volume_path.parent.name)
        relative = volume_path.relative_to(root)
        patient_id = relative.parts[0]
        patients.setdefault(patient_id, []).append(
            {"image_id": image_id, "path": str(relative)}
        )
    if not patients:
        raise RuntimeError(f"No .nii or .nii.gz files found under {root}")
    payload = {
        "schema_version": 1,
        "root": str(root),
        "patients": {
            patient_id: sorted(entries, key=lambda entry: entry["image_id"])
            for patient_id, entries in sorted(patients.items())
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=manifest_path.parent,
        prefix=manifest_path.name + ".", suffix=".tmp", delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(manifest_path)
    logger.info("Created ADNI NIfTI manifest %s (%d patients)", manifest_path, len(patients))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest path (default: <root>/adni_nii_manifest.json)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Path:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / DEFAULT_MANIFEST_NAME
    )
    write_nii_manifest(root, output)
    return output


if __name__ == "__main__":
    main()
