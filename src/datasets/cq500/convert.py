"""Convert CQ500 DICOM series in place to NIfTI volumes.

Each ``CQ500CT*/<scan name>/`` directory is converted to an uncompressed
``volume.nii`` in the same directory.  A small ``meta.json`` sidecar records
the spatial metadata used by the dataset loaders, including slice spacing.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("/common/ganesanv/tlab/data/cq500")
VOLUME_NAME = "volume.nii"
META_NAME = "meta.json"


def discover_series(
    root: Union[str, Path] = DEFAULT_ROOT,
    patient_ids: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Path]]:
    """Return ``(patient_id, scan_directory)`` pairs containing DICOM files."""
    root_path = Path(root).expanduser().resolve()
    requested = {str(patient_id) for patient_id in patient_ids} if patient_ids else None
    series: List[Tuple[str, Path]] = []
    for patient_dir in sorted(root_path.glob("CQ500CT*")):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        if requested is not None and patient_id not in requested:
            continue
        for scan_dir in sorted(patient_dir.iterdir()):
            if not scan_dir.is_dir():
                continue
            if any(scan_dir.glob("*.dcm")):
                series.append((patient_id, scan_dir))
    return series


def convert_series_to_nifti(
    series_dir: Union[str, Path],
    *,
    patient_id: Optional[str] = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Convert one CQ500 DICOM series and write its metadata sidecar.

    ``dicom2nifti`` is intentionally imported lazily so importing this module
    does not require the conversion dependency.
    """
    series_path = Path(series_dir).expanduser().resolve()
    if not series_path.is_dir():
        raise FileNotFoundError(f"Missing DICOM series directory: {series_path}")
    dicom_paths = sorted(series_path.glob("*.dcm"))
    if not dicom_paths:
        raise RuntimeError(f"No .dcm files in {series_path}")

    output_path = series_path / VOLUME_NAME
    meta_path = series_path / META_NAME
    if skip_existing and output_path.is_file() and meta_path.is_file():
        with meta_path.open() as handle:
            return json.load(handle)

    try:
        import dicom2nifti
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "CQ500 conversion requires dicom2nifti and nibabel. "
            "Install them before running convert.py."
        ) from exc

    # ``reorient_nifti=True`` gives all output volumes a consistent orientation.
    dicom2nifti.dicom_series_to_nifti(
        str(series_path), str(output_path), reorient_nifti=True
    )
    image = nib.load(str(output_path), mmap="r")
    if len(image.shape) != 3:
        raise ValueError(
            f"Expected a 3-D NIfTI from {series_path}, got shape {image.shape}"
        )
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    resolved_patient_id = patient_id or series_path.parent.name
    metadata: Dict[str, Any] = {
        "patient_id": resolved_patient_id,
        "scan_name": series_path.name,
        "series_dir": str(series_path),
        "volume_path": VOLUME_NAME,
        "n_dicom_files": len(dicom_paths),
        "shape": [int(value) for value in image.shape],
        "voxel_spacing_mm": list(spacing),
        "slice_spacing_mm": float(spacing[2]),
    }
    with meta_path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata


def _convert_job(job: Tuple[str, str, bool]) -> Tuple[str, str, Optional[Dict[str, Any]], Optional[str]]:
    patient_id, series_dir, skip_existing = job
    try:
        metadata = convert_series_to_nifti(
            series_dir, patient_id=patient_id, skip_existing=skip_existing
        )
        return patient_id, series_dir, metadata, None
    except Exception as exc:  # noqa: BLE001 - report all failed series together.
        return patient_id, series_dir, None, f"{type(exc).__name__}: {exc}"


def convert_cq500_dataset(
    root: Union[str, Path] = DEFAULT_ROOT,
    *,
    patient_ids: Optional[Sequence[str]] = None,
    workers: int = 1,
    skip_existing: bool = True,
    fail_on_error: bool = False,
) -> Dict[str, Any]:
    """Convert all selected CQ500 series and return a conversion report."""
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    series = discover_series(root, patient_ids=patient_ids)
    if not series:
        raise RuntimeError(f"No CQ500 DICOM series found under {Path(root).resolve()}")

    jobs = [(patient_id, str(scan_dir), bool(skip_existing)) for patient_id, scan_dir in series]
    converted: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    if workers == 1:
        results: Iterable[Tuple[str, str, Optional[Dict[str, Any]], Optional[str]]] = (
            _convert_job(job) for job in jobs
        )
        for patient_id, series_dir, metadata, error in results:
            if error:
                errors.append({"patient_id": patient_id, "series_dir": series_dir, "error": error})
            elif metadata is not None:
                converted.append(metadata)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_convert_job, job) for job in jobs]
            for future in as_completed(futures):
                patient_id, series_dir, metadata, error = future.result()
                if error:
                    errors.append({"patient_id": patient_id, "series_dir": series_dir, "error": error})
                elif metadata is not None:
                    converted.append(metadata)

    report: Dict[str, Any] = {
        "root": str(Path(root).resolve()),
        "series_found": len(series),
        "series_converted": len(converted),
        "errors": sorted(errors, key=lambda item: (item["patient_id"], item["series_dir"])),
    }
    logger.info(
        "CQ500 conversion finished: %d/%d series converted, %d errors",
        report["series_converted"], report["series_found"], len(errors),
    )
    if errors and fail_on_error:
        raise RuntimeError(f"CQ500 conversion failed for {len(errors)} series: {errors[:3]}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--patient-id", action="append", dest="patient_ids")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="re-convert existing volumes")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    report = convert_cq500_dataset(
        args.root,
        patient_ids=args.patient_ids,
        workers=args.workers,
        skip_existing=not args.overwrite,
        fail_on_error=args.fail_on_error,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
