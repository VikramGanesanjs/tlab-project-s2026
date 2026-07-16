"""DICOM → uncompressed NIfTI conversion and phenotype export for Duke Breast MRI."""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pydicom

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_SCAN_TYPES: Tuple[str, ...] = (
    "pre",
    "post_1",
    "post_2",
    "post_3",
    "post_4",
    "T1",
)

DEFAULT_PHENOTYPE_COLUMNS: Tuple[str, ...] = (
    "tumor_location",  # L/R laterality
    "bilateral",
    "stage_t",
    "stage_n",
    "stage_m",
    "tumor_grade_tubule",
    "tumor_grade_nuclear",
    "tumor_grade_mitotic",
    "nottingham_grade",
    "er",
    "pr",
    "her2",
    "mol_subtype",
    "oncotype",
)

PHENOTYPE_SENTINEL = -1.0

_DEFAULT_RAW_ROOT = Path(
    "/common/ganesanv/tlab/data/tcia/duke_breast_cancer_mri"
)
_DEFAULT_OUT_ROOT = Path(
    "/common/ganesanv/tlab/data/tcia/duke_breast_cancer_mri_nifti"
)
_DEFAULT_MAPPING_XLSX = _DEFAULT_RAW_ROOT / "Breast-Cancer-MRI-filepath_filename-mapping.xlsx"
_DEFAULT_CLINICAL_XLSX = _DEFAULT_RAW_ROOT / "Clinical_and_Other_Features.xlsx"

_ORIGINAL_PATH_RE = re.compile(
    r"DICOM_Images/(Breast_MRI_\d+)/([^/]+)/([^/]+)$"
)
_SLICE_IDX_RE = re.compile(r"_(\d+)\.dcm$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Scan resolution
# ---------------------------------------------------------------------------


def resolve_scan(scan: Union[str, int]) -> str:
    """Resolve a scan name or ordinal index to a canonical scan type string."""
    if isinstance(scan, int):
        if scan < 0 or scan >= len(CANONICAL_SCAN_TYPES):
            raise ValueError(
                f"scan index {scan} out of range [0, {len(CANONICAL_SCAN_TYPES) - 1}]"
            )
        return CANONICAL_SCAN_TYPES[scan]
    if scan not in CANONICAL_SCAN_TYPES:
        raise ValueError(
            f"Unknown scan type {scan!r}. Expected one of {CANONICAL_SCAN_TYPES}"
        )
    return scan


# ---------------------------------------------------------------------------
# Mapping table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappingSlice:
    patient_id: str
    scan_type: str
    sop_instance_uid: str
    classic_path: str
    series_sort: Optional[str]
    slice_index: int  # 1-based from filename when available


def load_mapping_table(
    mapping_xlsx: Union[str, Path] = _DEFAULT_MAPPING_XLSX,
) -> pd.DataFrame:
    """Load the TCIA filepath/filename mapping spreadsheet."""
    df = pd.read_excel(mapping_xlsx, engine="openpyxl")
    required = {
        "sop_instance_UID",
        "original_path_and_filename",
        "classic_path",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mapping xlsx missing columns: {sorted(missing)}")
    return df


def _parse_original_path(path: str) -> Optional[Tuple[str, str, str]]:
    m = _ORIGINAL_PATH_RE.search(str(path).replace("\\", "/"))
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def _slice_index_from_name(filename: str, series_sort: Any) -> int:
    if isinstance(series_sort, str) and series_sort:
        m = re.search(r"(\d+)", series_sort)
        if m:
            return int(m.group(1))
    m = _SLICE_IDX_RE.search(filename)
    if m:
        return int(m.group(1))
    return -1


def build_series_directory_map(
    mapping_df: pd.DataFrame,
    raw_root: Union[str, Path],
) -> Dict[Tuple[str, str], Path]:
    """Map (patient_id, scan_type) -> on-disk DICOM series directory.

    classic_path uses bare UIDs; on-disk series folders are prefixed with ``MR_``.
    """
    raw_root = Path(raw_root)
    series_dirs: Dict[Tuple[str, str], Path] = {}
    for row in mapping_df.itertuples(index=False):
        parsed = _parse_original_path(getattr(row, "original_path_and_filename"))
        if parsed is None:
            continue
        patient_id, scan_type, _fname = parsed
        key = (patient_id, scan_type)
        if key in series_dirs:
            continue
        classic = str(getattr(row, "classic_path", "")).replace("\\", "/")
        parts = classic.split("/")
        # Duke-Breast-Cancer-MRI / Breast_MRI_XXX / studyUID / seriesUID / file
        if len(parts) < 5:
            continue
        study_uid, series_uid = parts[2], parts[3]
        candidate = raw_root / patient_id / study_uid / f"MR_{series_uid}"
        if not candidate.is_dir():
            # Some extractions omit the MR_ prefix
            alt = raw_root / patient_id / study_uid / series_uid
            candidate = alt if alt.is_dir() else candidate
        series_dirs[key] = candidate
    return series_dirs


def mapping_slices_for_series(
    mapping_df: pd.DataFrame,
    patient_id: str,
    scan_type: str,
) -> List[MappingSlice]:
    """Return mapping rows for one series, sorted by axial slice index."""
    slices: List[MappingSlice] = []
    for row in mapping_df.itertuples(index=False):
        parsed = _parse_original_path(getattr(row, "original_path_and_filename"))
        if parsed is None:
            continue
        pid, scan, fname = parsed
        if pid != patient_id or scan != scan_type:
            continue
        series_sort = getattr(row, "series_sort", None)
        if pd.isna(series_sort):
            series_sort = None
        slices.append(
            MappingSlice(
                patient_id=pid,
                scan_type=scan,
                sop_instance_uid=str(getattr(row, "sop_instance_UID")),
                classic_path=str(getattr(row, "classic_path")),
                series_sort=series_sort,
                slice_index=_slice_index_from_name(fname, series_sort),
            )
        )
    slices.sort(key=lambda s: (s.slice_index if s.slice_index >= 0 else 10**9, s.sop_instance_uid))
    return slices


# ---------------------------------------------------------------------------
# DICOM sorting / reading
# ---------------------------------------------------------------------------


@dataclass
class DicomSliceMeta:
    path: Path
    instance_number: int
    slice_location: float
    sop_instance_uid: str
    image_position: Optional[Tuple[float, float, float]]
    image_orientation: Optional[Tuple[float, ...]]
    pixel_spacing: Optional[Tuple[float, float]]
    slice_thickness: Optional[float]
    rows: Optional[int]
    cols: Optional[int]


def _read_dicom_header(path: Union[str, Path]) -> DicomSliceMeta:
    path = Path(path)
    ds = pydicom.dcmread(
        str(path),
        stop_before_pixels=True,
        specific_tags=[
            "InstanceNumber",
            "SliceLocation",
            "SOPInstanceUID",
            "ImagePositionPatient",
            "ImageOrientationPatient",
            "PixelSpacing",
            "SliceThickness",
            "Rows",
            "Columns",
        ],
    )
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    spacing = getattr(ds, "PixelSpacing", None)
    sl = getattr(ds, "SliceLocation", None)
    if sl is None and ipp is not None:
        sl = float(ipp[2])
    return DicomSliceMeta(
        path=path,
        instance_number=int(getattr(ds, "InstanceNumber", -1)),
        slice_location=float(sl) if sl is not None else float("nan"),
        sop_instance_uid=str(getattr(ds, "SOPInstanceUID", "")),
        image_position=tuple(float(x) for x in ipp) if ipp is not None else None,
        image_orientation=tuple(float(x) for x in iop) if iop is not None else None,
        pixel_spacing=tuple(float(x) for x in spacing) if spacing is not None else None,
        slice_thickness=float(ds.SliceThickness)
        if getattr(ds, "SliceThickness", None) is not None
        else None,
        rows=int(ds.Rows) if getattr(ds, "Rows", None) is not None else None,
        cols=int(ds.Columns) if getattr(ds, "Columns", None) is not None else None,
    )


def sorted_dicom_paths(
    series_dir: Union[str, Path],
    mapping_slices: Optional[Sequence[MappingSlice]] = None,
) -> List[DicomSliceMeta]:
    """Sort DICOM files in a series directory along the axial axis.

    Prefers mapping SOP order when ``mapping_slices`` is provided and covers
    the series; otherwise sorts by ``InstanceNumber`` from DICOM headers
    (pixels are never read).
    """
    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM files in {series_dir}")

    headers = [_read_dicom_header(f) for f in files]
    by_sop = {h.sop_instance_uid: h for h in headers}

    if mapping_slices:
        ordered: List[DicomSliceMeta] = []
        missing = 0
        for ms in mapping_slices:
            h = by_sop.get(ms.sop_instance_uid)
            if h is None:
                missing += 1
                continue
            ordered.append(h)
        if ordered and missing == 0 and len(ordered) == len(headers):
            return ordered
        if ordered and len(ordered) >= max(1, int(0.9 * len(headers))):
            logger.warning(
                "Partial mapping match in %s (%d/%d); using mapping order for matched SOPs",
                series_dir,
                len(ordered),
                len(headers),
            )
            return ordered

    headers.sort(key=lambda h: (h.instance_number, h.slice_location, h.path.name))
    return headers


def _filter_by_slice_location(
    headers: Sequence[DicomSliceMeta],
    lower_bound: Optional[float],
    upper_bound: Optional[float],
) -> List[DicomSliceMeta]:
    if lower_bound is None and upper_bound is None:
        return list(headers)
    kept: List[DicomSliceMeta] = []
    for h in headers:
        sl = h.slice_location
        if np.isnan(sl):
            continue
        if lower_bound is not None and sl < lower_bound:
            continue
        if upper_bound is not None and sl > upper_bound:
            continue
        kept.append(h)
    return kept


def _build_affine(headers: Sequence[DicomSliceMeta]) -> np.ndarray:
    """Construct a RAS-ish affine from the first slice DICOM geometry."""
    affine = np.eye(4, dtype=np.float64)
    h0 = headers[0]
    if h0.pixel_spacing is None or h0.image_orientation is None or h0.image_position is None:
        # Fallback: identity with 1mm isotropic and Z from slice locations
        if len(headers) >= 2 and not np.isnan(headers[0].slice_location):
            dz = abs(headers[1].slice_location - headers[0].slice_location) or 1.0
            affine[2, 2] = dz
        return affine

    row_cos = np.array(h0.image_orientation[0:3], dtype=np.float64)
    col_cos = np.array(h0.image_orientation[3:6], dtype=np.float64)
    # DICOM patient coordinates: X <- col, Y <- row direction spacing
    px, py = h0.pixel_spacing
    affine[:3, 0] = row_cos * px
    affine[:3, 1] = col_cos * py
    if len(headers) >= 2 and h0.image_position is not None and headers[1].image_position is not None:
        delta = np.array(headers[1].image_position, dtype=np.float64) - np.array(
            h0.image_position, dtype=np.float64
        )
        affine[:3, 2] = delta
    elif h0.slice_thickness:
        normal = np.cross(row_cos, col_cos)
        affine[:3, 2] = normal * float(h0.slice_thickness)
    affine[:3, 3] = np.array(h0.image_position, dtype=np.float64)
    return affine


def _load_pixel_array(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert_series_to_nifti(
    series_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    scan_type: str,
    patient_id: str,
    mapping_slices: Optional[Sequence[MappingSlice]] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert one DICOM series to uncompressed ``volume.nii`` + ``meta.json``."""
    import nibabel as nib

    series_dir = Path(series_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = sorted_dicom_paths(series_dir, mapping_slices=mapping_slices)
    headers = _filter_by_slice_location(headers, lower_bound, upper_bound)
    if not headers:
        raise RuntimeError(
            f"No slices left after bounds filter for {patient_id}/{scan_type} in {series_dir}"
        )

    slices = [_load_pixel_array(h.path) for h in headers]
    volume = np.stack(slices, axis=-1)  # (H, W, Z)
    affine = _build_affine(headers)
    nii_path = out_dir / "volume.nii"
    nib.Nifti1Image(volume.astype(np.float32), affine).to_filename(str(nii_path))

    meta = {
        "patient_id": patient_id,
        "scan_type": scan_type,
        "series_dir": str(series_dir),
        "n_slices": int(volume.shape[-1]),
        "shape": list(volume.shape),
        "dtype": "float32",
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "slice_locations": [h.slice_location for h in headers],
        "instance_numbers": [h.instance_number for h in headers],
        "sop_instance_uids": [h.sop_instance_uid for h in headers],
        "affine": affine.tolist(),
        "volume_path": "volume.nii",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _convert_one_job(args: Tuple) -> Tuple[str, str, Optional[Dict[str, Any]], Optional[str]]:
    (
        patient_id,
        scan_type,
        series_dir,
        out_dir,
        mapping_rows,
        lower_bound,
        upper_bound,
    ) = args
    try:
        mapping_slices = [
            MappingSlice(**m) if isinstance(m, dict) else m for m in (mapping_rows or [])
        ]
        meta = convert_series_to_nifti(
            series_dir,
            out_dir,
            scan_type=scan_type,
            patient_id=patient_id,
            mapping_slices=mapping_slices or None,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        return patient_id, scan_type, meta, None
    except Exception as exc:  # noqa: BLE001 — collect per-series errors
        return patient_id, scan_type, None, f"{type(exc).__name__}: {exc}"


def convert_duke_dataset(
    raw_root: Union[str, Path] = _DEFAULT_RAW_ROOT,
    out_root: Union[str, Path] = _DEFAULT_OUT_ROOT,
    mapping_xlsx: Union[str, Path] = _DEFAULT_MAPPING_XLSX,
    clinical_xlsx: Union[str, Path] = _DEFAULT_CLINICAL_XLSX,
    scan_types: Optional[Sequence[str]] = None,
    patient_ids: Optional[Sequence[str]] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    workers: int = 4,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Convert selected patients/scans to NIfTI and write index + phenotypes."""
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "index").mkdir(exist_ok=True)
    (out_root / "labels").mkdir(exist_ok=True)

    scan_types = list(scan_types) if scan_types else list(CANONICAL_SCAN_TYPES)
    for st in scan_types:
        if st not in CANONICAL_SCAN_TYPES:
            raise ValueError(f"Unknown scan type {st!r}")

    logger.info("Loading mapping table from %s", mapping_xlsx)
    mapping_df = load_mapping_table(mapping_xlsx)
    series_dirs = build_series_directory_map(mapping_df, raw_root)

    # Restrict patient list
    all_patients = sorted({pid for pid, _ in series_dirs})
    if patient_ids is not None:
        wanted = set(patient_ids)
        all_patients = [p for p in all_patients if p in wanted]

    # Pre-group mapping slices for fast worker args (dict-serializable)
    mapping_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for (pid, scan), _dir in series_dirs.items():
        if scan not in scan_types:
            continue
        if patient_ids is not None and pid not in set(patient_ids):
            continue
        ms = mapping_slices_for_series(mapping_df, pid, scan)
        mapping_by_key[(pid, scan)] = [
            {
                "patient_id": m.patient_id,
                "scan_type": m.scan_type,
                "sop_instance_uid": m.sop_instance_uid,
                "classic_path": m.classic_path,
                "series_sort": m.series_sort,
                "slice_index": m.slice_index,
            }
            for m in ms
        ]

    jobs = []
    for pid in all_patients:
        for scan in scan_types:
            key = (pid, scan)
            if key not in series_dirs:
                continue
            series_dir = series_dirs[key]
            if not series_dir.is_dir():
                logger.warning("Missing series dir %s", series_dir)
                continue
            out_dir = out_root / pid / scan
            if skip_existing and (out_dir / "volume.nii").is_file() and (out_dir / "meta.json").is_file():
                continue
            jobs.append(
                (
                    pid,
                    scan,
                    str(series_dir),
                    str(out_dir),
                    mapping_by_key.get(key, []),
                    lower_bound,
                    upper_bound,
                )
            )

    logger.info("Converting %d series with %d workers", len(jobs), workers)
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    # Also load already-converted series into the index
    for pid in all_patients:
        for scan in scan_types:
            meta_path = out_root / pid / scan / "meta.json"
            if meta_path.is_file():
                with open(meta_path) as f:
                    results.append(json.load(f))

    if jobs:
        if workers <= 1:
            for job in jobs:
                pid, scan, meta, err = _convert_one_job(job)
                if err:
                    errors.append({"patient_id": pid, "scan_type": scan, "error": err})
                    logger.error("Failed %s/%s: %s", pid, scan, err)
                elif meta:
                    # Replace any stale entry
                    results = [
                        r
                        for r in results
                        if not (r.get("patient_id") == pid and r.get("scan_type") == scan)
                    ]
                    results.append(meta)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_convert_one_job, job): job for job in jobs}
                for fut in as_completed(futs):
                    pid, scan, meta, err = fut.result()
                    if err:
                        errors.append({"patient_id": pid, "scan_type": scan, "error": err})
                        logger.error("Failed %s/%s: %s", pid, scan, err)
                    elif meta:
                        results = [
                            r
                            for r in results
                            if not (r.get("patient_id") == pid and r.get("scan_type") == scan)
                        ]
                        results.append(meta)

    series_index = {
        "scan_types": scan_types,
        "series": [
            {
                "patient_id": r["patient_id"],
                "scan_type": r["scan_type"],
                "rel_dir": f"{r['patient_id']}/{r['scan_type']}",
                "n_slices": r["n_slices"],
                "volume_path": f"{r['patient_id']}/{r['scan_type']}/volume.nii",
                "meta_path": f"{r['patient_id']}/{r['scan_type']}/meta.json",
            }
            for r in sorted(results, key=lambda x: (x["patient_id"], x["scan_type"]))
        ],
        "errors": errors,
    }
    with open(out_root / "index" / "series_index.json", "w") as f:
        json.dump(series_index, f, indent=2)

    manifest = {
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "mapping_xlsx": str(mapping_xlsx),
        "clinical_xlsx": str(clinical_xlsx),
        "scan_types": scan_types,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "n_series": len(series_index["series"]),
        "n_errors": len(errors),
    }
    with open(out_root / "index" / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    build_phenotype_json(clinical_xlsx, out_root / "labels")
    logger.info(
        "Wrote %d series to %s (%d errors)",
        len(series_index["series"]),
        out_root,
        len(errors),
    )
    return {"manifest": manifest, "series_index": series_index}


# ---------------------------------------------------------------------------
# Phenotypes
# ---------------------------------------------------------------------------

_CLINICAL_COL_MAP = {
    "tumor_location": "Tumor Location",
    "bilateral": "Bilateral Information",
    "stage_t": "Staging(Tumor Size)# [T]",
    "stage_n": "Staging(Nodes)#(Nx replaced by -1)[N]",
    "stage_m": "Staging(Metastasis)#(Mx -replaced by -1)[M]",
    "tumor_grade_tubule": "Tumor Grade",
    "tumor_grade_nuclear": "Unnamed: 32",
    "tumor_grade_mitotic": "Unnamed: 33",
    "nottingham_grade": "Nottingham grade",
    "er": "ER",
    "pr": "PR",
    "her2": "HER2",
    "mol_subtype": "Mol Subtype",
    "oncotype": "Oncotype score",
}

_RAW_EXTRA_COLS = (
    "Position",
    "Histologic type",
)


def _to_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return PHENOTYPE_SENTINEL
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return PHENOTYPE_SENTINEL
        return float(value)
    s = str(value).strip()
    if s == "" or s.upper() in {"NA", "NC", "NP", "NAN", "NONE"}:
        return PHENOTYPE_SENTINEL
    # Laterality
    if s.upper() in {"L", "LEFT"}:
        return 0.0
    if s.upper() in {"R", "RIGHT"}:
        return 1.0
    try:
        return float(s)
    except ValueError:
        return PHENOTYPE_SENTINEL


def build_phenotype_json(
    clinical_xlsx: Union[str, Path] = _DEFAULT_CLINICAL_XLSX,
    labels_dir: Union[str, Path] = _DEFAULT_OUT_ROOT / "labels",
    columns: Sequence[str] = DEFAULT_PHENOTYPE_COLUMNS,
) -> Dict[str, Any]:
    """Encode clinical spreadsheet into phenotypes.json + schema."""
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(clinical_xlsx, sheet_name="Data", header=1, engine="openpyxl")
    # Row after header is an encoding legend with non-patient IDs
    df = df[df["Patient ID"].astype(str).str.startswith("Breast_MRI")].copy()

    phenotypes: Dict[str, Any] = {}
    for _, row in df.iterrows():
        pid = str(row["Patient ID"])
        vector = []
        raw: Dict[str, Any] = {"patient_id": pid}
        for key in columns:
            src = _CLINICAL_COL_MAP[key]
            val = row[src] if src in row.index else np.nan
            raw[key] = None if (isinstance(val, float) and np.isnan(val)) else (
                val.item() if isinstance(val, np.generic) else val
            )
            # JSON-serialize numpy types / NaN
            if isinstance(raw[key], float) and np.isnan(raw[key]):
                raw[key] = None
            vector.append(_to_float(val))
        for extra in _RAW_EXTRA_COLS:
            if extra in row.index:
                ev = row[extra]
                raw[extra] = None if pd.isna(ev) else (ev.item() if isinstance(ev, np.generic) else ev)
        phenotypes[pid] = {
            "vector": vector,
            "raw": raw,
        }

    schema = {
        "columns": list(columns),
        "source_columns": {k: _CLINICAL_COL_MAP[k] for k in columns},
        "sentinel": PHENOTYPE_SENTINEL,
        "encodings": {
            "tumor_location": {"L": 0.0, "R": 1.0},
            "bilateral": "0=no, 1=yes",
            "er": "0=neg, 1=pos",
            "pr": "0=neg, 1=pos",
            "her2": "0=neg, 1=pos, 2=borderline",
            "mol_subtype": "0=luminal-like, 1=ER/PR+ HER2+, 2=her2, 3=trip neg",
            "nottingham_grade": "1=low, 2=intermediate, 3=high",
            "missing": PHENOTYPE_SENTINEL,
        },
        "n_patients": len(phenotypes),
    }

    def _json_default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if pd.isna(o):
            return None
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(labels_dir / "phenotypes.json", "w") as f:
        json.dump(phenotypes, f, indent=2, default=_json_default)
    with open(labels_dir / "phenotype_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    logger.info("Wrote phenotypes for %d patients to %s", len(phenotypes), labels_dir)
    return schema


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Duke Breast MRI → NIfTI converter / utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="Convert DICOM series to uncompressed NIfTI")
    c.add_argument("--raw-root", type=Path, default=_DEFAULT_RAW_ROOT)
    c.add_argument("--out-root", type=Path, default=_DEFAULT_OUT_ROOT)
    c.add_argument("--mapping-xlsx", type=Path, default=_DEFAULT_MAPPING_XLSX)
    c.add_argument("--clinical-xlsx", type=Path, default=_DEFAULT_CLINICAL_XLSX)
    c.add_argument(
        "--scan-types",
        nargs="+",
        default=list(CANONICAL_SCAN_TYPES),
        help="Scan types to convert",
    )
    c.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Optional patient IDs (e.g. Breast_MRI_001). Default: all",
    )
    c.add_argument("--lower-bound", type=float, default=None)
    c.add_argument("--upper-bound", type=float, default=None)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--no-skip-existing", action="store_true")

    ph = sub.add_parser("phenotypes", help="Rebuild phenotype JSON only")
    ph.add_argument("--clinical-xlsx", type=Path, default=_DEFAULT_CLINICAL_XLSX)
    ph.add_argument("--labels-dir", type=Path, default=_DEFAULT_OUT_ROOT / "labels")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.cmd == "convert":
        convert_duke_dataset(
            raw_root=args.raw_root,
            out_root=args.out_root,
            mapping_xlsx=args.mapping_xlsx,
            clinical_xlsx=args.clinical_xlsx,
            scan_types=args.scan_types,
            patient_ids=args.patients,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound,
            workers=args.workers,
            skip_existing=not args.no_skip_existing,
        )
    elif args.cmd == "phenotypes":
        build_phenotype_json(args.clinical_xlsx, args.labels_dir)
    else:
        raise SystemExit(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
