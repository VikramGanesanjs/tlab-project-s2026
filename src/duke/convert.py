"""DICOM → uncompressed NIfTI conversion and phenotype export for Duke Breast MRI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
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
    "/common/ganesanv/tlab/data/tcia/duke_breast_cancer_processed"
)
_DEFAULT_MAPPING_XLSX = _DEFAULT_RAW_ROOT / "Breast-Cancer-MRI-filepath_filename-mapping.xlsx"
_DEFAULT_CLINICAL_XLSX = _DEFAULT_RAW_ROOT / "Clinical_and_Other_Features.xlsx"
_DEFAULT_BREASTDIVIDER_CSV = _DEFAULT_RAW_ROOT / "breastdivider_id_mapping.csv"
_DEFAULT_BREASTDIVIDER_BATCHES: Tuple[Path, ...] = (
    _DEFAULT_RAW_ROOT / "labelsTr_batch1",
    _DEFAULT_RAW_ROOT / "labelsTr_batch2",
)
_BREAST_DIVIDER_MASK_NAME = "breast_divider.nii.gz"
_LEFT_BREAST_VOLUME_NAME = "left.nii"
_RIGHT_BREAST_VOLUME_NAME = "right.nii"
_BREAST_DIVIDER_ROTATION_BY_ROUNDED_IOP: Dict[Tuple[int, int, int, int, int, int], int] = {
    # Positive row/column direction cosines: rotate fixed-orientation masks 90° CCW.
    (1, 0, 0, 0, 1, 0): 1,
    # Negative row/column direction cosines: rotate fixed-orientation masks 90° CW.
    (-1, 0, 0, 0, -1, 0): -1,
    # Rare Duke case (Breast_MRI_127): negative row direction, positive column direction.
    (-1, 0, 0, 0, 1, 0): 1,
}

_ORIGINAL_PATH_RE = re.compile(
    r"DICOM_Images/(Breast_MRI_\d+)/([^/]+)/([^/]+)$"
)
_SLICE_IDX_RE = re.compile(r"_(\d+)\.dcm$", re.IGNORECASE)
_DUKE_BREASTDIVIDER_ID_RE = re.compile(r"^Duke_Breast_MRI_(\d+)_(.+)$")
_DESC_SERIES_RE = re.compile(r"/(\d+\.\d{6}-.*)/[^/]+$")


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


def build_mapping_slices_index(
    mapping_df: pd.DataFrame,
    *,
    scan_types: Optional[Sequence[str]] = None,
    patient_ids: Optional[Sequence[str]] = None,
) -> Dict[Tuple[str, str], List[MappingSlice]]:
    """One-pass index of mapping rows keyed by ``(patient_id, scan_type)``.

    Avoids the O(n_series × n_rows) cost of calling
    :func:`mapping_slices_for_series` repeatedly.
    """
    scan_filter = set(scan_types) if scan_types is not None else None
    patient_filter = set(patient_ids) if patient_ids is not None else None
    by_key: Dict[Tuple[str, str], List[MappingSlice]] = {}

    for row in mapping_df.itertuples(index=False):
        parsed = _parse_original_path(getattr(row, "original_path_and_filename"))
        if parsed is None:
            continue
        pid, scan, fname = parsed
        if scan_filter is not None and scan not in scan_filter:
            continue
        if patient_filter is not None and pid not in patient_filter:
            continue
        series_sort = getattr(row, "series_sort", None)
        if pd.isna(series_sort):
            series_sort = None
        key = (pid, scan)
        by_key.setdefault(key, []).append(
            MappingSlice(
                patient_id=pid,
                scan_type=scan,
                sop_instance_uid=str(getattr(row, "sop_instance_UID")),
                classic_path=str(getattr(row, "classic_path")),
                series_sort=series_sort,
                slice_index=_slice_index_from_name(fname, series_sort),
            )
        )

    for slices in by_key.values():
        slices.sort(
            key=lambda s: (s.slice_index if s.slice_index >= 0 else 10**9, s.sop_instance_uid)
        )
    return by_key


def mapping_slices_for_series(
    mapping_df: pd.DataFrame,
    patient_id: str,
    scan_type: str,
) -> List[MappingSlice]:
    """Return mapping rows for one series, sorted by axial slice index."""
    return build_mapping_slices_index(
        mapping_df,
        scan_types=[scan_type],
        patient_ids=[patient_id],
    ).get((patient_id, scan_type), [])


def mapping_slices_as_dicts(
    slices: Sequence[MappingSlice],
) -> List[Dict[str, Any]]:
    """Serialize :class:`MappingSlice` objects for process-pool worker args."""
    return [
        {
            "patient_id": m.patient_id,
            "scan_type": m.scan_type,
            "sop_instance_uid": m.sop_instance_uid,
            "classic_path": m.classic_path,
            "series_sort": m.series_sort,
            "slice_index": m.slice_index,
        }
        for m in slices
    ]

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


def _bbox_from_mask(mask: np.ndarray, margin: int = 0) -> Tuple[Tuple[int, int], ...]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot crop an empty breast mask")
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    if margin > 0:
        lo = np.maximum(lo - margin, 0)
        hi = np.minimum(hi + margin, np.array(mask.shape))
    return tuple((int(a), int(b)) for a, b in zip(lo, hi))


def _crop_affine(affine: np.ndarray, bbox: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    crop_affine = np.array(affine, dtype=np.float64, copy=True)
    start = np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1.0], dtype=np.float64)
    crop_affine[:3, 3] = (np.asarray(affine, dtype=np.float64) @ start)[:3]
    return crop_affine


def _breast_label_sides(
    mask_data: np.ndarray,
    affine: np.ndarray,
) -> Dict[str, float]:
    labels = sorted(float(v) for v in np.unique(mask_data) if v > 0)
    if len(labels) != 2:
        raise ValueError(f"Expected exactly 2 positive breast mask labels; found {labels}")

    centers: List[Tuple[float, float, float]] = []
    for label in labels:
        coords = np.argwhere(mask_data == label)
        if coords.size == 0:
            continue
        center_ijk = coords.mean(axis=0)
        world = np.asarray(affine, dtype=np.float64) @ np.array(
            [center_ijk[0], center_ijk[1], center_ijk[2], 1.0],
            dtype=np.float64,
        )
        # DICOM patient coordinates use increasing X toward patient left.
        centers.append((float(label), float(world[0]), float(center_ijk[0])))

    if len(centers) != 2:
        raise ValueError("Expected two non-empty breast masks")
    centers.sort(key=lambda item: (item[1], item[2]))
    return {"right": centers[0][0], "left": centers[1][0]}


def _read_series_orientation(
    series_dir: Union[str, Path],
) -> Tuple[Optional[Tuple[float, ...]], Optional[str]]:
    """Read ``ImageOrientationPatient`` and ``PatientPosition`` from a DICOM series."""
    series_dir = Path(series_dir)
    dicoms = sorted(series_dir.glob("*.dcm"))
    if not dicoms:
        return None, None
    ds = pydicom.dcmread(
        str(dicoms[0]),
        stop_before_pixels=True,
        specific_tags=["ImageOrientationPatient", "PatientPosition"],
    )
    iop = getattr(ds, "ImageOrientationPatient", None)
    pos = getattr(ds, "PatientPosition", None)
    iop_t = tuple(float(x) for x in iop) if iop is not None else None
    pos_s = str(pos) if pos is not None else None
    return iop_t, pos_s


def breast_divider_rot90_k(
    image_orientation_patient: Optional[Sequence[float]],
    patient_position: Optional[str] = None,
) -> int:
    """Choose in-plane ``np.rot90`` k for a BreastDivider mask.

    BreastDivider masks are authored in a fixed array orientation. After affine
    resampling onto a DICOM volume, an additional in-plane rotation is required
    so the two breast labels land on the breasts in voxel space.

    Uses DICOM ``ImageOrientationPatient`` rounded to nearest integer. In this
    Duke dataset the mask-bearing series use exactly three rounded orientation
    vectors:

      * ``(1, 0, 0, 0, 1, 0)`` → ``k=1``  (90° counter-clockwise)
      * ``(-1, 0, 0, 0, -1, 0)`` → ``k=-1`` (90° clockwise)
      * ``(-1, 0, 0, 0, 1, 0)`` → ``k=1``  (90° counter-clockwise; rare mixed case)

    Returns ``k`` for ``np.rot90(..., k=k, axes=(0, 1))`` (also ``0`` / ``2``
    reserved if a future rule needs no-op / 180°).
    """
    if image_orientation_patient is None or len(image_orientation_patient) < 6:
        return 1
    rounded_iop = tuple(int(round(float(x))) for x in image_orientation_patient[:6])
    if rounded_iop in _BREAST_DIVIDER_ROTATION_BY_ROUNDED_IOP:
        return _BREAST_DIVIDER_ROTATION_BY_ROUNDED_IOP[rounded_iop]

    _ = (patient_position or "").upper()  # recorded by callers in meta
    raise ValueError(
        "Unsupported rounded ImageOrientationPatient for BreastDivider rotation: "
        f"{rounded_iop}. Add it to _BREAST_DIVIDER_ROTATION_BY_ROUNDED_IOP."
    )


def _rot90_k_name(k: int) -> str:
    return {0: "none", 1: "rot90_ccw", -1: "rot90_cw", 2: "rot180", 3: "rot90_cw"}.get(
        int(k), f"rot90_k={k}"
    )

def _resample_mask_to_volume(
    mask_path: Union[str, Path],
    volume_shape: Tuple[int, ...],
    affine: np.ndarray,
) -> np.ndarray:
    """Resample a BreastDivider mask onto the volume voxel grid.

    The BreastDivider ``.nii.gz`` masks carry their own affine that differs from
    the DICOM-derived volume affine (a Z-axis flip plus translation offsets).
    Resampling through both affines (nearest-neighbor, to preserve integer
    labels) aligns the mask voxel-for-voxel with ``volume.nii`` so bounding-box
    crops index the same underlying image data.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    mask_img = nib.load(str(mask_path))
    resampled = resample_from_to(
        mask_img,
        (tuple(int(s) for s in volume_shape), np.asarray(affine, dtype=np.float64)),
        order=0,
    )
    return np.ascontiguousarray(
        np.rint(np.asarray(resampled.get_fdata(dtype=np.float32))).astype(np.float32)
    )


def write_breast_side_crops(
    volume: np.ndarray,
    affine: np.ndarray,
    out_dir: Union[str, Path],
    mask_path: Union[str, Path],
    *,
    margin: int = 0,
    series_dir: Optional[Union[str, Path]] = None,
    image_orientation_patient: Optional[Sequence[float]] = None,
    patient_position: Optional[str] = None,
) -> Dict[str, Any]:
    """Write ``left.nii`` and ``right.nii`` crops using a BreastDivider mask."""
    import nibabel as nib

    out_dir = Path(out_dir)
    mask_path = Path(mask_path)
    if series_dir is None:
        series_dir = mask_path.parent
    if image_orientation_patient is None or patient_position is None:
        iop_read, pos_read = _read_series_orientation(series_dir)
        if image_orientation_patient is None:
            image_orientation_patient = iop_read
        if patient_position is None:
            patient_position = pos_read

    rot_k = breast_divider_rot90_k(image_orientation_patient, patient_position)
    mask_data = _resample_mask_to_volume(mask_path, volume.shape, affine)
    if rot_k % 4 != 0:
        mask_data = np.ascontiguousarray(np.rot90(mask_data, k=rot_k, axes=(0, 1)))
    if tuple(mask_data.shape) != tuple(volume.shape):
        raise ValueError(
            f"Aligned mask shape {mask_data.shape} does not match volume shape "
            f"{volume.shape} for {mask_path}"
        )

    label_sides = _breast_label_sides(mask_data, affine)
    crops: Dict[str, Any] = {}
    for side, label in label_sides.items():
        bbox = _bbox_from_mask(mask_data == label, margin=margin)
        crop = volume[
            bbox[0][0] : bbox[0][1],
            bbox[1][0] : bbox[1][1],
            bbox[2][0] : bbox[2][1],
        ]
        filename = _LEFT_BREAST_VOLUME_NAME if side == "left" else _RIGHT_BREAST_VOLUME_NAME
        crop_affine = _crop_affine(affine, bbox)
        nib.Nifti1Image(crop.astype(np.float32), crop_affine).to_filename(
            str(out_dir / filename)
        )
        crops[side] = {
            "path": filename,
            "label": label,
            "bbox_ijk": [[a, b] for a, b in bbox],
            "shape": list(crop.shape),
        }
    return {
        "mask_path": str(mask_path),
        "margin": int(margin),
        "mask_alignment": f"resample_from_to(order=0)+{_rot90_k_name(rot_k)}",
        "mask_rot90_k": int(rot_k),
        "mask_rot90_name": _rot90_k_name(rot_k),
        "image_orientation_patient": (
            list(image_orientation_patient) if image_orientation_patient is not None else None
        ),
        "rounded_image_orientation_patient": (
            [int(round(float(x))) for x in image_orientation_patient[:6]]
            if image_orientation_patient is not None
            else None
        ),
        "patient_position": patient_position,
        "crops": crops,
    }


def write_breast_side_crops_from_files(
    volume_path: Union[str, Path],
    out_dir: Union[str, Path],
    mask_path: Union[str, Path],
    *,
    margin: int = 0,
    series_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Write side crops from an existing ``volume.nii`` and mask file."""
    import nibabel as nib

    img = nib.load(str(volume_path))
    volume = np.asarray(img.get_fdata(dtype=np.float32))
    return write_breast_side_crops(
        volume,
        img.affine,
        out_dir,
        mask_path,
        margin=margin,
        series_dir=series_dir if series_dir is not None else Path(mask_path).parent,
    )


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
    split_breasts: bool = False,
    breast_crop_margin: int = 0,
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

    breast_crops = None
    breast_mask_path = series_dir / _BREAST_DIVIDER_MASK_NAME
    if split_breasts:
        if not breast_mask_path.is_file():
            raise FileNotFoundError(f"Missing BreastDivider mask: {breast_mask_path}")
        # PatientPosition is not always in the header cache; read from series DICOM.
        _iop, patient_position = _read_series_orientation(series_dir)
        iop = headers[0].image_orientation or _iop
        breast_crops = write_breast_side_crops(
            volume,
            affine,
            out_dir,
            breast_mask_path,
            margin=breast_crop_margin,
            series_dir=series_dir,
            image_orientation_patient=iop,
            patient_position=patient_position,
        )

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
    if breast_crops is not None:
        meta["breast_divider"] = breast_crops
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
        split_breasts,
        breast_crop_margin,
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
            split_breasts=split_breasts,
            breast_crop_margin=breast_crop_margin,
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
    split_breasts: bool = False,
    breast_crop_margin: int = 0,
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

    # Pre-group mapping slices for fast worker args (dict-serializable).
    # One pass over the spreadsheet (~seconds) instead of per-series scans (~hours).
    logger.info(
        "Indexing mapping slices for %d scan type(s)%s",
        len(scan_types),
        f" and {len(patient_ids)} patient(s)" if patient_ids is not None else "",
    )
    slices_index = build_mapping_slices_index(
        mapping_df,
        scan_types=scan_types,
        patient_ids=patient_ids,
    )
    mapping_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {
        key: mapping_slices_as_dicts(slices) for key, slices in slices_index.items()
    }
    logger.info("Indexed mapping slices for %d series", len(mapping_by_key))

    jobs = []
    errors: List[Dict[str, str]] = []
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
            volume_path = out_dir / "volume.nii"
            meta_path = out_dir / "meta.json"
            crops_exist = (out_dir / _LEFT_BREAST_VOLUME_NAME).is_file() and (
                out_dir / _RIGHT_BREAST_VOLUME_NAME
            ).is_file()
            if (
                skip_existing
                and volume_path.is_file()
                and meta_path.is_file()
                and (not split_breasts or crops_exist)
            ):
                continue
            if skip_existing and split_breasts and volume_path.is_file() and meta_path.is_file():
                mask_path = series_dir / _BREAST_DIVIDER_MASK_NAME
                if not mask_path.is_file():
                    errors.append(
                        {
                            "patient_id": pid,
                            "scan_type": scan,
                            "error": f"Missing BreastDivider mask: {mask_path}",
                        }
                    )
                    continue
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    meta["breast_divider"] = write_breast_side_crops_from_files(
                        volume_path,
                        out_dir,
                        mask_path,
                        margin=breast_crop_margin,
                        series_dir=series_dir,
                    )
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
                    continue
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "patient_id": pid,
                            "scan_type": scan,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    logger.error("Failed breast crops %s/%s: %s", pid, scan, exc)
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
                    split_breasts,
                    breast_crop_margin,
                )
            )

    logger.info("Converting %d series with %d workers", len(jobs), workers)
    results: List[Dict[str, Any]] = []

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
                **(
                    {
                        "left_path": f"{r['patient_id']}/{r['scan_type']}/{_LEFT_BREAST_VOLUME_NAME}",
                        "right_path": f"{r['patient_id']}/{r['scan_type']}/{_RIGHT_BREAST_VOLUME_NAME}",
                    }
                    if "breast_divider" in r
                    else {}
                ),
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
        "split_breasts": split_breasts,
        "breast_crop_margin": breast_crop_margin,
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
# BreastDivider mask organization
# ---------------------------------------------------------------------------


def _normalize_breastdivider_series_key(series: str) -> str:
    """Normalize descriptive series folder names to BreastDivider id keys."""
    s = series.replace("/", "").replace("+", "").replace(" ", "-")
    return s.replace("Duke_Breast", "Breast")


def _extract_descriptive_series(descriptive_path: str) -> Optional[str]:
    """Extract the series folder from a TCIA descriptive_path (may contain '/')."""
    m = _DESC_SERIES_RE.search(str(descriptive_path).replace("\\", "/"))
    return m.group(1) if m else None


def build_breastdivider_duke_scan_map(
    mapping_xlsx: Union[str, Path] = _DEFAULT_MAPPING_XLSX,
    id_mapping_csv: Union[str, Path] = _DEFAULT_BREASTDIVIDER_CSV,
) -> pd.DataFrame:
    """Map BreastDivider mask ids to Duke ``(patient_id, scan_type)``.

    Uses ``breastdivider_id_mapping.csv`` (Duke_* rows) plus the TCIA filepath
    mapping spreadsheet (``original_path_and_filename`` → scan type,
    ``descriptive_path`` → series folder matched to the BreastDivider id).
    """
    id_df = pd.read_csv(id_mapping_csv)
    duke = id_df[id_df["id"].astype(str).str.startswith("Duke_Breast_MRI", na=False)].copy()
    if duke.empty:
        raise ValueError(f"No Duke_Breast_MRI rows in {id_mapping_csv}")

    parsed = duke["id"].astype(str).str.extract(_DUKE_BREASTDIVIDER_ID_RE)
    if parsed[0].isna().any():
        bad = duke.loc[parsed[0].isna(), "id"].head(5).tolist()
        raise ValueError(f"Unparseable Duke BreastDivider ids (sample): {bad}")
    duke["patient_id"] = parsed[0].astype(int).map(lambda n: f"Breast_MRI_{n:03d}")
    duke["series_key"] = parsed[1].map(_normalize_breastdivider_series_key)

    mapping_df = load_mapping_table(mapping_xlsx)
    rows: List[Dict[str, str]] = []
    seen: set = set()
    for row in mapping_df.itertuples(index=False):
        parsed_orig = _parse_original_path(getattr(row, "original_path_and_filename"))
        if parsed_orig is None:
            continue
        patient_id, scan_type, _fname = parsed_orig
        series_raw = _extract_descriptive_series(str(getattr(row, "descriptive_path", "")))
        if series_raw is None:
            continue
        series_key = _normalize_breastdivider_series_key(series_raw)
        key = (patient_id, series_key)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "patient_id": patient_id,
                "scan_type": scan_type,
                "series_key": series_key,
            }
        )
    series_map = pd.DataFrame(rows)

    merged = duke.merge(series_map, on=["patient_id", "series_key"], how="left")
    missing = merged["scan_type"].isna().sum()
    if missing:
        sample = merged.loc[merged["scan_type"].isna(), "id"].head(5).tolist()
        raise ValueError(
            f"Failed to resolve scan_type for {missing}/{len(merged)} Duke masks "
            f"(sample ids: {sample})"
        )
    return merged[
        ["BreastDivider_id", "id", "patient_id", "scan_type", "series_key"]
    ].reset_index(drop=True)


def _find_breastdivider_mask(
    breast_divider_id: str,
    batch_dirs: Sequence[Path],
) -> Optional[Path]:
    name = f"{breast_divider_id}.nii.gz"
    for batch in batch_dirs:
        candidate = Path(batch) / name
        if candidate.is_file():
            return candidate
    return None


def _place_mask(src: Path, dest: Path, *, overwrite: bool) -> str:
    """Move ``src`` to ``dest``. Returns action: 'moved' | 'skipped' | 'replaced'."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if not overwrite:
            return "skipped"
        dest.unlink()
        action = "replaced"
    else:
        action = "moved"
    try:
        os.link(src, dest)
        os.unlink(src)
    except OSError:
        shutil.move(str(src), str(dest))
    return action


def organize_breast_divider_masks(
    raw_root: Union[str, Path] = _DEFAULT_RAW_ROOT,
    mapping_xlsx: Union[str, Path] = _DEFAULT_MAPPING_XLSX,
    id_mapping_csv: Union[str, Path] = _DEFAULT_BREASTDIVIDER_CSV,
    batch_dirs: Optional[Sequence[Union[str, Path]]] = None,
    nifti_root: Optional[Union[str, Path]] = None,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Place Duke BreastDivider masks next to their matching DICOM series.

    For each Duke entry in ``breastdivider_id_mapping.csv``, resolves the
    canonical scan type (``pre``, ``post_1``, …, ``T1``) via the TCIA filepath
    mapping spreadsheet, finds the on-disk ``MR_*`` series directory under
    ``raw_root``, and writes ``breast_divider.nii.gz`` inside that folder.

    Mask sources (first hit wins):
      1. ``labelsTr_batch*`` BreastDivider ``.nii.gz`` files
      2. Optional ``nifti_root/{patient}/{scan}/breast_divider.nii.gz``
         (e.g. after a prior misplaced organization)

    Non-Duke BreastDivider masks are left untouched in the batch folders.

    Returns a manifest DataFrame of actions taken.
    """
    raw_root = Path(raw_root)
    batches = [
        Path(p)
        for p in (
            batch_dirs
            if batch_dirs is not None
            else _DEFAULT_BREASTDIVIDER_BATCHES
        )
    ]
    nifti_root_path = Path(nifti_root) if nifti_root is not None else None

    scan_map = build_breastdivider_duke_scan_map(mapping_xlsx, id_mapping_csv)
    mapping_df = load_mapping_table(mapping_xlsx)
    series_dirs = build_series_directory_map(mapping_df, raw_root)

    records: List[Dict[str, Any]] = []
    for row in scan_map.itertuples(index=False):
        key = (row.patient_id, row.scan_type)
        series_dir = series_dirs.get(key)
        dest = (
            (series_dir / _BREAST_DIVIDER_MASK_NAME)
            if series_dir is not None
            else None
        )

        src = _find_breastdivider_mask(row.BreastDivider_id, batches)
        src_kind = "batch"
        if src is None and nifti_root_path is not None:
            alt = (
                nifti_root_path
                / row.patient_id
                / row.scan_type
                / _BREAST_DIVIDER_MASK_NAME
            )
            if alt.is_file():
                src = alt
                src_kind = "nifti"

        record: Dict[str, Any] = {
            "BreastDivider_id": row.BreastDivider_id,
            "source_id": row.id,
            "patient_id": row.patient_id,
            "scan_type": row.scan_type,
            "series_dir": str(series_dir) if series_dir else None,
            "dest": str(dest) if dest else None,
            "src": str(src) if src else None,
            "src_kind": src_kind if src else None,
            "action": None,
        }

        if series_dir is None or not series_dir.is_dir():
            record["action"] = "missing_series_dir"
            logger.warning(
                "No MR series dir for %s/%s (BreastDivider %s)",
                row.patient_id,
                row.scan_type,
                row.BreastDivider_id,
            )
        elif src is None:
            record["action"] = "missing_mask"
            logger.warning(
                "Mask not found for %s (%s/%s)",
                row.BreastDivider_id,
                row.patient_id,
                row.scan_type,
            )
        elif dry_run:
            record["action"] = "would_move"
        else:
            record["action"] = _place_mask(src, dest, overwrite=overwrite)

        records.append(record)

    manifest = pd.DataFrame(records)
    counts = manifest["action"].value_counts().to_dict()
    logger.info(
        "organize_breast_divider_masks: %d Duke masks; actions=%s",
        len(manifest),
        counts,
    )
    return manifest


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
    c.add_argument(
        "--split-breasts",
        action="store_true",
        help="Use breast_divider.nii.gz masks to write left.nii and right.nii crops",
    )
    c.add_argument(
        "--breast-crop-margin",
        type=int,
        default=0,
        help="Voxel margin to add around each breast bounding box",
    )

    ph = sub.add_parser("phenotypes", help="Rebuild phenotype JSON only")
    ph.add_argument("--clinical-xlsx", type=Path, default=_DEFAULT_CLINICAL_XLSX)
    ph.add_argument("--labels-dir", type=Path, default=_DEFAULT_OUT_ROOT / "labels")

    bd = sub.add_parser(
        "organize-breast-dividers",
        help="Place Duke BreastDivider masks into raw MR series folders",
    )
    bd.add_argument("--raw-root", type=Path, default=_DEFAULT_RAW_ROOT)
    bd.add_argument("--mapping-xlsx", type=Path, default=_DEFAULT_MAPPING_XLSX)
    bd.add_argument("--id-mapping-csv", type=Path, default=_DEFAULT_BREASTDIVIDER_CSV)
    bd.add_argument(
        "--batch-dirs",
        type=Path,
        nargs="*",
        default=None,
        help="BreastDivider label batch dirs (default: labelsTr_batch1/2 under raw-root)",
    )
    bd.add_argument(
        "--nifti-root",
        type=Path,
        default=_DEFAULT_OUT_ROOT,
        help="Also look here for previously misplaced breast_divider.nii.gz files",
    )
    bd.add_argument(
        "--no-nifti-fallback",
        action="store_true",
        help="Do not search nifti-root for masks",
    )
    bd.add_argument("--overwrite", action="store_true")
    bd.add_argument("--dry-run", action="store_true")
    bd.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write the action manifest CSV",
    )

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
            split_breasts=args.split_breasts,
            breast_crop_margin=args.breast_crop_margin,
        )
    elif args.cmd == "phenotypes":
        build_phenotype_json(args.clinical_xlsx, args.labels_dir)
    elif args.cmd == "organize-breast-dividers":
        manifest = organize_breast_divider_masks(
            raw_root=args.raw_root,
            mapping_xlsx=args.mapping_xlsx,
            id_mapping_csv=args.id_mapping_csv,
            batch_dirs=args.batch_dirs,
            nifti_root=None if args.no_nifti_fallback else args.nifti_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if args.manifest is not None:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_csv(args.manifest, index=False)
            logger.info("Wrote manifest to %s", args.manifest)
    else:
        raise SystemExit(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
