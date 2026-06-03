"""Constants for 21-element magnetic moment alloy slab dataset.

21 elements spanning primary magnetic (Fe, Co, Ni, Mn), secondary
magnetic (Cr, V), induced-moment NM (Pt, Pd, Ir, Rh, Ru, Os, Mo, W,
Cu, Au, Ag, Re), and extended NM (Ti, Al, Zn). Supports FCC(111),
BCC(110), and HCP(0001) slabs. Slab size is per-row (4x4x4 default
for random alloys and dilute; 2x2x4 for binary intermetallic
prototypes; 4x4x4 for ternary Heuslers).
"""

import csv
import json
from pathlib import Path
from typing import Any

from .errors import CompositionError


class LatticeType:
    """Crystal structure identifiers."""

    FCC: str = 'fcc'
    BCC: str = 'bcc'
    HCP: str = 'hcp'
    ALL: tuple[str, ...] = ('fcc', 'bcc', 'hcp')


SUPPORTED_ELEMENTS: list[str] = sorted([
    'Ag', 'Al', 'Au', 'Co', 'Cr', 'Cu', 'Fe', 'Ir',
    'Mn', 'Mo', 'Ni', 'Os', 'Pd', 'Pt', 'Re', 'Rh',
    'Ru', 'Ti', 'V', 'W', 'Zn',
])

N_SUPPORTED_ELEMENTS: int = 21

ELEMENT_CRYSTAL_STRUCTURE: dict[str, str] = {
    'Ag': LatticeType.FCC, 'Al': LatticeType.FCC,
    'Au': LatticeType.FCC, 'Ir': LatticeType.FCC,
    'Ni': LatticeType.FCC, 'Pd': LatticeType.FCC,
    'Pt': LatticeType.FCC, 'Rh': LatticeType.FCC,
    'Cu': LatticeType.FCC,
    'Cr': LatticeType.BCC, 'Fe': LatticeType.BCC,
    'Mn': LatticeType.BCC, 'Mo': LatticeType.BCC,
    'V':  LatticeType.BCC, 'W':  LatticeType.BCC,
    'Co': LatticeType.HCP, 'Os': LatticeType.HCP,
    'Re': LatticeType.HCP, 'Ru': LatticeType.HCP,
    'Ti': LatticeType.HCP, 'Zn': LatticeType.HCP,
}

IDEAL_HCP_CA_RATIO: float = 1.633

DEFAULT_SLAB_SIZE: tuple[int, int, int] = (4, 4, 4)
N_SLAB_ATOMS_DEFAULT: int = 64

MAGNETIC_METALS: set[str] = {
    'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Mo', 'Ni', 'Os', 'Ru', 'V', 'W',
}

MAGNETIC_PRIMARY: set[str] = {'Co', 'Cr', 'Fe', 'Mn', 'Ni', 'V'}

N_FROZEN_LAYERS: int = 2
DEFAULT_VACUUM_ANG: float = 10.0
DEFAULT_SEED: int = 42
DEFAULT_MAX_ATTEMPTS: int = 10000
PROBE_HEIGHT_ANG: float = 4.0
SURFACE_TOLERANCE_ANG: float = 0.5

COVALENT_RADII_PM: dict[str, float] = {
    'Ag': 145, 'Al': 121, 'Au': 136, 'Co': 126,
    'Cr': 127, 'Cu': 132, 'Fe': 126, 'Ir': 141,
    'Mn': 132, 'Mo': 154, 'Ni': 124, 'Os': 144,
    'Pd': 139, 'Pt': 136, 'Re': 151, 'Rh': 142,
    'Ru': 146, 'Ti': 160, 'V': 134, 'W': 162,
    'Zn': 122,
}

ELEMENT_COMPATIBILITY_GROUPS: dict[str, list[str]] = {
    'noble_pgm': ['Ag', 'Au', 'Ir', 'Os', 'Pd', 'Pt', 'Rh', 'Ru'],
    '3d_transition': ['Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V'],
    'refractory': ['Mo', 'Re', 'W'],
    'light_metals': ['Al', 'Ti', 'Zn'],
}

ELEMENT_TO_GROUP: dict[str, str] = {}
for _group, _elements in ELEMENT_COMPATIBILITY_GROUPS.items():
    for _elem in _elements:
        ELEMENT_TO_GROUP[_elem] = _group

GROUP_ADJACENCY: set[tuple[str, str]] = {
    ('noble_pgm', '3d_transition'),
    ('noble_pgm', 'refractory'),
    ('3d_transition', 'refractory'),
    ('3d_transition', 'light_metals'),
    ('refractory', 'light_metals'),
}

VARIANT_COUNTS: dict[str, int] = {
    'pure': 1,
    'binary': 4,
    'ternary': 3,
    'quaternary': 2,
    'quinary': 1,
    'senary': 1,
}

COMPOSITIONS_PER_COMBO: dict[int, int] = {
    2: 5,
    3: 4,
    4: 4,
    5: 3,
    6: 2,
}

SAMPLED_COMBOS: dict[int, int] = {
    3: 600,
    4: 400,
    5: 200,
    6: 80,
}

BINARY_FRACTIONS: list[float] = [0.125, 0.25, 0.50, 0.75, 0.875]

HR_STRICT_THRESHOLD: float = 0.18
LATTICE_TIE_BUFFER: float = 0.10

COMPOSITION_CSV_ELEMENTS: list[str] = [
    element.lower() for element in SUPPORTED_ELEMENTS
]

COMPOSITION_CSV_HEADER: list[str] = (
    ['index', 'type', 'lattice_type', 'n_variants']
    + COMPOSITION_CSV_ELEMENTS
)

ORDERED_CSV_HEADER: list[str] = (
    ['index', 'type', 'lattice_type', 'placement',
     'nx', 'ny', 'nz', 'n_variants']
    + COMPOSITION_CSV_ELEMENTS
)


def determine_lattice_types(
    composition: dict[str, float],
    tie_buffer: float = LATTICE_TIE_BUFFER,
) -> list[str]:
    """Determine slab lattice types by weighted crystal vote.

    Returns all lattices whose vote is within tie_buffer of the
    leading lattice (absolute fraction of total). Ties prioritise
    FCC > HCP > BCC for deterministic listing order.

    Args:
        composition: Element to mole fraction mapping
        tie_buffer: Lattices with vote >= top_vote - buffer kept

    Returns:
        List of lattice types (1-3 entries)
    """
    assert len(composition) > 0, "Composition must not be empty"
    assert 0.0 <= tie_buffer < 1.0, "Buffer must be in [0, 1)"

    votes: dict[str, float] = {
        LatticeType.FCC: 0.0,
        LatticeType.BCC: 0.0,
        LatticeType.HCP: 0.0,
    }
    for element, fraction in composition.items():
        if element not in ELEMENT_CRYSTAL_STRUCTURE:
            raise KeyError(f"Unknown element: {element}")
        votes[ELEMENT_CRYSTAL_STRUCTURE[element]] += fraction

    top = max(votes.values())
    assert top > 0, "No elements with positive fraction"
    threshold = top - tie_buffer

    result: list[str] = []
    for lattice in [LatticeType.FCC, LatticeType.HCP, LatticeType.BCC]:
        if votes[lattice] >= threshold - 1e-9 and votes[lattice] > 0:
            result.append(lattice)
    assert len(result) >= 1
    return result


def determine_lattice_type(composition: dict[str, float]) -> str:
    """Single-lattice weighted vote (FCC > HCP > BCC tie-break).

    Args:
        composition: Element to mole fraction mapping

    Returns:
        Lattice type string
    """
    return determine_lattice_types(composition, tie_buffer=0.0)[0]


def validate_composition(
    composition: dict[str, float], tolerance: float = 1e-6,
) -> bool:
    """Check that composition sums to 1.0 and fractions in [0,1].

    Args:
        composition: Element symbol to mole fraction mapping
        tolerance: Allowed deviation from sum of 1.0

    Returns:
        True if valid

    Raises:
        CompositionError: If composition is invalid
    """
    assert isinstance(composition, dict), "Composition must be a dict"
    assert len(composition) > 0, "Composition must not be empty"

    total = sum(composition.values())
    if abs(total - 1.0) > tolerance:
        raise CompositionError(
            f"Composition must sum to 1.0, got {total:.6f}"
        )

    for element, fraction in composition.items():
        if fraction < 0 or fraction > 1:
            raise CompositionError(
                f"Fraction for {element} must be in [0, 1], "
                f"got {fraction}"
            )

    return True


def composition_to_id(
    composition: dict[str, float],
    lattice_type: str | None = None,
) -> str:
    """Convert composition to canonical hyphen-separated string.

    Elements sorted alphabetically, fractions as 3-digit percentages.
    If lattice_type is given, it is appended as suffix.

    Args:
        composition: Element symbol to mole fraction mapping
        lattice_type: Optional lattice suffix (fcc/bcc/hcp)

    Returns:
        Canonical composition ID string
    """
    assert len(composition) > 0, "Composition must not be empty"

    parts = []
    for element in sorted(composition.keys()):
        percentage = int(round(composition[element] * 100))
        parts.append(f"{element.lower()}{percentage:03d}")

    base = "-".join(parts)
    if lattice_type is None:
        return base
    assert lattice_type in LatticeType.ALL
    return f"{base}-{lattice_type}"


def write_compositions_csv(
    path: Path,
    entries: list[dict[str, Any]],
) -> None:
    """Write random-alloy composition entries to CSV.

    Args:
        path: Output CSV file path
        entries: List of dicts with composition, n_variants, type,
                 lattice_type
    """
    assert len(entries) > 0, "Entries list must not be empty"
    assert all("composition" in e for e in entries)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COMPOSITION_CSV_HEADER)
        for i, entry in enumerate(entries, start=1):
            comp = entry["composition"]
            row = [
                i, entry["type"], entry["lattice_type"],
                entry["n_variants"],
            ]
            for elem in SUPPORTED_ELEMENTS:
                row.append(f"{comp.get(elem, 0.0):.6f}")
            writer.writerow(row)


def read_compositions_csv(path: Path) -> list[dict[str, Any]]:
    """Read random-alloy composition entries from CSV.

    Args:
        path: Path to compositions CSV

    Returns:
        List of dicts with composition, n_variants, type,
        lattice_type
    """
    assert path.exists(), f"Compositions CSV not found: {path}"

    entries: list[dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            comp: dict[str, float] = {}
            for elem, col in zip(
                SUPPORTED_ELEMENTS, COMPOSITION_CSV_ELEMENTS,
            ):
                fraction = float(row[col])
                if fraction > 0:
                    comp[elem] = fraction
            entries.append({
                "composition": comp,
                "n_variants": int(row["n_variants"]),
                "type": row["type"],
                "lattice_type": row["lattice_type"],
            })

    assert len(entries) > 0, "CSV contains no entries"
    return entries


def write_ordered_csv(
    path: Path,
    entries: list[dict[str, Any]],
) -> None:
    """Write ordered/dilute composition entries to CSV.

    Schema columns: index, type, lattice_type, placement,
    nx, ny, nz, n_variants, then one column per element.

    Args:
        path: Output CSV path
        entries: List of dicts with composition, type, lattice_type,
                 placement, nx, ny, nz, n_variants
    """
    assert len(entries) > 0, "Entries list must not be empty"
    required = {
        "composition", "type", "lattice_type", "placement",
        "nx", "ny", "nz", "n_variants",
    }
    assert all(required.issubset(e.keys()) for e in entries), (
        "Entries missing required fields"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ORDERED_CSV_HEADER)
        for i, entry in enumerate(entries, start=1):
            comp = entry["composition"]
            row = [
                i, entry["type"], entry["lattice_type"],
                entry["placement"],
                entry["nx"], entry["ny"], entry["nz"],
                entry["n_variants"],
            ]
            for elem in SUPPORTED_ELEMENTS:
                row.append(f"{comp.get(elem, 0.0):.6f}")
            writer.writerow(row)


def read_ordered_csv(path: Path) -> list[dict[str, Any]]:
    """Read ordered/dilute composition entries from CSV.

    Args:
        path: Path to ordered-compositions CSV

    Returns:
        List of dicts with composition, type, lattice_type,
        placement, nx, ny, nz, n_variants
    """
    assert path.exists(), f"Ordered CSV not found: {path}"

    entries: list[dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            comp: dict[str, float] = {}
            for elem, col in zip(
                SUPPORTED_ELEMENTS, COMPOSITION_CSV_ELEMENTS,
            ):
                fraction = float(row[col])
                if fraction > 0:
                    comp[elem] = fraction
            entries.append({
                "composition": comp,
                "type": row["type"],
                "lattice_type": row["lattice_type"],
                "placement": row["placement"],
                "nx": int(row["nx"]),
                "ny": int(row["ny"]),
                "nz": int(row["nz"]),
                "n_variants": int(row["n_variants"]),
            })

    assert len(entries) > 0, "CSV contains no entries"
    return entries


def load_eos_lattice_constants(
    eos_path: Path,
) -> dict[str, dict[str, float]]:
    """Load EOS-derived lattice constants from consolidated JSON.

    Args:
        eos_path: Path to lattice-constants.json

    Returns:
        Nested {element: {lattice_type: a0_ang}}. Empty dict if the
        EOS file does not yet exist; callers should fall back to
        ase_default_lattice_const for missing entries.
    """
    if not eos_path.exists():
        return {}

    with open(eos_path) as f:
        raw: dict[str, Any] = json.load(f)

    result: dict[str, dict[str, float]] = {}
    for element, lattice_data in raw.items():
        result[element] = {}
        for lattice_type, eos_data in lattice_data.items():
            if "a0_ang" in eos_data:
                result[element][lattice_type] = float(
                    eos_data["a0_ang"]
                )
    return result


def load_eos_magmoms(
    eos_path: Path,
) -> dict[str, dict[str, float]]:
    """Load EOS-derived initial magnetic moments per (element, lattice).

    Args:
        eos_path: Path to lattice-constants.json

    Returns:
        Nested {element: {lattice_type: m0_bohr}}. Empty dict if the
        EOS file does not yet exist; missing entries default to 0.0.
    """
    if not eos_path.exists():
        return {}

    with open(eos_path) as f:
        raw: dict[str, Any] = json.load(f)

    result: dict[str, dict[str, float]] = {}
    for element, lattice_data in raw.items():
        result[element] = {}
        for lattice_type, eos_data in lattice_data.items():
            result[element][lattice_type] = float(
                eos_data.get("m0_bohr", 0.0)
            )
    return result


def ase_default_lattice_const(
    element: str,
    lattice_type: str,
) -> float:
    """Return ASE-data lattice constant for (element, lattice_type).

    Uses ase.data.reference_states when the element's natural lattice
    matches; otherwise falls back to a covalent-radius geometric
    estimate (close-packed nearest-neighbour distance = 2r).

    Args:
        element: Chemical symbol
        lattice_type: One of LatticeType.FCC, BCC, HCP

    Returns:
        Lattice constant a in Angstrom
    """
    from ase.data import atomic_numbers, covalent_radii, reference_states

    assert lattice_type in LatticeType.ALL, (
        f"Unknown lattice type: {lattice_type}"
    )
    z = atomic_numbers[element]
    ref = reference_states[z] if z < len(reference_states) else None
    if ref is not None and ref.get('symmetry') == lattice_type:
        return float(ref['a'])

    r_ang = covalent_radii[z]
    if lattice_type == LatticeType.FCC:
        return 2.0 * (2.0 ** 0.5) * r_ang
    if lattice_type == LatticeType.BCC:
        return 4.0 * r_ang / (3.0 ** 0.5)
    return 2.0 * r_ang


def ase_default_ca_ratio(element: str) -> float:
    """Return ASE-data c/a ratio for HCP element, or ideal value.

    Args:
        element: Chemical symbol

    Returns:
        c/a ratio; IDEAL_HCP_CA_RATIO if not in ASE reference data
    """
    from ase.data import atomic_numbers, reference_states

    z = atomic_numbers[element]
    ref = reference_states[z] if z < len(reference_states) else None
    if ref is not None and ref.get('symmetry') == LatticeType.HCP:
        ca = ref.get('c/a')
        if ca is not None:
            return float(ca)
    return IDEAL_HCP_CA_RATIO


def is_magnetic(element: str) -> bool:
    """Check if element triggers spin-polarized calculation."""
    assert isinstance(element, str)
    return element in MAGNETIC_METALS


def any_magnetic(elements: list[str]) -> bool:
    """Check if any element triggers spin polarization."""
    assert len(elements) > 0
    return any(is_magnetic(e) for e in elements)


def hume_rothery_strict(
    elements: list[str],
    threshold: float = HR_STRICT_THRESHOLD,
) -> bool:
    """Strict pair-wise Hume-Rothery check.

    Returns False if ANY pair of elements differs in covalent radius
    by more than `threshold` relative to the smaller radius.

    Args:
        elements: Element symbols (length >= 1)
        threshold: Max allowed relative pair difference

    Returns:
        True if all pairs are within threshold
    """
    assert len(elements) >= 1
    if len(elements) == 1:
        return True

    radii = [COVALENT_RADII_PM[e] for e in elements]
    min_r = min(radii)
    max_r = max(radii)
    return (max_r - min_r) / min_r <= threshold


def hume_rothery_check(elements: list[str]) -> bool:
    """Legacy Hume-Rothery check (majority-mismatched, 15%).

    Retained for backwards compatibility; new code should use
    hume_rothery_strict.

    Args:
        elements: Element symbols

    Returns:
        True if at most half of elements exceed +15% radius
    """
    assert len(elements) >= 2
    radii = [COVALENT_RADII_PM[e] for e in elements]
    min_r = min(radii)
    n_mismatch = sum(
        1 for r in radii if (r - min_r) / min_r > 0.15
    )
    return n_mismatch <= len(elements) / 2
