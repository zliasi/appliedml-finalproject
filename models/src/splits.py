"""Shared dataset/version parsing, comp lookup, and split helpers.

Used by ``scripts/01-build-graphs.py``, ``scripts/05-train-baselines.py``,
and ``scripts/06-evaluate.py`` so the splits stay byte-for-byte
identical across the pipeline.
"""

import importlib.util
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType

import numpy as np

from .baselines import composition_to_vector
from .targets import WorkItem

logger = logging.getLogger(__name__)

DATASET_SRC_PKG_PREFIX: str = "_atlas_dataset_src"


def load_dataset_constants(dataset_dir: Path) -> ModuleType:
    """Import ``datasets/<name>/src/constants.py`` without name collision.

    The dataset's ``src/`` package shadows ``models/src/`` if loaded
    via plain ``from src.constants import ...`` because ``src`` is
    already cached in ``sys.modules`` as the models shared lib.
    We sidestep by registering the dataset's ``src/`` under a unique
    synthetic package name (one per dataset) and loading ``constants``
    as its submodule.

    Args:
        dataset_dir: ``datasets/<name>``.

    Returns:
        The loaded ``constants`` module (exposes ``composition_to_id``,
        ``read_compositions_csv``, ...).
    """
    src_dir = dataset_dir / "src"
    pkg_init = src_dir / "__init__.py"
    constants_path = src_dir / "constants.py"
    assert pkg_init.exists(), f"Missing: {pkg_init}"
    assert constants_path.exists(), f"Missing: {constants_path}"

    pkg_name = f"{DATASET_SRC_PKG_PREFIX}_{dataset_dir.name}"
    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_name, pkg_init,
            submodule_search_locations=[str(src_dir)],
        )
        assert spec is not None and spec.loader is not None
        package = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = package
        spec.loader.exec_module(package)

    constants_name = f"{pkg_name}.constants"
    if constants_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            constants_name, constants_path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[constants_name] = module
        spec.loader.exec_module(module)

    return sys.modules[constants_name]

SPLIT_SEED: int = 42
VAL_RATIO: float = 0.10
TEST_RATIO: float = 0.10

DATASET_TOKEN_RE = re.compile(r"^(?P<name>.+?)-(?P<version>v\d+p\d+)$")


def parse_dataset_token(token: str) -> tuple[str, str]:
    """Split ``<name>-v<M>p<N>`` into ``(name, version)``.

    Raises:
        ValueError: If ``token`` doesn't match the expected shape.
    """
    match = DATASET_TOKEN_RE.match(token)
    if not match:
        raise ValueError(
            f"Bad dataset token {token!r}: "
            f"expected <name>-v<M>p<N>, e.g. fcc12-v1p1"
        )
    return match["name"], match["version"]


def load_composition_index(
    dataset_dir: Path,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Build (comp_types, compositions) keyed by comp_id.

    Reads ``data/compositions.csv`` plus
    ``data/augmentation/*/compositions.csv`` so augmented runs see
    the full set. Uses the dataset's own ``src.constants`` for
    composition-to-id derivation and CSV parsing.
    """
    constants = load_dataset_constants(dataset_dir)
    composition_to_id = constants.composition_to_id
    read_compositions_csv = constants.read_compositions_csv

    sources = [dataset_dir / "data" / "compositions.csv"]
    sources.extend(sorted(
        (dataset_dir / "data" / "augmentation").glob(
            "*/compositions.csv",
        )
    ))

    comp_types: dict[str, str] = {}
    compositions: dict[str, dict[str, float]] = {}
    for csv_path in sources:
        if not csv_path.exists():
            continue
        for entry in read_compositions_csv(csv_path):
            comp_id = composition_to_id(entry["composition"])
            comp_types[comp_id] = entry["type"]
            compositions[comp_id] = entry["composition"]
    assert len(compositions) > 0, "No compositions loaded"
    logger.info(
        "Loaded %d comp_ids from %d csv(s)",
        len(compositions), len(sources),
    )
    return comp_types, compositions


def stratified_split(
    group_keys: list[str],
    comp_types: dict[str, str],
) -> tuple[set[str], set[str], set[str]]:
    """80/10/10 stratified by composition_type.

    Partition unit is ``group_key`` (slab-key for hads, comp_id for
    wf). All items sharing a group_key go in the same split.

    Args:
        group_keys: Group identifiers to partition.
        comp_types: Map comp_id -> composition_type. The comp_id is
            obtained from ``group_key.split("/")[0]``.

    Returns:
        (train, val, test) as sets of group_keys.
    """
    rng = np.random.default_rng(SPLIT_SEED)
    by_type: dict[str, list[str]] = defaultdict(list)
    for key in group_keys:
        comp_id = key.split("/")[0]
        by_type[comp_types.get(comp_id, "unknown")].append(key)

    train: set[str] = set()
    val: set[str] = set()
    test: set[str] = set()
    for ctype, keys in sorted(by_type.items()):
        shuffled = list(keys)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_val = max(1, int(round(n_total * VAL_RATIO)))
        n_test = max(1, int(round(n_total * TEST_RATIO)))
        n_train = n_total - n_val - n_test
        assert n_train > 0, f"Not enough for {ctype}"
        train.update(shuffled[:n_train])
        val.update(shuffled[n_train:n_train + n_val])
        test.update(shuffled[n_train + n_val:])
    return train, val, test


def items_to_features(
    items: list[WorkItem],
    compositions: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Composition-vector + target arrays from WorkItems."""
    assert len(items) > 0, "No items"
    x_list = []
    y_list = []
    for wi in items:
        x_list.append(composition_to_vector(compositions[wi.comp_id]))
        y_list.append(wi.target_value)
    return np.array(x_list), np.array(y_list)
