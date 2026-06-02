"""LMDB-backed graph dataset and save/load helpers.

Lightweight wrapper around lmdb. Each graph is stored as a pickled
PyG ``Data`` object keyed by integer string; total count under
``b"length"``. Random access without loading the full dataset into
RAM. Compatible with PyG DataLoader.

Fork safety: ``LmdbGraphDataset.env`` is opened lazily and re-opened
when the owning pid changes, so DataLoader workers (which fork from
the parent) each get their own handle. Inheriting an mmap-backed
lmdb env across fork is not safe and was previously causing worker
crashes in multi-worker training.
"""

import logging
import os
import pickle
from pathlib import Path

from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def save_graphs(
    graphs: list[Data], output_path: Path,
) -> None:
    """Save graphs to LMDB database.

    Each graph is stored as a pickle-serialized value with
    integer key. Total count stored under b"length".

    Args:
        graphs: List of PyTorch Geometric Data objects
        output_path: Path for LMDB file

    Raises:
        OSError: If LMDB cannot be created
    """
    import lmdb

    assert len(graphs) > 0, "Must have at least one graph"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(output_path), map_size=2**40,
        subdir=False, meminit=False, map_async=True,
    )

    with env.begin(write=True) as txn:
        for i, graph in enumerate(graphs):
            txn.put(
                str(i).encode("ascii"),
                pickle.dumps(graph, protocol=-1),
            )
        txn.put(b"length", pickle.dumps(len(graphs)))

    env.close()
    assert output_path.exists(), "Save failed"
    logger.info(
        "Saved %d graphs to %s", len(graphs), output_path,
    )


class LmdbGraphDataset:
    """Memory-mapped graph dataset backed by LMDB.

    Provides random access to individual graphs without loading
    the full dataset into RAM. Compatible with PyG DataLoader,
    including with ``num_workers > 0``.

    Fork model: the env handle is opened lazily and re-opened
    whenever the owning pid changes. The constructor reads the
    length header by opening + closing a one-shot env, so no env
    handle is alive at fork time. Each forked DataLoader worker
    opens its own env on its first ``__getitem__``.

    Attributes:
        lmdb_path: Path to the .lmdb file.
        length: Total number of graphs (read once at construction).
    """

    def __init__(self, lmdb_path: Path):
        """Read the length header; defer opening the read env.

        Args:
            lmdb_path: Path to .lmdb file.

        Raises:
            FileNotFoundError: If LMDB file not found.
        """
        import lmdb

        lmdb_path = Path(lmdb_path)
        if not lmdb_path.exists():
            raise FileNotFoundError(
                f"LMDB not found: {lmdb_path}"
            )
        self.lmdb_path = lmdb_path
        self._env = None
        self._env_pid: int | None = None

        env = lmdb.open(
            str(lmdb_path), readonly=True,
            subdir=False, lock=False, meminit=False,
        )
        with env.begin() as txn:
            raw = txn.get(b"length")
            assert raw is not None, "Missing length key"
            self.length = pickle.loads(raw)
        env.close()

        assert self.length > 0, "LMDB is empty"
        logger.info(
            "LMDB header: %d graphs from %s",
            self.length, lmdb_path,
        )

    @property
    def env(self):
        """Per-process lmdb env. Opens lazily; re-opens after fork."""
        pid = os.getpid()
        if self._env is None or self._env_pid != pid:
            import lmdb
            self._env = lmdb.open(
                str(self.lmdb_path), readonly=True,
                subdir=False, lock=False, meminit=False,
            )
            self._env_pid = pid
        return self._env

    def __getstate__(self) -> dict:
        """Strip the env handle when pickled (spawn-mode workers)."""
        state = self.__dict__.copy()
        state["_env"] = None
        state["_env_pid"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __len__(self) -> int:
        """Return number of graphs in dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Data:
        """Load a single graph by index.

        Raises:
            IndexError: If idx out of range.
        """
        if idx < 0 or idx >= self.length:
            raise IndexError(
                f"Index {idx} out of range [0, {self.length})"
            )
        with self.env.begin() as txn:
            raw = txn.get(str(idx).encode("ascii"))
            assert raw is not None, f"Missing key: {idx}"
            return pickle.loads(raw)

    def close(self) -> None:
        """Close the LMDB environment (if open in this process)."""
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_pid = None


def load_graphs(input_path: Path) -> LmdbGraphDataset:
    """Load graphs from LMDB database.

    Args:
        input_path: Path to LMDB file

    Returns:
        LmdbGraphDataset with random access to graphs

    Raises:
        FileNotFoundError: If LMDB file not found
    """
    return LmdbGraphDataset(input_path)
