from typing import Iterable, List

import torch
from ase import Atoms
from metatomic.torch import System, systems_to_torch


REQUIRED_SYSTEM_COLUMNS = ("positions", "cell", "pbc", "atomic_numbers")


def _import_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "pyarrow is required to read parquet files. "
            "Install it with `pip install pyarrow`."
        ) from e
    return pq


def _read_parquet_columns(filename: str, columns: Iterable[str]):
    pq = _import_pyarrow()
    return pq.read_table(filename, columns=list(columns))


def read_systems(filename: str) -> List[System]:
    """Read systems from a ColabFit-style parquet file.

    Each row is one structure. Required columns are:

    * ``positions``        list<list<double>>, shape (n_atoms, 3), Å
    * ``cell``             list<list<double>>, shape (3, 3), Å
    * ``pbc``              list<bool>, length 3
    * ``atomic_numbers``   list<int64>, length n_atoms

    :param filename: path to a single ``.parquet`` file
    :return: list of :py:class:`System` objects, one per row, in ``float64``
    """
    table = _read_parquet_columns(filename, REQUIRED_SYSTEM_COLUMNS)

    missing = [c for c in REQUIRED_SYSTEM_COLUMNS if c not in table.column_names]
    if missing:
        raise ValueError(
            f"parquet file {filename!r} is missing required columns: {missing}"
        )

    positions_col = table.column("positions")
    cell_col = table.column("cell")
    pbc_col = table.column("pbc")
    numbers_col = table.column("atomic_numbers")

    atoms_list = [
        Atoms(
            numbers=numbers_col[i].as_py(),
            positions=positions_col[i].as_py(),
            cell=cell_col[i].as_py(),
            pbc=pbc_col[i].as_py(),
        )
        for i in range(table.num_rows)
    ]

    return systems_to_torch(atoms_list, dtype=torch.float64)
