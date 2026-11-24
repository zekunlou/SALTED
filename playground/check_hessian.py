"""Check a bounded slice of an archived SALTED Hessian against SciPy."""
import argparse
import os
from pathlib import Path

import numpy as np
from scipy import sparse

from salted.numba_sparse import get_hessian_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["graphene_3x3", "TiS2_3x3"])
    args = parser.parse_args()
    if not os.environ.get("SALTED_MATRIX_PATH"):
        parser.error("set SALTED_MATRIX_PATH to the archived input directory")
    root = Path(os.environ["SALTED_MATRIX_PATH"])
    overlap = np.load(root / f"data_dense_{args.case}.npy", mmap_mode="r")
    features = sparse.load_npz(root / f"data_sparse_{args.case}.npz").tocsc()
    occupied = np.flatnonzero(np.diff(features.indptr))
    if len(occupied) < 64:
        raise ValueError("expected at least 64 occupied feature columns")
    columns = occupied[np.linspace(0, len(occupied) - 1, 64, dtype=int)]
    psi = features[:, columns].tocsr()
    reference = psi.T @ (overlap @ psi)
    actual = get_hessian_engine(*psi.shape).compute(overlap, psi)
    error = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
    print(f"{args.case} shape={psi.shape} relative_frobenius_error={error:.3e}")
    if not error < 1e-12:
        raise SystemExit("Hessian disagrees with the SciPy reference")


if __name__ == "__main__":
    main()
