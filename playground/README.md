# SALTED experiments

Run commands at the repository root in a Linux environment with SALTED's test dependencies, an MPI launcher and GNU time. Use the same interpreter for both revisions. Set `SALTED_DATASETS_PATH` to the local SALTED-datasets Git checkout. The optional `PYTHON` variable selects the interpreter for the shell runner.

```bash
python -m pytest tests/unit -q
bash playground/baseline.sh
```

The baseline trains and validates the AIMS water-monomer model with its standard 40 training structures. Each run creates a fresh workspace and Numba cache under `playground/runs/`. The pytest log records timings by stage and density RMSE in percent. GNU time reports elapsed time and maximum resident memory for the pytest invocation. The default is one thread. Set `SALTED_BENCH_THREADS` to compare a different fixed thread count.

## Solid-state cases

Set `SALTED_MATRIX_PATH` to the archive containing paired `data_dense_*.npy` overlap matrices and `data_sparse_*.npz` RKHS features. The following check uses this checkout's Numba Hessian engine and an independent SciPy product. It selects 64 occupied feature columns spread across the original width and retains every overlap row.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python -m playground.check_hessian graphene_3x3
```

| Case | Available inputs | Role |
| --- | --- | --- |
| Graphene 3×3 | 3,096 rows × 17,200 features | Development case for Hessian arithmetic. Use the full width for a production-size kernel timing after checking available memory. |
| TiS₂ 3×3 | 12,510 rows × 44,878 features | Multi-species confirmation. Use `TiS2_3x3` with the same command. A full float64 Hessian alone takes about 15 GiB. |
| ZrS₂ 3×3 | 17,064 rows × 59,560 features | The archived dense input has negative diagonal elements and relative asymmetry 1.414. Recover its provenance or regenerate the overlap before using it to validate a physical calculation. |

The matrix checks exercise one stage. End-to-end density accuracy on a solid needs geometries paired with the matching basis, coefficients and overlaps. Geometry-only trajectories can already exercise descriptor generation. Machine-specific locations are kept in the local run directory.

## Worker brief

Record the base commit, hypothesis, editable files, development command, expected effect and trial budget before editing. Return the branch and commit, reproduction commands, numerical differences, repeated timing samples, memory and a one-sentence interpretation. The coordinator records a keep, discard or inconclusive decision after independent verification.
