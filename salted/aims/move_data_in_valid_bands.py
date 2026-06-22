import os
import shutil
import sys
from pathlib import Path

import numpy as np

from salted.sys_utils import ParseConfig, read_system


def build():
    inp = ParseConfig().parse_input()

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(
        filename=inp.system.filename,
        spelist=inp.system.species,
        dfbasis=inp.qm.dfbasis,
    )  # load from training dataset

    ntrain = int(inp.gpr.trainfrac * inp.gpr.Ntrain)
    reg_log10_intstr = str(int(np.log10(inp.gpr.regul)))
    validation_dpath = (
        Path(inp.salted.saltedpath)
        / f"validations_{inp.salted.saltedname}/M{inp.gpr.Menv}_zeta{inp.gpr.z}/N{ntrain}_reg{reg_log10_intstr}"
    )
    regrdir_dpath = Path(inp.salted.saltedpath) / f"regrdir_{inp.salted.saltedname}"
    trainrangetot = np.loadtxt(regrdir_dpath / f"training_set_N{inp.gpr.Ntrain}.txt", dtype=int)
    trainrange = trainrangetot[:ntrain]  # training subset in the training dataset, start from 0
    testrange = np.setdiff1d(list(range(ndata)), trainrangetot)  # valid subset in the training dataset, start from 0
    # check trainrange testrange no overlap
    assert len(np.intersect1d(trainrange, testrange)) == 0, "Train and test ranges overlap!"
    testtrainrange = np.concatenate((testrange, trainrange))

    # check if all files are present
    valid_coeffs_fnames = [i for i in os.listdir(validation_dpath) if i.startswith("COEFFS-")]
    valid_coeffs_indices = [int(i.split("-")[1].split(".")[0]) - 1 for i in valid_coeffs_fnames]  # start from 0
    missing_indices = set(testtrainrange) - set(valid_coeffs_indices)
    if len(missing_indices) > 0:
        print(f"Missing coefficient files for indices: {sorted(missing_indices)}")
        raise FileNotFoundError("Some coefficient files are missing in the validation directory.")

    # prepare aims dir and copy files
    aims_valid_dpath = Path(inp.salted.saltedpath) / "aims_valid_data"
    os.makedirs(aims_valid_dpath, exist_ok=True)
    np.savetxt(
        aims_valid_dpath / "valid_trainset_indices_start_1.dat", trainrange + 1, fmt="%d"
    )  # save as 1-based indices
    np.savetxt(
        aims_valid_dpath / "valid_validset_indices_start_1.dat", testrange + 1, fmt="%d"
    )  # save as 1-based indices

    for idx in testtrainrange:  # idx start from 0
        # print(f"Copying validation data for index {idx + 1}...")
        os.makedirs(aims_valid_dpath / f"{idx + 1}", exist_ok=True)  # start from 1
        shutil.copyfile(
            validation_dpath / f"COEFFS-{idx + 1}.dat",  # start from 1
            aims_valid_dpath / f"{idx + 1}/ri_restart_coeffs_predicted.out",
            # follow_symlinks=False,
        )


if __name__ == "__main__":
    build()
