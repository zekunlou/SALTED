import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
from scipy import sparse
from ase.io import read

from salted import basis
from salted.cp2k.utils import compute_charge_and_dipole, compute_polarizability, init_moments
from salted.sys_utils import ParseConfig, get_atom_idx, get_conf_range, init_property_file, read_system


def get_config():
    parser = argparse.ArgumentParser(description="Validate SALTED model by prediction dataset, only single process, no MPI")
    parser.add_argument(
        "--pred_dpath",
        type=str,
        default=None,
        help="Path to the directory containing ground truth DF coefficients of pred dataset, only valid if dataset_part is 'pred'. Default is None.",
    )
    parser.add_argument(
        "--pred_indices",
        type=lambda s: [int(x) for x in s.split(",")],
        default=None,
        help="List of geometry indices to run, only valid if dataset_part is 'pred'. Default is None, which means using all geometries."
            "Comma-separated list of geometry indices, e.g. 1,2,3,4,5, always start from 1",
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        choices=["residual", "total"],
        default="residual",
        help="Type of norm to use for error calculation. Default is 'residual'.",
    )
    args = parser.parse_args()
    return args

def build():

    inp = ParseConfig().parse_input()
    (saltedname, saltedpath, saltedtype,
    filename, species, average, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data, alpha_only,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel, nspe1, nspe2, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL) = ParseConfig().get_all_params()

    assert saltedtype == "density", "Only density type is supported for prediction validation"
    assert qmcode == "aims", "Only FHI-aims is supported for prediction validation"

    # # just remote this mpi part
    # # make sure only single process
    # if parallel:
    #     from mpi4py import MPI
    #     # MPI information
    #     comm = MPI.COMM_WORLD
    #     size = comm.Get_size()
    #     rank = comm.Get_rank()
    # else:
    #     comm = None
    #     size = 1
    #     rank = 0
    # if rank != 0:
    #     exit(0)  # only single process

    config = get_config()
    norm_method = config.norm_method
    pred_indices = config.pred_indices
    pred_dpath = config.pred_dpath

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(filename_pred, species, dfbasis)
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    pdir = osp.join(saltedpath, f"predictions_{saltedname}_{predname}")
    if pred_indices is None:
        pred_indices = list(range(1, 1+len(read(filename_pred, ":"))))
    print(f"Geometry indices: {pred_indices}", flush=True)

    reg_log10_intstr = str(int(np.log10(regul)))

########################### I AM HERE !!! ###########################

    if average:
        # Load spherical averages
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load(os.path.join(saltedpath, "coefficients", "averages", f"averages_{spe}.npy"))

    ntrain = int(Ntrain * trainfrac)
    pdir = osp.join(
        saltedpath,
        f"predictions_{saltedname}_{predname}"
    )
    dirpath = osp.join(pdir,
        f"M{Menv}_zeta{zeta}",
        f"N{ntrain}_reg{reg_log10_intstr}",
    )
    efile = open(osp.join(dirpath, f"error_{norm_method}.dat"), "w")

    error_density = 0
    variance = 0
    for iconf in pred_indices:  # note: pred_indices is 1-based index
        iconf -= 1  # recover 0-based index

        overl = np.load(osp.join(pred_dpath, "overlaps", f"overlap_conf{iconf+1}.npy"))
        ref_coefs = np.load(osp.join(pred_dpath, "coefficients", f"coefficients_conf{iconf+1}.npy"))
        ref_projs = np.load(osp.join(pred_dpath, "projections", f"projections_conf{iconf+1}.npy"))
        pred_coefs = np.loadtxt(osp.join(dirpath, f"COEFFS-{iconf+1}.dat"))
        # print(f"{len(ref_coefs)=}, {len(ref_projs)=}, {len(pred_coefs)=}")
        # print(f"{natoms=}, {len(atomic_symbols[0])=}")

        if average:
            # Compute vector of isotropic average coefficients
            Av_coeffs = np.zeros(len(ref_coefs))
            i = 0
            for iat in range(natoms[iconf]):
                # print(f"{iat=}")
                spe = atomic_symbols[iconf][iat]
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        if l==0:
                            # print(f"{iat=}, {i=}, {l=}, {n=}, {av_coefs[spe][n]=}")
                            Av_coeffs[i] = av_coefs[spe][n]
                        i += 2*l+1

        # Compute predicted density projections <phi|rho>
        pred_projs = np.dot(overl,pred_coefs)

        # compute error
        error = np.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)
        error_density += error
        if (norm_method == "residual") and average:  # normalize by residual variance
            ref_projs -= np.dot(overl,Av_coeffs)  # subtract spherical average
            ref_coefs -= Av_coeffs
        else:  # normalize by total density
            pass
        var = np.dot(ref_coefs,ref_projs)
        variance += var
        print(f"{iconf+1:d} {(np.sqrt(error/var)*100):.3e}", file=efile)
        print(f"{iconf+1}: {(np.sqrt(error/var)*100):.3e} % RMSE", flush=True)

    print(f"\n % RMSE: {(100*np.sqrt(error_density/variance)):.3e}", file=efile)
    print(f"\n % RMSE: {(100*np.sqrt(error_density/variance)):.3e}", flush=True)

    time.sleep(1)
    efile.close()

if __name__ == "__main__":
    build()
