import os
import sys
import numpy as np
import classy


"""Redshift space distortions likelihood.

This module computes the likelihood for RSD measurements compiled in XXX. It is written for a usage in the
ECLAIR suite https://github.com/s-ilic/ECLAIR.

This likelihood was developed for the study XXX. Please find the reference of the measurements therein in Table 1.
"""


# ************
# *** Data ***
# ************
path_to_data = (
    os.path.dirname(os.path.realpath(sys.argv[0])) + "/likelihoods/RSD_fsigma8_all"
)
x, y, erry, omo = np.loadtxt(path_to_data + "/data.txt", comments="#", unpack=True)
invcov = np.linalg.inv(np.loadtxt(path_to_data + "/cov.txt", comments="#", unpack=True))


# ****************
# *** Fiducial ***
# ****************
cosmo_fid = {
    "output": "mPk",
    "z_max_pk": 2,
    "omega_b": 0.022,
    "omega_cdm": 0,
}

# computes the fiducial cosmology of each surveys
fids_class_run = []
for i in range(len(x)):
    cosmo_fid["omega_cdm"] = omo[i] * 0.6777 ** 2 - cosmo_fid["omega_b"]
    fid_run = classy.Class()
    fid_run.set(cosmo_fid)
    fid_run.compute()
    fids_class_run.append(fid_run)


# ***********************************
# *** Alcock-Paczynski correction ***
# ***********************************
def alpha_par(z, run_fid, run_mod):
    res = (run_fid.Hubble(z) / run_fid.h()) / (run_mod.Hubble(z) / run_mod.h())
    return res


def alpha_perp(z, run_fid, run_mod):
    res = (run_mod.angular_distance(z) * run_mod.h()) / (
        run_fid.angular_distance(z) * run_fid.h()
    )
    return res


def correction(z, run_fid, run_mod):
    a = alpha_par(z, run_fid, run_mod) / alpha_perp(z, run_fid, run_mod)
    A = (1 - a) * (5 - a * (1 + a) / 2) / 4
    res = 1 - 2 * A / 7
    return res


# ***********************
# *** Growth function ***
# ***********************
def fsigma8(z, engine):
    res = engine.scale_independent_growth_factor_f(z) * engine.sigma(
        8.0 / engine.h(), z
    )
    return res


# ***************************
# *** Likelihood function ***
# ***************************
def get_loglike(class_input, likes_input, class_run):
    vec_diff = []
    for i, z in enumerate(x):
        q = correction(z, fids_class_run[i], class_run)
        vec_diff.append(y[i] - q * fsigma8(z, class_run))

    lnl = -0.5 * np.dot(np.dot(vec_diff, invcov), vec_diff)

    return lnl
