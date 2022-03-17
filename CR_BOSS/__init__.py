import classy
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d

"""BOSS DR7 and DR12 clustering ratio likelihood.

This module computes the likelihood of the clustering ratio (CR) for the
combination of the BOSS DR7 and DR12 surveys. It is written for a usage in the
ECLAIR suite https://github.com/s-ilic/ECLAIR.

This likelihood was developed for the study XXX
The CR was fisrt developed in https://arxiv.org/abs/1210.2365.
The BOSS DR7 catalogue was released in https://arxiv.org/abs/0812.0649
The BOSS DR12 catalogue was released in https://arxiv.org/abs/1501.00963
The corresponding CR measurements were released in https://arxiv.org/abs/1712.02886.
"""

# ************
# *** Data ***
# ************
# BOSS DR7 release
boss_dr7_z_min = [0.15]
boss_dr7_z_max = [0.43]
boss_dr7_cr = [0.096]
boss_dr7_err = [0.007]

# BOSS DR12 release
boss_dr12_z_min = [0.30, 0.53]
boss_dr12_z_max = [0.53, 0.67]
boss_dr12_cr = [0.094, 0.105]
boss_dr12_err = [0.006, 0.011]

# Merging
data_z_min = np.append(boss_dr7_z_min, boss_dr12_z_min)
data_z_max = np.append(boss_dr7_z_max, boss_dr12_z_max)
data_cr = np.append(boss_dr7_cr, boss_dr12_cr)
data_err = np.append(boss_dr7_err, boss_dr12_err)
data_z_moy = (data_z_max - data_z_min) / 2 + data_z_min
len_data = len(data_z_max)

# The CR will be averaged over 10 values for each bin in the likelihood
div = 10
z_int = (
    np.linspace(data_z_min[0], data_z_max[0], div),
    np.linspace(data_z_min[1], data_z_max[1], div),
    np.linspace(data_z_min[2], data_z_max[2], div),
)


# ****************
# *** Fiducial ***
# ****************
n_fid = 2.1
R_fid = 22
Om0_fid = 0.3
cosmo_fid = {
    "output": "mPk",
    "z_max_pk": 0.67,
    "P_k_max_1/Mpc": 10,
    "Omega_b": 0.156 * Om0_fid,
    "Omega_cdm": 0.839 * Om0_fid,
}
run_fid = classy.Class()
run_fid.set(cosmo_fid)
run_fid.compute()


# ***********************************
# *** Alcock-Paczynski correction ***
# ***********************************
def alpha_par(z, run_mod):
    res = (run_fid.Hubble(z) / run_fid.h()) / (run_mod.Hubble(z) / run_mod.h())
    return res


def alpha_perp(z, run_mod):
    res = (run_mod.angular_distance(z) * run_mod.h()) / (
        run_fid.angular_distance(z) * run_fid.h()
    )
    return res


def alpha(z, run_mod):
    res = np.power(
        np.square(alpha_perp(z, run_mod)) * alpha_par(z, run_mod),
        1.0 / 3,
    )
    return res


# *******************
# *** CR function ***
# *******************
def cr(z, class_run):

    # Computing matter power spectrum
    pk, k, zz = class_run.get_pk_and_k_and_z(nonlinear=False)

    # Preparing change variable to log k for an integration with better precision
    k_int = np.exp(np.linspace(np.log(min(k)), np.log(max(k)), 1000))

    # Interpolation for better integration
    pk = np.transpose(interp1d(zz, pk)(z))
    pk = np.transpose(interp1d(k, pk, fill_value="extrapolate")(k_int))

    # Applying the AP correction to the fiducial smoothing radius
    R = alpha(z, class_run) * R_fid

    # Computing k*R with Converting R into Mpc
    kR = k_int * R / class_run.h()

    # Variance
    W = 3 * (kR * np.cos(kR) - np.sin(kR)) / kR ** 3
    sigma_R = k_int ** 3 * pk * W ** 2

    # 2-pt correlation function
    kRn = kR * n_fid
    xi_R = sigma_R * np.sin(kRn) / kRn

    # Integration
    cr = trapz(xi_R, np.log(k_int)) / trapz(sigma_R, np.log(k_int))

    return cr


# ***************************
# *** Likelihood function ***
# ***************************
def get_loglike(class_input, likes_input, class_run):
    lnl = 0
    for i, (y, err) in enumerate(zip(data_cr, data_err)):
        cr_mod = []
        for z in z_int[i]:
            cr_mod.append(cr(z, class_run))
        lnl += -0.5 * (np.mean(cr_mod) - y) ** 2.0 / err ** 2.0
    return lnl
