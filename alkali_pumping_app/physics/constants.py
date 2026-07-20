"""Atomic, buffer-gas, vapor-pressure, and spin-exchange constants."""

from math import pi, sqrt

import numpy as np

# ============================================================
# 1. Atomic constants
# ============================================================
#
# Frequencies are in MHz.
#
# Hyperfine energy:
#   E_hfs(F) = A K/2
#            + B [3K(K+1)/4 - I(I+1)J(J+1)]
#                / [2I(2I-1) 2J(2J-1)]
# where
#   K = F(F+1) - I(I+1) - J(J+1).
#
# For J = 1/2, B is not used.
#
# The values below are a practical built-in database. You can edit them if you
# need a specific data convention or newer constants.

ATOMS = {
    "Rb87": {
        "I": 3/2,
        "mass_amu": 86.9091805,
        "lambda_D1_nm": 794.978851,
        "lambda_D2_nm": 780.241209,
        "ground": {"J": 1/2, "A": 3417.341305452, "B": 0.0},
        "D1": {"Jp": 1/2, "A": 406.2, "B": 0.0, "gamma_nat_MHz": 5.75},
        "D2": {"Jp": 3/2, "A": 84.7185, "B": 12.4965, "gamma_nat_MHz": 6.07},
    },
    "Rb85": {
        "I": 5/2,
        "mass_amu": 84.9117897,
        "lambda_D1_nm": 794.979,
        "lambda_D2_nm": 780.241,
        "ground": {"J": 1/2, "A": 1011.910813, "B": 0.0},
        "D1": {"Jp": 1/2, "A": 120.527, "B": 0.0, "gamma_nat_MHz": 5.75},
        "D2": {"Jp": 3/2, "A": 25.002, "B": 25.79, "gamma_nat_MHz": 6.07},
    },
    "Cs133": {
        "I": 7/2,
        "mass_amu": 132.90545196,
        "lambda_D1_nm": 894.592959,
        "lambda_D2_nm": 852.347275,
        "ground": {"J": 1/2, "A": 2298.1579425, "B": 0.0},
        "D1": {"Jp": 1/2, "A": 291.9201, "B": 0.0, "gamma_nat_MHz": 4.56},
        "D2": {"Jp": 3/2, "A": 50.28827, "B": -0.4934, "gamma_nat_MHz": 5.23},
    },
    "K39": {
        "I": 3/2,
        "mass_amu": 38.96370649,
        "lambda_D1_nm": 770.108136,
        "lambda_D2_nm": 766.700921,
        "ground": {"J": 1/2, "A": 230.8598601, "B": 0.0},
        "D1": {"Jp": 1/2, "A": 27.775, "B": 0.0, "gamma_nat_MHz": 5.96},
        "D2": {"Jp": 3/2, "A": 6.093, "B": 2.786, "gamma_nat_MHz": 6.04},
    },
}


# Ground-state magnetic constants used for the linear Zeeman shift.
# The nuclear moments are in nuclear magnetons. The hyperfine g_F values are
# calculated in Bohr-magneton units, including the small nuclear contribution.
GROUND_STATE_GJ = 2.00233113
MU_N_OVER_MU_B = 1.0 / 1836.15267343
NUCLEAR_MAGNETIC_MOMENT_MU_N = {
    "Rb87": 2.751818,
    "Rb85": 1.35335171,
    "Cs133": 2.582025,
    "K39": 0.3914662,
}


def ground_hyperfine_lande_g(atom_name, atom, F):
    """Return the ground-state hyperfine Lande g factor for one F manifold."""
    I = float(atom["I"])
    J = float(atom["ground"]["J"])
    F = float(F)
    if F <= 0.0:
        return 0.0

    denominator = 2.0 * F * (F + 1.0)
    electronic_projection = (
        F * (F + 1.0) + J * (J + 1.0) - I * (I + 1.0)
    ) / denominator
    nuclear_projection = (
        F * (F + 1.0) + I * (I + 1.0) - J * (J + 1.0)
    ) / denominator

    mu_I_mu_N = NUCLEAR_MAGNETIC_MOMENT_MU_N.get(atom_name, 0.0)
    g_I_bohr = (mu_I_mu_N / I) * MU_N_OVER_MU_B if I > 0.0 else 0.0
    return float(
        GROUND_STATE_GJ * electronic_projection
        + g_I_bohr * nuclear_projection
    )


def ground_zeeman_shifts_hz(
    atom_name,
    atom,
    ground_states,
    upper_manifold_larmor_hz,
):
    """Return linear ground-state Zeeman shifts nu_B(F,m) in Hz.

    The entered signed frequency is defined as the adjacent-level Larmor
    frequency of the upper ground hyperfine manifold F_+=I+1/2. For every
    displayed manifold,

        nu_B(F,m) = m [g_F/g_(F_+)] nu_B,input.

    Thus the lower ground hyperfine manifold automatically receives its
    physically opposite Zeeman slope, including the small nuclear correction.
    """
    upper_F = max(float(state["F"]) for state in ground_states)
    g_upper = ground_hyperfine_lande_g(atom_name, atom, upper_F)
    if abs(g_upper) < 1e-15:
        raise ValueError("Upper-manifold hyperfine g factor is numerically zero.")

    g_by_F = {
        float(F): ground_hyperfine_lande_g(atom_name, atom, float(F))
        for F in sorted({float(state["F"]) for state in ground_states})
    }
    ratio_by_F = {F: gF / g_upper for F, gF in g_by_F.items()}
    shifts = np.array([
        float(state["m"])
        * ratio_by_F[float(state["F"])]
        * float(upper_manifold_larmor_hz)
        for state in ground_states
    ], dtype=float)

    return shifts, {
        "upper_F": upper_F,
        "g_upper": g_upper,
        "g_by_F": g_by_F,
        "ratio_by_F": ratio_by_F,
        "input_larmor_hz": float(upper_manifold_larmor_hz),
    }


# ============================================================
# 2. N2 pressure coefficients
# ============================================================
#
# Units: MHz/Torr.
#
# width: Lorentzian FWHM broadening coefficient.
# shift: resonance frequency shift coefficient.
#
# A negative shift means red shift of the optical resonance.
#
# These values are good starting values. Keep them editable because published
# coefficients depend on atom, isotope, line, temperature, and convention.

DEFAULT_N2_COEFFS = {
    "Rb87": {
        "D1": {"width": 17.8, "shift": -8.25},
        "D2": {"width": 18.1, "shift": -5.90},
    },
    "Rb85": {
        "D1": {"width": 17.8, "shift": -8.25},
        "D2": {"width": 18.1, "shift": -5.90},
    },
    "Cs133": {
        "D1": {"width": 19.51, "shift": -8.23},
        "D2": {"width": 22.68, "shift": -6.73},
    },
    "K39": {
        "D1": {"width": 18.6, "shift": -6.1},
        "D2": {"width": 17.8, "shift": -5.1},
    },
}


# ============================================================
# 2b. Alkali vapor density and spin-exchange defaults
# ============================================================
#
# The spin-exchange rate is estimated as
#
#     R_SE = n(T) * sigma_SE * v_rel_bar .
#
# The vapor pressure model below is the common older Steck/Nesmeyanov-style
# two-parameter form log10(P_Torr)=A-B/T_K. It is intended as a practical
# rate estimate for this population app. Edit the built-in vapor-pressure or
# cross-section data if you need a different density or collision convention.

VAPOR_PRESSURE_MODELS = {
    "Rb87": {
        "melting_K": 312.46,
        "solid": {"A": 7.738, "B": 4215.0},
        "liquid": {"A": 7.193, "B": 4040.0},
    },
    "Rb85": {
        "melting_K": 312.46,
        "solid": {"A": 7.738, "B": 4215.0},
        "liquid": {"A": 7.193, "B": 4040.0},
    },
    "Cs133": {
        "melting_K": 301.59,
        "solid": {"A": 7.592, "B": 3999.0},
        "liquid": {"A": 7.046, "B": 3830.0},
    },
    "K39": {
        "melting_K": 336.53,
        "solid": {"A": 7.842, "B": 4646.0},
        "liquid": {"A": 7.283, "B": 4453.0},
    },
}

SPIN_EXCHANGE_CROSS_SECTION_CM2 = {
    "Rb87": 1.9e-14,
    "Rb85": 1.7e-14,
    "Cs133": 2.0e-14,
    "K39": 2.0e-14,
}



def alkali_vapor_pressure_torr(atom_name, temperature_C):
    """Estimate saturated alkali vapor pressure in Torr."""
    T_K = float(temperature_C) + 273.15
    if T_K <= 0:
        return 0.0

    model = VAPOR_PRESSURE_MODELS.get(atom_name)
    if model is None:
        return 0.0

    phase = "solid" if T_K < model["melting_K"] else "liquid"
    coeff = model[phase]
    log10_p_torr = coeff["A"] - coeff["B"] / T_K
    return float(10.0 ** log10_p_torr)


def alkali_vapor_density_cm3(atom_name, temperature_C):
    """Estimate saturated alkali vapor number density in cm^-3."""
    T_K = float(temperature_C) + 273.15
    if T_K <= 0:
        return 0.0

    pressure_torr = alkali_vapor_pressure_torr(atom_name, temperature_C)
    pressure_pa = pressure_torr * 133.32236842105263
    kB = 1.380649e-23
    density_m3 = pressure_pa / (kB * T_K)
    return float(density_m3 / 1e6)


def spin_exchange_rate_info(atom_name, atom, temperature_C):
    """Return the temperature-inferred alkali-alkali spin-exchange rate."""
    T_K = float(temperature_C) + 273.15
    if T_K <= 0:
        T_K = 1e-9

    kB = 1.380649e-23
    amu = 1.66053906660e-27
    mass_kg = atom["mass_amu"] * amu

    density_cm3 = alkali_vapor_density_cm3(atom_name, temperature_C)
    pressure_torr = alkali_vapor_pressure_torr(atom_name, temperature_C)
    sigma_cm2 = SPIN_EXCHANGE_CROSS_SECTION_CM2.get(atom_name, 2.0e-14)
    vrel_m_s = sqrt(16.0 * kB * T_K / (pi * mass_kg))
    vrel_cm_s = 100.0 * vrel_m_s
    rate_s = density_cm3 * sigma_cm2 * vrel_cm_s

    return {
        "pressure_torr": pressure_torr,
        "density_cm3": density_cm3,
        "sigma_cm2": sigma_cm2,
        "vrel_cm_s": vrel_cm_s,
        "rate_s": float(rate_s),
    }

