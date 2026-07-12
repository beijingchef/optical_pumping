# alkali_pumping_v3.1.py
#
# Streamlit app:
#   Steady-state ground-state population distribution of alkali vapors
#   with N2 buffer gas, three monochromatic optical-pumping beams,
#   pressure broadening, pressure shift, electron-randomization relaxation,
#   and alkali-alkali spin-exchange relaxation.
#
# Run:
#   pip install streamlit numpy scipy sympy pandas matplotlib
#   streamlit run alkali_pumping_v3.1.py
#
# Model:
#   dp/dt = [L_op,1 + L_op,2 + L_op,3 + Gamma_ER (M_ER - I)] p
#          + R_SE(T) [M_SE[p] - I] p
#
# The app is population-only. It does not keep Zeeman coherences, excited-state
# coherences, spin-exchange coherences, or optical propagation effects. Spin exchange is
# included as a population-only mean-field term.

import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from math import sqrt, pi
from scipy.special import wofz
from scipy.linalg import lstsq
from sympy.physics.wigner import wigner_3j, wigner_6j
from sympy import S


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


# ============================================================
# 3. Angular momentum utilities
# ============================================================

def allowed_F(I, J):
    fmin = abs(I - J)
    fmax = I + J
    n = int(round(fmax - fmin)) + 1
    return [fmin + k for k in range(n)]


def m_values(F):
    return [float(m) for m in np.arange(-F, F + 1, 1)]


def hfs_energy_MHz(I, J, F, A, B=0.0):
    K = F * (F + 1) - I * (I + 1) - J * (J + 1)
    E = 0.5 * A * K

    if abs(B) > 0 and J > 0.5 and I > 0.5:
        numerator = 0.75 * K * (K + 1) - I * (I + 1) * J * (J + 1)
        denominator = 2 * I * (2 * I - 1) * 2 * J * (2 * J - 1)
        E += B * numerator / denominator

    return float(E)


def build_ground_states(atom):
    I = atom["I"]
    Jg = atom["ground"]["J"]
    A = atom["ground"]["A"]
    B = atom["ground"]["B"]

    states = []
    for F in allowed_F(I, Jg):
        E = hfs_energy_MHz(I, Jg, F, A, B)
        for m in m_values(F):
            states.append({
                "F": float(F),
                "m": float(m),
                "E": E,
                "label": f"F={F:g}, m={m:g}",
            })
    return states


def build_excited_states(atom, line):
    I = atom["I"]
    Jp = atom[line]["Jp"]
    A = atom[line]["A"]
    B = atom[line]["B"]

    states = []
    for Fp in allowed_F(I, Jp):
        E = hfs_energy_MHz(I, Jp, Fp, A, B)
        for mp in m_values(Fp):
            states.append({
                "F": float(Fp),
                "m": float(mp),
                "E": E,
                "label": f"F'={Fp:g}, m'={mp:g}",
            })
    return states


def hyperfine_transition_allowed(Fg, Fe):
    if abs(Fe - Fg) > 1:
        return False
    if Fg == 0 and Fe == 0:
        return False
    return True


def dipole_strength(I, Jg, Je, Fg, mg, Fe, me, q):
    """
    Relative squared dipole matrix element:
        |<Fe me | d_q | Fg mg>|^2
    up to a common reduced electronic matrix element.
    """
    if abs(me - mg - q) > 1e-9:
        return 0.0
    if abs(q) > 1:
        return 0.0

    try:
        three_j = float(wigner_3j(S(Fe), S(1), S(Fg), S(-me), S(q), S(mg)))
        six_j = float(wigner_6j(S(Je), S(Fe), S(I), S(Fg), S(Jg), S(1)))
    except Exception:
        return 0.0

    strength = (2 * Fe + 1) * (2 * Fg + 1) * three_j**2 * six_j**2
    return max(0.0, float(strength))


# ============================================================
# 4. Light geometry and polarization
# ============================================================

def unit_vector(axis):
    if axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=complex)
    if axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=complex)
    if axis == "z":
        return np.array([0.0, 0.0, 1.0], dtype=complex)
    raise ValueError(axis)


def transverse_basis_for_k(k_axis):
    """
    Return two transverse unit vectors a,b such that a x b = k.
    This defines the local beam frame.
    """
    if k_axis == "z":
        return unit_vector("x"), unit_vector("y")
    if k_axis == "x":
        return unit_vector("y"), unit_vector("z")
    if k_axis == "y":
        return unit_vector("z"), unit_vector("x")
    raise ValueError(k_axis)


def allowed_polarizations(k_axis):
    """
    For each propagation direction, allow sigma+, sigma-, and the two lab-linear
    directions perpendicular to k.
    """
    if k_axis == "z":
        return ["sigma+", "sigma-", "linear x", "linear y"]
    if k_axis == "x":
        return ["sigma+", "sigma-", "linear y", "linear z"]
    if k_axis == "y":
        return ["sigma+", "sigma-", "linear z", "linear x"]
    raise ValueError(k_axis)


def lab_e_field(k_axis, pol):
    """
    Complex electric field in lab x,y,z coordinates.

    Convention:
      If k = z and quantization axis = z, sigma+ gives q=+1.
    """
    a, b = transverse_basis_for_k(k_axis)

    if pol == "sigma+":
        return -(a + 1j * b) / sqrt(2)
    if pol == "sigma-":
        return (a - 1j * b) / sqrt(2)
    if pol.startswith("linear"):
        ax = pol.split()[-1]
        return unit_vector(ax)

    raise ValueError(pol)


def local_components(E_lab, q_axis):
    """
    Components in a local frame with local z along the chosen quantization axis.
    """
    if q_axis == "z":
        ux, uy, uz = unit_vector("x"), unit_vector("y"), unit_vector("z")
    elif q_axis == "x":
        ux, uy, uz = unit_vector("y"), unit_vector("z"), unit_vector("x")
    elif q_axis == "y":
        ux, uy, uz = unit_vector("z"), unit_vector("x"), unit_vector("y")
    else:
        raise ValueError(q_axis)

    return np.array([
        np.vdot(ux, E_lab),
        np.vdot(uy, E_lab),
        np.vdot(uz, E_lab),
    ], dtype=complex)


def spherical_weights_relative_to_quant_axis(k_axis, pol, q_axis):
    """
    Return |E_q|^2 for q=-1,0,+1 relative to the selected quantization axis.
    """
    E_lab = lab_e_field(k_axis, pol)
    Ex, Ey, Ez = local_components(E_lab, q_axis)

    E_plus = -(Ex - 1j * Ey) / sqrt(2)
    E_zero = Ez
    E_minus = (Ex + 1j * Ey) / sqrt(2)

    weights = {
        -1: abs(E_minus)**2,
        0: abs(E_zero)**2,
        +1: abs(E_plus)**2,
    }

    s = sum(weights.values())
    if s <= 0:
        return {-1: 0.0, 0: 0.0, +1: 0.0}
    return {q: float(w / s) for q, w in weights.items()}


# ============================================================
# 5. Line shape
# ============================================================

def doppler_fwhm_MHz(atom, line, temperature_C):
    """
    Doppler FWHM in MHz:
        Delta_nu_D = 2 nu0 sqrt(2 kT ln2 / mc^2)
    """
    kB = 1.380649e-23
    c = 299792458.0
    amu = 1.66053906660e-27

    T = temperature_C + 273.15
    mass = atom["mass_amu"] * amu
    lam = atom[f"lambda_{line}_nm"] * 1e-9
    nu0 = c / lam

    fwhm = 2 * nu0 * sqrt(2 * kB * T * np.log(2) / (mass * c**2))
    return fwhm / 1e6




def line_center_frequency_MHz(atom, line):
    """Absolute zero-pressure fine-structure D-line frequency in MHz."""
    c = 299792458.0
    lam = atom[f"lambda_{line}_nm"] * 1e-9
    return c / lam / 1e6


def MHz_to_THz(freq_MHz):
    return freq_MHz / 1e6

def complex_voigt_response_relative(delta_MHz, lorentz_fwhm_MHz, doppler_fwhm_MHz_val):
    """Complex Voigt response normalized to the on-resonance absorption.

    The real part is the absorption profile used for optical pumping, normalized
    to Re[response](Delta=0)=1. The imaginary part is the corresponding
    Doppler-averaged dispersive profile with the same normalization.

    In the zero-Doppler limit, this returns
        absorption  = gamma^2 / (Delta^2 + gamma^2)
        dispersion  = gamma Delta / (Delta^2 + gamma^2)
    where gamma is the Lorentzian HWHM. Therefore
        0.5 * dispersion = absorption * Delta / Gamma_FWHM,
    which reproduces the old far-wing two-level formula when Doppler broadening
    is negligible.
    """
    gamma_hwhm = lorentz_fwhm_MHz / 2.0
    sigma = doppler_fwhm_MHz_val / (2 * sqrt(2 * np.log(2)))

    if gamma_hwhm <= 0:
        gamma_hwhm = 1e-12

    if sigma <= 1e-12:
        denom = delta_MHz**2 + gamma_hwhm**2
        absorption = gamma_hwhm**2 / denom
        dispersion = gamma_hwhm * delta_MHz / denom
        return complex(float(absorption), float(dispersion))

    z = (delta_MHz + 1j * gamma_hwhm) / (sigma * sqrt(2))
    z0 = (1j * gamma_hwhm) / (sigma * sqrt(2))

    W = wofz(z) / (sigma * sqrt(2 * pi))
    W0 = wofz(z0) / (sigma * sqrt(2 * pi))
    absorption0 = float(np.real(W0))

    if absorption0 <= 0:
        return 0.0 + 0.0j

    response = W / absorption0
    return complex(float(np.real(response)), float(np.imag(response)))


def voigt_profile_relative(delta_MHz, lorentz_fwhm_MHz, doppler_fwhm_MHz_val):
    """
    Dimensionless Voigt absorption profile normalized to V(0)=1.
    delta_MHz is the laser detuning from the pressure-shifted hyperfine transition.
    """
    response = complex_voigt_response_relative(
        delta_MHz,
        lorentz_fwhm_MHz,
        doppler_fwhm_MHz_val,
    )
    return float(max(0.0, response.real))


def voigt_dispersion_relative(delta_MHz, lorentz_fwhm_MHz, doppler_fwhm_MHz_val):
    """
    Doppler-averaged dispersive profile with the same normalization as
    voigt_profile_relative.
    """
    response = complex_voigt_response_relative(
        delta_MHz,
        lorentz_fwhm_MHz,
        doppler_fwhm_MHz_val,
    )
    return float(response.imag)


# ============================================================
# 6. Transition tables and optical pumping
# ============================================================

def transition_shift_MHz(g_state, e_state):
    """
    Hyperfine transition shift relative to the fine-structure D-line center:
        delta_hfs(Fg -> Fe) = E_e(Fe) - E_g(Fg).
    """
    return float(e_state["E"] - g_state["E"])


def hyperfine_transition_table(
    atom,
    n2_pressure_torr,
    n2_coeffs,
    allowed_only=True,
    pump_beams=None,
):
    """
    Table of ground-hyperfine to excited-hyperfine transition centers.

    Detunings are relative to the corresponding zero-pressure D1 or D2
    fine-structure line center.

    The pressure-shifted detuning is:
        delta_with_N2 = delta_hfs + beta_N2 P_N2.

    Absolute optical frequencies are shown in MHz.
    """
    rows = []
    pump_beams = pump_beams or []

    I = atom["I"]
    Jg = atom["ground"]["J"]
    Ag = atom["ground"]["A"]
    Bg = atom["ground"]["B"]

    ground_Fs = []
    for Fg in allowed_F(I, Jg):
        Eg = hfs_energy_MHz(I, Jg, Fg, Ag, Bg)
        ground_Fs.append({"F": float(Fg), "E": Eg})

    for line in ["D1", "D2"]:
        Jp = atom[line]["Jp"]
        Ae = atom[line]["A"]
        Be = atom[line]["B"]

        line_center_MHz = line_center_frequency_MHz(atom, line)
        pressure_shift = n2_coeffs[line]["shift"] * n2_pressure_torr
        pressure_width = n2_coeffs[line]["width"] * n2_pressure_torr
        total_lorentz = atom[line]["gamma_nat_MHz"] + pressure_width
        doppler = doppler_fwhm_MHz(atom, line, st.session_state.get("temperature_C_for_table", 25.0))

        for g in ground_Fs:
            Fg = g["F"]
            Eg = g["E"]

            for Fe in allowed_F(I, Jp):
                if allowed_only and not hyperfine_transition_allowed(Fg, Fe):
                    continue

                Ee = hfs_energy_MHz(I, Jp, Fe, Ae, Be)
                det0 = Ee - Eg
                detP = det0 + pressure_shift
                transition_abs_MHz = line_center_MHz + detP

                pump1_abs_MHz = np.nan
                pump2_abs_MHz = np.nan
                pump3_abs_MHz = np.nan
                for beam in pump_beams:
                    if beam.get("line") != line:
                        continue
                    selected = beam.get("selected_transition") or {}
                    try:
                        selected_Fg = float(selected.get("Fg"))
                        selected_Fe = float(selected.get("Fe"))
                    except (TypeError, ValueError):
                        continue
                    if abs(selected_Fg - float(Fg)) > 1e-9 or abs(selected_Fe - float(Fe)) > 1e-9:
                        continue

                    pump_abs_MHz = line_center_MHz + float(beam.get("detuning", 0.0))
                    if beam.get("name") == "Beam 1":
                        pump1_abs_MHz = pump_abs_MHz
                    elif beam.get("name") == "Beam 2":
                        pump2_abs_MHz = pump_abs_MHz
                    elif beam.get("name") == "Beam 3":
                        pump3_abs_MHz = pump_abs_MHz

                rows.append({
                    "Line": line,
                    "Fg": f"{Fg:g}",
                    "F'": f"{Fe:g}",
                    "nu_D_absolute": line_center_MHz,
                    "detuning_zero_pressure": det0,
                    "N2_shift": pressure_shift,
                    "detuning_with_N2": detP,
                    "transition_frequency_with_N2": transition_abs_MHz,
                    "pump_1_frequency": pump1_abs_MHz,
                    "pump_2_frequency": pump2_abs_MHz,
                    "pump_3_frequency": pump3_abs_MHz,
                    "lorentz_FWHM_total": total_lorentz,
                    "doppler_FWHM": doppler,
                    "beta_width": n2_coeffs[line]["width"],
                    "beta_shift": n2_coeffs[line]["shift"],
                })

    return pd.DataFrame(rows)


def hyperfine_transition_choices(atom, line, n2_pressure_torr, n2_coeffs, allowed_only=True):
    """
    Return selectable hyperfine transitions for one optical line.

    Each item contains:
      label: text shown in the UI
      det0: transition center relative to the zero-pressure D-line center
      detP: pressure-shifted transition center relative to the zero-pressure D-line center
    """
    rows = []

    I = atom["I"]
    Jg = atom["ground"]["J"]
    Ag = atom["ground"]["A"]
    Bg = atom["ground"]["B"]
    Jp = atom[line]["Jp"]
    Ae = atom[line]["A"]
    Be = atom[line]["B"]

    pressure_shift = n2_coeffs[line]["shift"] * n2_pressure_torr

    for Fg in allowed_F(I, Jg):
        Eg = hfs_energy_MHz(I, Jg, Fg, Ag, Bg)
        for Fe in allowed_F(I, Jp):
            if allowed_only and not hyperfine_transition_allowed(Fg, Fe):
                continue
            Ee = hfs_energy_MHz(I, Jp, Fe, Ae, Be)
            det0 = float(Ee - Eg)
            detP = float(det0 + pressure_shift)
            # Keep the UI label independent of N2 pressure so Streamlit preserves
            # the user's selected reference transition when pressure changes.
            label = f"{Fg:g}→{Fe:g}"
            rows.append({
                "line": line,
                "Fg": float(Fg),
                "Fe": float(Fe),
                "det0": det0,
                "detP": detP,
                "label": label,
            })

    return rows


def transition_choice_labels(atom, line, n2_pressure_torr, n2_coeffs, allowed_only=True):
    return [
        row["label"]
        for row in hyperfine_transition_choices(
            atom, line, n2_pressure_torr, n2_coeffs, allowed_only=allowed_only
        )
    ]


def default_transition_label(atom, line, n2_pressure_torr, n2_coeffs, Fg_target, Fe_target, allowed_only=True):
    """Return the UI label for a requested hyperfine reference transition."""
    choices = hyperfine_transition_choices(
        atom, line, n2_pressure_torr, n2_coeffs, allowed_only=allowed_only
    )
    for row in choices:
        if abs(float(row["Fg"]) - float(Fg_target)) < 1e-9 and abs(float(row["Fe"]) - float(Fe_target)) < 1e-9:
            return row["label"]
    return choices[0]["label"] if choices else None


def absolute_detuning_from_transition_choice(
    atom,
    line,
    transition_label,
    relative_detuning_MHz,
    n2_pressure_torr,
    n2_coeffs,
    allowed_only=True,
):
    """
    Convert UI setting into the detuning expected by build_optical_L.

    The user sets Δ relative to a selected pressure-shifted hyperfine transition.
    build_optical_L expects ν_L - ν_D,zero-pressure. Therefore

        Δ_abs = [ν(F→F') - ν_D,zero-pressure with N₂] + Δ_relative.
    """
    choices = hyperfine_transition_choices(
        atom, line, n2_pressure_torr, n2_coeffs, allowed_only=allowed_only
    )
    if not choices:
        return float(relative_detuning_MHz), None

    selected = next((row for row in choices if row["label"] == transition_label), choices[0])
    return float(selected["detP"] + relative_detuning_MHz), selected


CONDITION_SCHEMA_VERSION = "2.23"
CONDITION_KEYS = (
    "condition_name",
    "atom_name", "gamma_ER", "q_axis", "temperature_C_for_table", "n2_pressure_torr",
    "include_spin_exchange",
    "D1_width", "D2_width", "D1_shift", "D2_shift",
    "line1", "transition1", "det_rel1", "rate1", "k1", "pol1",
    "line2", "transition2", "det_rel2", "rate2", "k2", "pol2",
    "line3", "transition3", "det_rel3", "rate3", "k3", "pol3",
    "show_allowed_only", "show_rate_matrices",
)


# Built-in startup condition, taken from ps400(1).json.
DEFAULT_STARTUP_CONDITION = {
    "condition_name": "ps400",
    "atom_name": "Rb87",
    "gamma_ER": 2.0,
    "q_axis": "z",
    "temperature_C_for_table": 23.0,
    "n2_pressure_torr": 0.0,
    "include_spin_exchange": True,
    "D1_width": 17.8,
    "D2_width": 18.1,
    "D1_shift": -8.25,
    "D2_shift": -5.9,
    "line1": "D1",
    "transition1": "1→2",
    "det_rel1": 0.0,
    "rate1": 1200.0,
    "k1": "x",
    "pol1": "linear z",
    "line2": "D1",
    "transition2": "2→2",
    "det_rel2": 400.0,
    "rate2": 400.0,
    "k2": "x",
    "pol2": "linear z",
    "line3": "D1",
    "transition3": "2→2",
    "det_rel3": 0.0,
    "rate3": 0.0,
    "k3": "x",
    "pol3": "linear z",
    "show_allowed_only": True,
    "show_rate_matrices": False,
}


def clean_condition_name(value):
    """Return a nonempty condition name without a .json extension."""
    name = str(value or "").strip()
    if name.lower().endswith(".json"):
        name = name[:-5].rstrip()
    return name or "default"



def build_condition_payload(values):
    """Build the strict v2.23 JSON payload from current condition values."""
    conditions = {key: values.get(key) for key in CONDITION_KEYS}
    conditions["condition_name"] = clean_condition_name(
        conditions.get("condition_name")
    )
    return {
        "app": "alkali_pumping",
        "format": "alkali_pumping_conditions",
        "version": CONDITION_SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "conditions": conditions,
    }


def current_condition_values(condition_name=None):
    """Collect all current sidebar conditions, including condition name."""
    values = {key: st.session_state.get(key) for key in CONDITION_KEYS}
    if condition_name is not None:
        values["condition_name"] = condition_name
    return values


def apply_loaded_condition_dict(payload):
    """Apply a strict alkali_pumping v2.23 condition file to session state."""
    if not isinstance(payload, dict):
        raise ValueError("The loaded file is not a JSON object.")
    if payload.get("app") != "alkali_pumping":
        raise ValueError("This is not an alkali_pumping condition file.")
    if payload.get("format") != "alkali_pumping_conditions":
        raise ValueError("The JSON file does not use the v2.23 condition format.")

    conditions = payload.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("The JSON file does not contain a conditions object.")

    missing = [key for key in CONDITION_KEYS if key not in conditions]
    if missing:
        raise ValueError(
            "The condition file is missing required fields: " + ", ".join(missing)
        )

    loaded_name = clean_condition_name(conditions["condition_name"])
    for key in CONDITION_KEYS:
        value = conditions[key]
        if value is not None:
            st.session_state[key] = value

    # Prevent atom-change default logic from overwriting loaded N2 coefficients.
    st.session_state["_last_atom_name_for_defaults"] = conditions["atom_name"]
    return loaded_name


def load_condition_callback():
    """Load a selected JSON in a callback, before keyed widgets are instantiated."""
    uploaded = st.session_state.get("condition_file_uploader")
    if uploaded is None:
        return

    try:
        payload = json.loads(uploaded.getvalue().decode("utf-8"))
        loaded_name = apply_loaded_condition_dict(payload)
        st.session_state["_condition_load_message"] = (
            f"Loaded condition: {loaded_name}"
        )
        st.session_state.pop("_condition_load_error", None)
    except Exception as exc:
        st.session_state["_condition_load_error"] = str(exc)
        st.session_state.pop("_condition_load_message", None)


def excited_decay_branching(atom, line, ground_states, e_state):
    """
    Branching probabilities from one excited Zeeman state to all ground states.
    Sum over emitted photon q=-1,0,+1.
    """
    I = atom["I"]
    Jg = atom["ground"]["J"]
    Je = atom[line]["Jp"]

    weights = []
    for g in ground_states:
        w = 0.0
        for q in [-1, 0, +1]:
            w += dipole_strength(I, Jg, Je, g["F"], g["m"], e_state["F"], e_state["m"], q)
        weights.append(w)

    weights = np.array(weights, dtype=float)
    total = weights.sum()

    if total <= 0:
        return np.zeros_like(weights)
    return weights / total


def build_optical_L(
    atom,
    line,
    ground_states,
    detuning_MHz,
    pump_rate_s,
    k_axis,
    pol,
    q_axis,
    n2_pressure_torr,
    temperature_C,
    n2_width_MHz_per_torr,
    n2_shift_MHz_per_torr,
    selected_transition=None,
    normalize_to_unpolarized=True,
):
    """
    Build optical-pumping generator L for one beam.

    detuning_MHz:
        Laser detuning from the zero-pressure D1 or D2 fine-structure center.

    pump_rate_s:
        If normalize_to_unpolarized=True, this is the pumping rate for the
        selected hyperfine transition Fg -> Fe. More precisely, it is the mean
        absorption/depopulation rate from atoms unpolarized within the selected
        ground hyperfine level Fg into the selected excited hyperfine level Fe,
        summed over the excited Zeeman sublevels m'. Nearby hyperfine
        transitions are still included in the subsequent dynamics after this
        selected-transition normalization sets the optical intensity scale.
    """
    N = len(ground_states)
    L = np.zeros((N, N), dtype=float)

    excited_states = build_excited_states(atom, line)
    I = atom["I"]
    Jg = atom["ground"]["J"]
    Je = atom[line]["Jp"]

    q_weights = spherical_weights_relative_to_quant_axis(k_axis, pol, q_axis)

    gamma_nat = atom[line]["gamma_nat_MHz"]

    # Pressure broadening and shift.
    pressure_width_MHz = n2_width_MHz_per_torr * n2_pressure_torr
    pressure_shift_MHz = n2_shift_MHz_per_torr * n2_pressure_torr

    lorentz_fwhm = gamma_nat + pressure_width_MHz
    doppler_fwhm = doppler_fwhm_MHz(atom, line, temperature_C)

    # Raw excitation rates from each ground sublevel to each excited sublevel.
    # raw_light_shift_ge_angular stores the matching Doppler-averaged dispersive
    # light-shift contribution before intensity normalization.
    R_ge = np.zeros((N, len(excited_states)), dtype=float)
    delta_ge_MHz = np.zeros((N, len(excited_states)), dtype=float)
    raw_light_shift_ge_angular = np.zeros((N, len(excited_states)), dtype=float)

    for gi, g in enumerate(ground_states):
        for ei, e in enumerate(excited_states):
            hfs_shift = transition_shift_MHz(g, e)

            # Laser detuning relative to the actual pressure-shifted transition.
            delta_to_transition = detuning_MHz - (hfs_shift + pressure_shift_MHz)
            delta_ge_MHz[gi, ei] = delta_to_transition

            response = complex_voigt_response_relative(
                delta_to_transition,
                lorentz_fwhm,
                doppler_fwhm,
            )
            profile = max(0.0, response.real)
            dispersion = response.imag

            strength_sum = 0.0
            for q, wq in q_weights.items():
                strength_sum += wq * dipole_strength(
                    I, Jg, Je,
                    g["F"], g["m"],
                    e["F"], e["m"],
                    q,
                )

            R_ge[gi, ei] = strength_sum * profile
            # The 0.5 factor makes the zero-Doppler limit exactly match the old
            # relation R_ge * Delta / Gamma_FWHM, while using the Doppler-averaged
            # dispersive Voigt response at finite Doppler width.
            raw_light_shift_ge_angular[gi, ei] = 0.5 * strength_sum * dispersion

    raw_leave = R_ge.sum(axis=1)

    reference_Fg = None
    reference_Fe = None
    if selected_transition is not None:
        try:
            reference_Fg = float(selected_transition.get("Fg"))
            reference_Fe = float(selected_transition.get("Fe"))
        except Exception:
            reference_Fg = None
            reference_Fe = None

    if reference_Fg is not None:
        reference_ground_indices = [
            gi for gi, g in enumerate(ground_states)
            if abs(float(g["F"]) - reference_Fg) < 1e-9
        ]
    else:
        reference_ground_indices = list(range(N))

    if reference_Fe is not None:
        reference_excited_indices = [
            ei for ei, e in enumerate(excited_states)
            if abs(float(e["F"]) - reference_Fe) < 1e-9
        ]
    else:
        reference_excited_indices = list(range(len(excited_states)))

    if reference_ground_indices and reference_excited_indices:
        reference_raw_avg_selected_transition = (
            R_ge[np.ix_(reference_ground_indices, reference_excited_indices)]
            .sum(axis=1)
            .mean()
        )
    else:
        reference_raw_avg_selected_transition = 0.0

    if normalize_to_unpolarized:
        scale = (
            pump_rate_s / reference_raw_avg_selected_transition
            if reference_raw_avg_selected_transition > 0
            else 0.0
        )
    else:
        scale = pump_rate_s

    R_ge *= scale
    light_shift_ge_angular = raw_light_shift_ge_angular * scale

    # g -> e -> g' redistribution.
    branching_cache = []
    for e in excited_states:
        branching_cache.append(excited_decay_branching(atom, line, ground_states, e))

    for gi in range(N):
        leave_rate = R_ge[gi, :].sum()
        if leave_rate <= 0:
            continue

        for ei in range(len(excited_states)):
            rate_ge = R_ge[gi, ei]
            if rate_ge <= 0:
                continue

            b = branching_cache[ei]
            for gj in range(N):
                L[gj, gi] += rate_ge * b[gj]

        L[gi, gi] -= leave_rate

    return L, {
        "q_weights": q_weights,
        "gamma_nat_MHz": gamma_nat,
        "pressure_width_MHz": pressure_width_MHz,
        "pressure_shift_MHz": pressure_shift_MHz,
        "lorentz_fwhm_MHz": lorentz_fwhm,
        "doppler_fwhm_MHz": doppler_fwhm,
        "n2_width_MHz_per_torr": n2_width_MHz_per_torr,
        "n2_shift_MHz_per_torr": n2_shift_MHz_per_torr,
        "R_ge": R_ge,
        "delta_ge_MHz": delta_ge_MHz,
        "light_shift_ge_angular": light_shift_ge_angular,
        "excited_states": excited_states,
        "reference_Fg": reference_Fg,
        "reference_Fe": reference_Fe,
        "reference_ground_indices": reference_ground_indices,
        "reference_excited_indices": reference_excited_indices,
        "reference_raw_avg_selected_transition": reference_raw_avg_selected_transition,
        "normalization_scale": scale,
    }


def light_shift_is_diagonal_for_beam(k_axis, pol, q_axis, tolerance=1e-9):
    """Return True only when the light has a single spherical component q.

    In that case the population-basis AC-Stark Hamiltonian is diagonal in the
    selected |F,m> basis and commutes with F_z along the chosen quantization
    axis. If the field has multiple spherical components, Raman cross terms can
    couple different m states, so this population-only app does not report a
    diagonal light shift.
    """
    q_weights = spherical_weights_relative_to_quant_axis(k_axis, pol, q_axis)
    return max(q_weights.values()) >= 1.0 - tolerance


def total_light_shift_Hz_from_diagnostics(ground_states, beam_inputs, diagnostics):
    """Calculate diagonal light shifts for all ground states.

    The optical pumping rates are proportional to the real part of the complex
    Voigt response. The light shifts are calculated from the matching imaginary
    part of the same complex Voigt response. This avoids using the far-wing
    Lorentzian approximation Delta/Gamma when the detuning is comparable to the
    Doppler width.
    """
    if len(diagnostics) == 0:
        return np.zeros(len(ground_states), dtype=float), True

    # Require every active pump beam to be diagonal in the chosen quantization basis.
    for b, _info in diagnostics:
        if not light_shift_is_diagonal_for_beam(b["k_axis"], b["pol"], b["q_axis"]):
            return np.full(len(ground_states), np.nan, dtype=float), False

    shift_angular = np.zeros(len(ground_states), dtype=float)
    for _b, info in diagnostics:
        # No fallback to the old far-wing Lorentzian Delta/Gamma formula.
        # Every active-beam diagnostic must carry the corrected Doppler-averaged
        # complex-Voigt light shift calculated in build_optical_L().
        shift_angular += np.sum(info["light_shift_ge_angular"], axis=1)

    shift_Hz = shift_angular / (2.0 * np.pi)
    return shift_Hz, True


# ============================================================
# 7. Electron-randomization matrix
# ============================================================

def cg_coeff_F_to_mI_mS(I, F, m, mI, mS):
    """
    Coupling coefficient:
        <I mI, S mS | F m>, with S=1/2.
    """
    S_el = 1/2

    if abs(mI + mS - m) > 1e-9:
        return 0.0

    try:
        exponent = int(round(I - S_el + m))
        val = (
            (-1) ** exponent
            * sqrt(2 * F + 1)
            * float(wigner_3j(S(I), S(S_el), S(F), S(mI), S(mS), S(-m)))
        )
        return float(val)
    except Exception:
        return 0.0


def build_ER_matrix(atom, ground_states):
    """
    M_ER maps initial populations p_b to post-ER populations p_a.

    ER model:
        electron spin is randomized,
        nuclear spin population is preserved.
    """
    I = atom["I"]
    mI_list = m_values(I)
    mS_list = [-0.5, +0.5]
    N = len(ground_states)

    prob_mI_given_state = np.zeros((N, len(mI_list)), dtype=float)

    for ai, a in enumerate(ground_states):
        for ii, mI in enumerate(mI_list):
            s_val = 0.0
            for mS in mS_list:
                c = cg_coeff_F_to_mI_mS(I, a["F"], a["m"], mI, mS)
                s_val += c * c
            prob_mI_given_state[ai, ii] = s_val

    M = np.zeros((N, N), dtype=float)

    for a in range(N):
        for b in range(N):
            total = 0.0
            for ii, _mI in enumerate(mI_list):
                total += (
                    prob_mI_given_state[b, ii]
                    * 0.5
                    * prob_mI_given_state[a, ii]
                )
            M[a, b] = total

    # Normalize columns.
    colsum = M.sum(axis=0)
    for j in range(N):
        if colsum[j] > 0:
            M[:, j] /= colsum[j]

    return M


def coupled_basis_amplitudes(atom, ground_states):
    """Return real amplitudes <I mI, S mS | F m> for all ground states."""
    I = atom["I"]
    mI_list = m_values(I)
    mS_list = [-0.5, +0.5]
    amplitudes = np.zeros(
        (len(ground_states), len(mI_list), len(mS_list)),
        dtype=float,
    )

    for ai, state in enumerate(ground_states):
        for ii, mI in enumerate(mI_list):
            for si, mS in enumerate(mS_list):
                amplitudes[ai, ii, si] = cg_coeff_F_to_mI_mS(
                    I, state["F"], state["m"], mI, mS
                )

    return amplitudes


def er_population_fractional_relaxation_rates(M_ER, p_steady, gamma_ER):
    """Return the signed ER fractional rate of each steady-state population.

    For a diagonal steady-state density matrix,

        (d p_a / dt)_ER = gamma_ER [(M_ER p)_a - p_a].

    The reported rate is

        Gamma_a^(ER,net) = -(d p_a / dt)_ER / p_a.

    Positive values mean ER removes population from the state; negative values
    mean ER replenishes it. States with numerically zero population are blank.
    """
    p = np.asarray(p_steady, dtype=float)
    er_derivative = float(gamma_ER) * (np.asarray(M_ER, dtype=float) @ p - p)
    rates = np.full_like(p, np.nan, dtype=float)
    populated = p > 1e-15
    rates[populated] = -er_derivative[populated] / p[populated]
    return rates


def er_adjacent_coherence_self_relaxation_rates(atom, ground_states, gamma_ER):
    """Return ER self-decay rates for infinitesimal rho_(m,m-1) coherences.

    The electron-randomization channel is

        E_ER(rho) = Tr_S(rho) tensor I_S/2.

    For each adjacent coherence |a><b| within one F manifold, this function
    evaluates its self-retention coefficient

        k_ab = <a| E_ER(|a><b|) |b>,

    and reports gamma_ER (1-k_ab). ER can also couple coherences with the same
    Delta m, so this is the local/self-decay coefficient of an infinitesimal
    coherence perturbation about the final diagonal steady state.
    """
    amplitudes = coupled_basis_amplitudes(atom, ground_states)
    rates = np.full(len(ground_states), np.nan, dtype=float)

    state_index = {
        (float(state["F"]), float(state["m"])): idx
        for idx, state in enumerate(ground_states)
    }

    for a_idx, a in enumerate(ground_states):
        key_b = (float(a["F"]), float(a["m"] - 1.0))
        b_idx = state_index.get(key_b)
        if b_idx is None:
            continue

        C_a = amplitudes[a_idx]
        C_b = amplitudes[b_idx]

        # Nuclear operator after tracing the electron spin from |a><b|.
        rho_I = np.einsum("is,js->ij", C_a, C_b)

        # Project (rho_I tensor I_S/2) back onto <a| ... |b>.
        self_retention = 0.5 * np.einsum("is,ij,js->", C_a, rho_I, C_b)
        rates[a_idx] = float(gamma_ER) * (1.0 - float(self_retention))

    return rates


# ============================================================
# 7b. Population-only spin-exchange matrix
# ============================================================

def hyperfine_uncoupled_probabilities(atom, ground_states):
    """Return |<I mI, S mS | F m>|^2 for all displayed ground states."""
    I = atom["I"]
    mI_list = m_values(I)
    mS_list = [-0.5, +0.5]
    probs = np.zeros((len(ground_states), len(mI_list), len(mS_list)), dtype=float)

    for ai, a in enumerate(ground_states):
        for ii, mI in enumerate(mI_list):
            for si, mS in enumerate(mS_list):
                c = cg_coeff_F_to_mI_mS(I, a["F"], a["m"], mI, mS)
                probs[ai, ii, si] = c * c

    return probs, mI_list, mS_list


def electron_marginal_from_population(atom, ground_states, p):
    """Electron spin marginal probabilities p(mS=-1/2), p(mS=+1/2)."""
    probs, _mI_list, _mS_list = hyperfine_uncoupled_probabilities(atom, ground_states)
    p = np.asarray(p, dtype=float)
    ps = np.einsum("a,ais->s", p, probs)
    total = ps.sum()
    if total > 0:
        ps = ps / total
    else:
        ps = np.array([0.5, 0.5], dtype=float)
    return ps


def build_spin_exchange_matrix(atom, ground_states, p_reference):
    """Build a population-only mean-field spin-exchange collision map.

    For a fixed ensemble electron-spin marginal rho_S[p_reference], one
    collision maps the source atom's nuclear marginal together with rho_S onto
    hyperfine populations. This gives a linear matrix M_SE[p_reference] used in
    a self-consistent fixed-point iteration.
    """
    probs, _mI_list, _mS_list = hyperfine_uncoupled_probabilities(atom, ground_states)
    p_reference = np.asarray(p_reference, dtype=float)
    p_reference = np.clip(p_reference, 0.0, None)
    if p_reference.sum() > 0:
        p_reference = p_reference / p_reference.sum()
    else:
        p_reference = np.ones(len(ground_states), dtype=float) / len(ground_states)

    electron_marginal = np.einsum("a,ais->s", p_reference, probs)
    if electron_marginal.sum() > 0:
        electron_marginal = electron_marginal / electron_marginal.sum()
    else:
        electron_marginal = np.array([0.5, 0.5], dtype=float)

    N = len(ground_states)
    M = np.zeros((N, N), dtype=float)

    target_given_nuclear_and_electron = probs
    for b in range(N):
        nuclear_given_b = probs[b, :, :].sum(axis=1)
        for a in range(N):
            M[a, b] = np.sum(
                nuclear_given_b[:, None]
                * electron_marginal[None, :]
                * target_given_nuclear_and_electron[a, :, :]
            )

    colsum = M.sum(axis=0)
    for j in range(N):
        if colsum[j] > 0:
            M[:, j] /= colsum[j]

    return M, electron_marginal


def spin_exchange_population_fractional_relaxation_rates(M_SE, p_steady, R_SE):
    """Return the signed SE fractional rate of each steady-state population.

    For the final self-consistent mean-field spin-exchange map,

        (d p_a / dt)_SE = R_SE [(M_SE p)_a - p_a].

    The reported rate is

        Gamma_a^(SE,net) = -(d p_a / dt)_SE / p_a.

    Positive values mean spin exchange removes population from the state;
    negative values mean spin exchange replenishes it. States with numerically
    zero population are blank.
    """
    p = np.asarray(p_steady, dtype=float)
    se_derivative = float(R_SE) * (np.asarray(M_SE, dtype=float) @ p - p)
    rates = np.full_like(p, np.nan, dtype=float)
    populated = p > 1e-15
    rates[populated] = -se_derivative[populated] / p[populated]
    return rates


def spin_exchange_adjacent_coherence_self_relaxation_rates(
    atom,
    ground_states,
    electron_marginal,
    R_SE,
):
    """Return SE self-decay rates for infinitesimal adjacent coherences.

    The app's fixed-reference mean-field spin-exchange channel is extended from
    populations to operators as

        E_SE(rho) = Tr_S(rho) tensor rho_S,

    where rho_S is the electron-spin marginal of the final steady state. For
    each adjacent coherence |a><b| in one F manifold, this function evaluates

        k_ab = <a| E_SE(|a><b|) |b>

    and reports R_SE (1-k_ab). The channel may also transfer amplitude among
    coherences with the same Delta m, so this is the local/self-decay
    coefficient of an infinitesimal perturbation, not an eigenmode decay rate.
    """
    amplitudes = coupled_basis_amplitudes(atom, ground_states)
    electron_marginal = np.asarray(electron_marginal, dtype=float)
    total = electron_marginal.sum()
    if total > 0:
        electron_marginal = electron_marginal / total
    else:
        electron_marginal = np.array([0.5, 0.5], dtype=float)

    rates = np.full(len(ground_states), np.nan, dtype=float)
    state_index = {
        (float(state["F"]), float(state["m"])): idx
        for idx, state in enumerate(ground_states)
    }

    for a_idx, a in enumerate(ground_states):
        key_b = (float(a["F"]), float(a["m"] - 1.0))
        b_idx = state_index.get(key_b)
        if b_idx is None:
            continue

        C_a = amplitudes[a_idx]
        C_b = amplitudes[b_idx]

        # Nuclear operator after tracing the source electron spin.
        rho_I = np.einsum("is,js->ij", C_a, C_b)

        # Project rho_I tensor rho_S back onto the same adjacent coherence.
        self_retention = np.einsum(
            "is,ij,s,js->",
            C_a,
            rho_I,
            electron_marginal,
            C_b,
        )
        rates[a_idx] = float(R_SE) * (1.0 - float(self_retention))

    return rates


def mirror_state_indices(ground_states):
    """Return indices implementing the transformation |F,m> -> |F,-m>."""
    state_index = {
        (float(state["F"]), float(state["m"])): idx
        for idx, state in enumerate(ground_states)
    }
    return np.array([
        state_index[(float(state["F"]), float(-state["m"]))]
        for state in ground_states
    ], dtype=int)


def symmetrize_populations_under_m_inversion(populations, ground_states):
    """Project a population vector onto the m -> -m symmetric subspace."""
    p = np.asarray(populations, dtype=float).copy()
    mirror = mirror_state_indices(ground_states)
    p = 0.5 * (p + p[mirror])
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total > 0:
        p /= total
    return p


def generator_has_m_inversion_symmetry(L, ground_states, rtol=1e-10, atol=1e-12):
    """Check whether the linear generator is invariant under m -> -m.

    For the mirror permutation Q, symmetry requires Q L Q = L. This is true
    for unpolarized relaxation and for optical pumping that does not distinguish
    +m from -m, such as pure pi light or equal sigma+ and sigma- components.
    """
    mirror = mirror_state_indices(ground_states)
    mirrored_L = np.asarray(L, dtype=float)[np.ix_(mirror, mirror)]
    scale = max(1.0, float(np.max(np.abs(L))))
    return bool(np.allclose(L, mirrored_L, rtol=rtol, atol=atol * scale))


def steady_state_with_spin_exchange(L_linear, atom, ground_states, R_SE, max_iter=200, tol=1e-12):
    """Solve the population steady state with nonlinear mean-field spin exchange.

    If the complete linear generator is invariant under m -> -m, the nonlinear
    fixed-point iteration is explicitly kept in that symmetric subspace. This
    prevents floating-point roundoff from selecting an arbitrary oriented branch
    when the physical conditions contain no handedness.
    """
    N = len(ground_states)
    enforce_mirror_symmetry = generator_has_m_inversion_symmetry(
        L_linear, ground_states
    )

    if R_SE <= 0:
        p = steady_state_from_L(L_linear)
        if enforce_mirror_symmetry:
            p = symmetrize_populations_under_m_inversion(p, ground_states)
        M_SE, electron_marginal = build_spin_exchange_matrix(atom, ground_states, p)
        return p, {
            "M_SE": M_SE,
            "L_effective": L_linear.copy(),
            "electron_marginal": electron_marginal,
            "iterations": 0,
            "converged": True,
            "residual": float(np.max(np.abs(L_linear @ p))),
            "mirror_symmetry_enforced": enforce_mirror_symmetry,
        }

    p = steady_state_from_L(L_linear)
    if enforce_mirror_symmetry:
        p = symmetrize_populations_under_m_inversion(p, ground_states)
    damping = 0.65
    converged = False
    residual = np.inf
    M_SE = np.eye(N)
    electron_marginal = np.array([0.5, 0.5], dtype=float)

    for iteration in range(1, max_iter + 1):
        M_SE, electron_marginal = build_spin_exchange_matrix(atom, ground_states, p)
        L_eff = L_linear + R_SE * (M_SE - np.eye(N))
        p_new = steady_state_from_L(L_eff)
        if enforce_mirror_symmetry:
            p_new = symmetrize_populations_under_m_inversion(
                p_new, ground_states
            )
        diff = float(np.max(np.abs(p_new - p)))
        p = damping * p_new + (1.0 - damping) * p
        p = np.clip(p, 0.0, None)
        if p.sum() > 0:
            p /= p.sum()
        if enforce_mirror_symmetry:
            p = symmetrize_populations_under_m_inversion(p, ground_states)

        M_SE, electron_marginal = build_spin_exchange_matrix(atom, ground_states, p)
        residual_vec = L_linear @ p + R_SE * (M_SE @ p - p)
        residual = float(np.max(np.abs(residual_vec)))

        if diff < tol and residual < max(1e-10, tol * max(1.0, R_SE)):
            converged = True
            break
    else:
        iteration = max_iter

    M_SE, electron_marginal = build_spin_exchange_matrix(atom, ground_states, p)
    L_eff = L_linear + R_SE * (M_SE - np.eye(N))
    residual_vec = L_linear @ p + R_SE * (M_SE @ p - p)
    residual = float(np.max(np.abs(residual_vec)))

    return p, {
        "M_SE": M_SE,
        "L_effective": L_eff,
        "electron_marginal": electron_marginal,
        "iterations": iteration,
        "converged": converged,
        "residual": residual,
        "mirror_symmetry_enforced": enforce_mirror_symmetry,
    }


# ============================================================
# 8. Steady-state solver
# ============================================================

def steady_state_from_L(L):
    """
    Solve L p = 0 with sum p = 1.
    """
    N = L.shape[0]
    A = L.copy()
    b = np.zeros(N)

    A[-1, :] = 1.0
    b[-1] = 1.0

    p, *_ = lstsq(A, b)
    p = np.real(p)

    # Remove tiny numerical negatives.
    p[p < 0] = 0.0

    if p.sum() > 0:
        p /= p.sum()
    else:
        p[:] = 1.0 / N

    return p


def expectation_m(ground_states, p):
    return float(sum(g["m"] * p[i] for i, g in enumerate(ground_states)))


def expectation_m2(ground_states, p):
    return float(sum((g["m"] ** 2) * p[i] for i, g in enumerate(ground_states)))


def population_by_F(df_pop):
    return df_pop.groupby("F", as_index=False)["population"].sum()


def add_population_difference_column(df_pop):
    """Add P_m - P_{m-1} within each ground hyperfine manifold F."""
    df = df_pop.copy()
    df["population_difference"] = np.nan

    for F_value, group in df.groupby("F", sort=False):
        group_sorted = group.sort_values("m")
        p_by_m = dict(zip(group_sorted["m"], group_sorted["population"]))
        for row_index, row in group_sorted.iterrows():
            m_value = row["m"]
            previous_m = m_value - 1.0
            if previous_m in p_by_m:
                df.loc[row_index, "population_difference"] = row["population"] - p_by_m[previous_m]

    return df


def add_light_shift_difference_column(df_pop):
    """Add nu_m - nu_{m-1} within each ground hyperfine manifold F.

    The light-shift frequency is in Hz. The lowest m state in each F manifold has
    no adjacent lower-m partner, so its difference is left blank.
    """
    df = df_pop.copy()
    df["light_shift_difference_Hz"] = np.nan

    if "light_shift_Hz" not in df.columns:
        return df

    for F_value, group in df.groupby("F", sort=False):
        group_sorted = group.sort_values("m")
        nu_by_m = dict(zip(group_sorted["m"], group_sorted["light_shift_Hz"]))
        for row_index, row in group_sorted.iterrows():
            m_value = row["m"]
            previous_m = m_value - 1.0
            if previous_m in nu_by_m:
                current_nu = row["light_shift_Hz"]
                previous_nu = nu_by_m[previous_m]
                if pd.notna(current_nu) and pd.notna(previous_nu):
                    df.loc[row_index, "light_shift_difference_Hz"] = current_nu - previous_nu

    return df


def total_pumping_rate_by_ground_state(ground_states, diagnostics):
    """Return R_m for every displayed |F,m> ground state.

    For each active beam, R_ge[ground, excited] is the state-resolved optical
    excitation rate after the beam intensity has been normalized to its selected
    reference-transition pumping rate. Summing over all excited states and all
    active beams gives the total optical depopulation rate R_m of each ground
    Zeeman sublevel.
    """
    rates = np.zeros(len(ground_states), dtype=float)
    for _beam, info in diagnostics:
        R_ge = np.asarray(info["R_ge"], dtype=float)
        if R_ge.shape[0] != len(ground_states):
            raise ValueError("Optical-pumping diagnostic has an incompatible ground-state dimension.")
        rates += np.sum(R_ge, axis=1)
    return rates


def optical_repopulation_fractional_rates(
    optical_generator, populations, pumping_rates_s, population_floor=1e-15
):
    """Return the optical repopulation rate A_m for each ground state.

    The optical generator can be decomposed as

        L_op = W_rep - diag(R),

    where R_m is the total excitation/depopulation rate from state m and
    W_rep contains spontaneous-emission repopulation into the ground states,
    including return to the same ground state. At the supplied population
    distribution p, the repopulation flow into state m is (W_rep p)_m.

    The table reports the corresponding fractional repopulation rate

        A_m = (W_rep p)_m / p_m,

    in s^-1, so that the signed net optical fractional population relaxation
    rate is R_m - A_m. States with negligible population are left blank.
    """
    L_op = np.asarray(optical_generator, dtype=float)
    p = np.asarray(populations, dtype=float)
    R = np.asarray(pumping_rates_s, dtype=float)

    if L_op.shape != (len(p), len(p)) or R.shape != p.shape:
        raise ValueError("Incompatible dimensions in optical repopulation calculation.")

    W_rep = L_op + np.diag(R)
    repopulation_flow = W_rep @ p
    # Remove only tiny negative roundoff; physical repopulation flow is nonnegative.
    repopulation_flow[np.abs(repopulation_flow) < 1e-14] = 0.0

    A = np.full_like(p, np.nan, dtype=float)
    mask = p > population_floor
    A[mask] = repopulation_flow[mask] / p[mask]
    return A


def add_adjacent_pumping_relaxation_columns(df_pop):
    """Add Gamma^R and Gamma^R/(2 pi) for adjacent Zeeman coherences.

    For the adjacent coherence rho_{m,m-1}, direct optical depopulation gives

        Gamma^R = (R_m + R_{m-1}) / 2

    in s^-1. The corresponding ordinary-frequency linewidth is
    Gamma^R/(2 pi) in Hz. The lowest-m state in each F manifold has no adjacent
    lower-m partner, so both entries are blank there.
    """
    df = df_pop.copy()
    rate_column = "adjacent_pumping_relaxation_s"
    hz_column = "adjacent_pumping_relaxation_Hz"
    df[rate_column] = np.nan
    df[hz_column] = np.nan

    if "pumping_rate_s" not in df.columns:
        return df

    for _F_value, group in df.groupby("F", sort=False):
        group_sorted = group.sort_values("m")
        rate_by_m = dict(zip(group_sorted["m"], group_sorted["pumping_rate_s"]))
        for row_index, row in group_sorted.iterrows():
            m_value = row["m"]
            previous_m = m_value - 1.0
            if previous_m in rate_by_m:
                gamma_R = 0.5 * (
                    row["pumping_rate_s"] + rate_by_m[previous_m]
                )
                df.loc[row_index, rate_column] = gamma_R
                df.loc[row_index, hz_column] = gamma_R / (2.0 * np.pi)

    return df


def apply_two_line_column_headers(df, header_map):
    """Use real pandas MultiIndex headers so Streamlit shows units on a second header row."""
    display_df = df.copy()
    display_df.columns = pd.MultiIndex.from_tuples([
        header_map.get(col, (str(col), "")) for col in display_df.columns
    ])
    return display_df


def render_transition_table_html(df):
    """Render transition table with controlled two-line headers and HTML subscripts."""
    import html

    columns = [
        ("Line", "Line", None, "text"),
        ("Fg", "F<sub>g</sub>", None, "text"),
        ("F'", "F′", None, "text"),
        ("nu_D_absolute", "ν<sub>D</sub> absolute", "MHz", "1f"),
        ("detuning_zero_pressure", "ν<sub>FF′</sub> − ν<sub>D</sub>, P=0", "MHz", "1f"),
        ("N2_shift", "N<sub>2</sub> shift βP", "MHz", "1f"),
        ("detuning_with_N2", "ν<sub>FF′</sub> − ν<sub>D</sub>, with N<sub>2</sub>", "MHz", "1f"),
        ("transition_frequency_with_N2", "ν<sub>FF′</sub>, with N<sub>2</sub>", "MHz", "1f"),
        ("pump_1_frequency", "ν<sub>pump1</sub>", "MHz", "1f"),
        ("pump_2_frequency", "ν<sub>pump2</sub>", "MHz", "1f"),
        ("pump_3_frequency", "ν<sub>pump3</sub>", "MHz", "1f"),
        ("lorentz_FWHM_total", "Lorentz FWHM total", "MHz", "1f"),
        ("doppler_FWHM", "Doppler FWHM", "MHz", "1f"),
    ]

    def fmt_value(value, kind):
        if pd.isna(value):
            return ""
        if kind == "text":
            return html.escape(str(value))
        if kind == "9f":
            return f"{float(value):.9f}"
        if kind == "1f":
            return f"{float(value):.1f}"
        return html.escape(str(value))

    header_cells = []
    for _col, title, unit, _kind in columns:
        if unit:
            header_cells.append(
                f"<th><div class='quantity'>{title}</div><div class='unit'>({html.escape(unit)})</div></th>"
            )
        else:
            header_cells.append(f"<th><div class='quantity'>{title}</div></th>")

    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col, _title, _unit, kind in columns:
            cells.append(f"<td>{fmt_value(row[col], kind)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
<style>
.transition-table-wrap {{
    max-height: 420px;
    overflow: auto;
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.35rem;
}}
.transition-table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.86rem;
}}
.transition-table th {{
    position: sticky;
    top: 0;
    background: rgb(250, 250, 250);
    z-index: 1;
    border-bottom: 1px solid rgba(49, 51, 63, 0.25);
    padding: 0.35rem 0.45rem;
    text-align: left;
    white-space: nowrap;
}}
.transition-table td {{
    border-bottom: 1px solid rgba(49, 51, 63, 0.12);
    padding: 0.30rem 0.45rem;
    white-space: nowrap;
    text-align: right;
}}
.transition-table td:first-child,
.transition-table td:nth-child(2),
.transition-table td:nth-child(3) {{
    text-align: left;
}}
.transition-table .quantity {{
    line-height: 1.15;
}}
.transition-table .unit {{
    line-height: 1.15;
    font-size: 0.78rem;
    font-weight: 400;
    color: rgba(49, 51, 63, 0.72);
}}
</style>
<div class='transition-table-wrap'>
<table class='transition-table'>
<thead><tr>{''.join(header_cells)}</tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table>
</div>
"""


def render_zeeman_properties_table_html(df):
    """Render the Zeeman-sublevel table with a guaranteed hyperfine separator.

    A custom HTML table is used because Streamlit's dataframe renderer does not
    reliably preserve pandas Styler border rules. The first row of every new F
    manifold receives a 2 px top border, making the boundary between
    F=I-1/2 and F=I+1/2 clearly visible. Column widths are determined
    automatically from the rendered header and cell contents.
    """
    import html

    columns = [
        ("F", "F", None, "g"),
        ("m", "m", None, "g"),
        ("P_F", "P<sub>F</sub>", None, ".3f"),
        ("Pₘ", "P<sub>m</sub>", None, ".3f"),
        ("Dₘ", "D<sub>m</sub>", None, ".3f"),
        ("νLS (Hz)", "ν<sub>LS</sub>", "Hz", ".1f"),
        ("Δν (Hz)", "Δν", "Hz", ".1f"),
        ("Γ^{ER}_{m} (s^-1)", "Γ<sup>ER</sup><sub>m</sub>", "s<sup>−1</sup>", ".2f"),
        ("Γ^{SE}_{m} (s^-1)", "Γ<sup>SE</sup><sub>m</sub>", "s<sup>−1</sup>", ".2f"),
        ("Aₘ (s⁻¹)", "A<sub>m</sub>", "s<sup>−1</sup>", ".1f"),
        ("Rₘ (s⁻¹)", "R<sub>m</sub>", "s<sup>−1</sup>", ".1f"),
        ("Γ^R (s^-1)", "Γ<sup>R</sup>", "s<sup>−1</sup>", ".1f"),
        ("Γ^R/2π (Hz)", "Γ<sup>R</sup>/2π", "Hz", ".1f"),
        ("Γ^{ER}_{m,m-1} (s^-1)", "Γ<sup>ER</sup><sub>m,m−1</sub>", "s<sup>−1</sup>", ".2f"),
        ("Γ^{SE}_{m,m-1} (s^-1)", "Γ<sup>SE</sup><sub>m,m−1</sub>", "s<sup>−1</sup>", ".2f"),
    ]

    def fmt(value, spec):
        if pd.isna(value):
            return ""
        if spec == "g":
            return f"{float(value):g}"
        return format(float(value), spec)

    header_cells = []
    for _key, title, unit, _spec in columns:
        unit_html = f"<div class='unit'>({unit})</div>" if unit else ""
        header_cells.append(
            f"<th><div class='quantity'>{title}</div>{unit_html}</th>"
        )
    headers = "".join(header_cells)

    body_rows = []
    previous_F = None
    for _, row in df.iterrows():
        current_F = float(row["F"])
        separator = previous_F is not None and not np.isclose(current_F, previous_F)
        row_class = " class='hyperfine-separator'" if separator else ""
        cells = "".join(
            f"<td>{html.escape(fmt(row[key], spec))}</td>"
            for key, _title, _unit, spec in columns
        )
        body_rows.append(f"<tr{row_class}>{cells}</tr>")
        previous_F = current_F

    return f"""
<style>
.zeeman-properties-wrap {{
    max-height: 315px;
    overflow: auto;
    border: 1px solid rgba(49, 51, 63, 0.20);
    border-radius: 0.35rem;
}}
.zeeman-properties-table {{
    border-collapse: collapse;
    width: max-content;
    min-width: 0;
    table-layout: auto;
    font-size: 0.86rem;
}}
.zeeman-properties-table th {{
    position: sticky;
    top: 0;
    z-index: 2;
    background: rgb(250, 250, 250);
    border-bottom: 1px solid rgba(49, 51, 63, 0.30);
    padding: 0.35rem 0.45rem;
    text-align: right;
    white-space: nowrap;
}}
.zeeman-properties-table th .quantity {{
    line-height: 1.15;
}}
.zeeman-properties-table th .unit {{
    line-height: 1.15;
    font-size: 0.78rem;
    font-weight: 400;
    color: rgba(49, 51, 63, 0.72);
}}
.zeeman-properties-table td {{
    border-bottom: 1px solid rgba(49, 51, 63, 0.12);
    padding: 0.30rem 0.45rem;
    text-align: right;
    white-space: nowrap;
}}
.zeeman-properties-table tr.hyperfine-separator td {{
    border-top: 2px solid rgba(49, 51, 63, 0.90) !important;
}}
</style>
<div class="zeeman-properties-wrap">
<table class="zeeman-properties-table">
<thead><tr>{headers}</tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table>
</div>
"""



APP_BASE_TITLE = "alkali pumping"


def current_browser_title():
    """Return the browser title from the live condition name field."""
    raw_name = st.session_state.get("condition_name", "")
    if not str(raw_name or "").strip():
        return APP_BASE_TITLE
    name = clean_condition_name(raw_name)
    return f"{APP_BASE_TITLE}: {name}"


# ============================================================
# 9. Streamlit UI
# ============================================================

# Initialize every condition field before page configuration and widget creation.
# Later JSON loads still overwrite these values through apply_loaded_condition_dict().
for _key, _value in DEFAULT_STARTUP_CONDITION.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value
st.session_state.setdefault(
    "_last_atom_name_for_defaults",
    DEFAULT_STARTUP_CONDITION["atom_name"],
)

st.set_page_config(
    page_title=current_browser_title(),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Alkali pumping: three pumps + ER + SE")

# ============================================================
# Sidebar: all input condition values
# ============================================================

with st.sidebar:
    # Defaults used before widgets are rendered.
    # Put defaults in st.session_state, then create widgets using only key=...,
    # so Streamlit does not warn about competing default and session-state values.
    if "show_allowed_only" not in st.session_state:
        st.session_state["show_allowed_only"] = True
    if "show_rate_matrices" not in st.session_state:
        st.session_state["show_rate_matrices"] = False
    if "n2_pressure_torr" not in st.session_state:
        st.session_state["n2_pressure_torr"] = 0.0
    if "temperature_C_for_table" not in st.session_state:
        st.session_state["temperature_C_for_table"] = 25.0
    if "gamma_ER" not in st.session_state:
        st.session_state["gamma_ER"] = 1.0
    if "include_spin_exchange" not in st.session_state:
        st.session_state["include_spin_exchange"] = True

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section {
            padding: 0.15rem 0.25rem;
            min-height: 2.2rem;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] small {
            display: none;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
            padding: 0.20rem 0.25rem;
            min-height: 2.2rem;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button,
        section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] button {
            height: 2.25rem;
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("condition")
    if "condition_name" not in st.session_state:
        st.session_state["condition_name"] = "default"

    # Keep the controls visually at the top, but populate this placeholder only
    # after every sidebar widget has been instantiated. This ensures that the
    # downloaded JSON contains the complete, current condition.
    condition_controls_placeholder = st.empty()

    st.header("Atom / cell")

    atom_row_col1, atom_row_col2, atom_row_col3 = st.columns(3, gap="small")
    with atom_row_col1:
        atom_name = st.selectbox("Alkali atom", list(ATOMS.keys()), index=0, key="atom_name")
    with atom_row_col2:
        n2_pressure_torr = st.number_input(
            "N₂ pressure (Torr)",
            min_value=0.0,
            step=10.0,
            format="%.1f",
            key="n2_pressure_torr",
        )
    with atom_row_col3:
        temperature_C = st.number_input(
            "Temperature (°C)",
            step=1.0,
            format="%.1f",
            key="temperature_C_for_table",
        )

    atom = ATOMS[atom_name]

    # If the atom is changed manually, initialize the N2 coefficients to that atom's defaults.
    # Loading from a JSON condition overwrites these keys before widgets are created.
    if st.session_state.get("_last_atom_name_for_defaults") != atom_name:
        st.session_state["D1_width"] = DEFAULT_N2_COEFFS[atom_name]["D1"]["width"]
        st.session_state["D2_width"] = DEFAULT_N2_COEFFS[atom_name]["D2"]["width"]
        st.session_state["D1_shift"] = DEFAULT_N2_COEFFS[atom_name]["D1"]["shift"]
        st.session_state["D2_shift"] = DEFAULT_N2_COEFFS[atom_name]["D2"]["shift"]
        st.session_state["_last_atom_name_for_defaults"] = atom_name

    cell_row_col1, cell_row_col2, cell_row_col3 = st.columns(3, gap="small")
    with cell_row_col1:
        include_spin_exchange = st.checkbox(
            "Include spin exchange",
            key="include_spin_exchange",
        )
    with cell_row_col2:
        gamma_ER = st.number_input(
            "ER rate (s⁻¹)",
            min_value=0.0,
            step=1.0,
            format="%.1f",
            key="gamma_ER",
        )
    with cell_row_col3:
        q_axis = st.selectbox(
            "Quantization axis",
            ["z", "x", "y"],
            index=0,
            key="q_axis",
        )

    se_rate_preview = spin_exchange_rate_info(atom_name, atom, temperature_C)
    st.caption(
        f"R_SE={se_rate_preview['rate_s']:.3g}s⁻¹ "
        f"for n={se_rate_preview['density_cm3']:.2g}cm⁻³ and "
        f"σ_SE={se_rate_preview['sigma_cm2']:.2g}cm²."
    )

    with st.expander("N₂ coefficients", expanded=False):
        c1, c2 = st.columns(2, gap="small")
        with c1:
            D1_width = st.number_input(
                "D1 broadening",
                step=0.1,
                key="D1_width",
                help="N2 pressure-broadening coefficient, MHz/Torr",
            )
            D2_width = st.number_input(
                "D2 broadening",
                step=0.1,
                key="D2_width",
                help="N2 pressure-broadening coefficient, MHz/Torr",
            )

        with c2:
            D1_shift = st.number_input(
                "D1 shift",
                step=0.1,
                key="D1_shift",
                help="N2 pressure shift coefficient, MHz/Torr",
            )
            D2_shift = st.number_input(
                "D2 shift",
                step=0.1,
                key="D2_shift",
                help="N2 pressure shift coefficient, MHz/Torr",
            )

        st.caption("Broadening and shifts are in MHz/Torr. Negative shift = red shift.")

    n2_coeffs = {
        "D1": {"width": D1_width, "shift": D1_shift},
        "D2": {"width": D2_width, "shift": D2_shift},
    }

    show_allowed_only = st.session_state["show_allowed_only"]
    show_rate_matrices = st.session_state["show_rate_matrices"]

    def beam_config_ui(beam_number, default_line_index=0, default_Fg=None, default_Fe=None, default_rate=10.0):
        st.header(f"Beam {beam_number}")
        det_rel_key = f"det_rel{beam_number}"
        rate_key = f"rate{beam_number}"
        if det_rel_key not in st.session_state:
            st.session_state[det_rel_key] = 0.0
        if rate_key not in st.session_state:
            st.session_state[rate_key] = float(default_rate)

        line = st.selectbox("Reference Line", ["D1", "D2"], index=default_line_index, key=f"line{beam_number}")

        transition_options = transition_choice_labels(
            atom, line, n2_pressure_torr, n2_coeffs, allowed_only=show_allowed_only
        )
        transition_key = f"transition{beam_number}"
        if transition_options and st.session_state.get(transition_key) not in transition_options:
            preferred_transition = None
            if default_Fg is not None and default_Fe is not None:
                preferred_transition = default_transition_label(
                    atom, line, n2_pressure_torr, n2_coeffs, default_Fg, default_Fe, allowed_only=show_allowed_only
                )
            st.session_state[transition_key] = preferred_transition if preferred_transition in transition_options else transition_options[0]
        transition = st.selectbox(
            "hpf-transition",
            transition_options,
            key=transition_key,
        )
        det_rel = st.number_input(
            "Detuning (MHz)",
            step=10.0,
            format="%g",
            key=det_rel_key,
            help="Laser detuning is defined relative to this pressure-shifted selected hyperfine transition.",
        )
        rate = st.number_input(
            "Pump rate (s⁻¹)",
            min_value=0.0,
            step=10.0,
            format="%.0f",
            key=rate_key,
            help="Average depopulation rate for the selected transition of an unpolarized atom.",
        )
        k_axis = st.selectbox(
            "beam direction",
            ["z", "x", "y"],
            index=0,
            key=f"k{beam_number}",
        )
        pol_options = allowed_polarizations(k_axis)
        pol_key = f"pol{beam_number}"
        if st.session_state.get(pol_key) not in pol_options:
            st.session_state[pol_key] = pol_options[0]
        pol = st.selectbox(
            "Polarization",
            pol_options,
            key=pol_key,
        )
        return line, transition, det_rel, rate, k_axis, pol

    bcol1, bcol2, bcol3 = st.columns(3, gap="small")
    with bcol1:
        line1, transition1, det_rel1, rate1, k1, pol1 = beam_config_ui(1, default_Fg=1, default_Fe=2)
    with bcol2:
        line2, transition2, det_rel2, rate2, k2, pol2 = beam_config_ui(2, default_Fg=2, default_Fe=2)
    with bcol3:
        line3, transition3, det_rel3, rate3, k3, pol3 = beam_config_ui(3, default_Fg=2, default_Fe=2, default_rate=0.0)

    st.divider()
    st.header("Display")
    show_allowed_only = st.checkbox(
        "Only show allowed hyperfine transitions",
        key="show_allowed_only",
    )
    show_rate_matrices = st.checkbox(
        "Show rate matrices",
        key="show_rate_matrices",
    )

    # Populate the top placeholder after all condition widgets exist.
    # The condition name widget is evaluated before the download payload is
    # serialized, so the JSON always contains the value currently visible here.
    with condition_controls_placeholder.container():
        load_col, save_col = st.columns([0.38,0.62], gap="small")

        with load_col:
            st.file_uploader(
                "Load condition",
                type=["json"],
                key="condition_file_uploader",
                help="Choose an alkali_pumping v2.23 JSON condition file.",
                label_visibility="collapsed",
                on_change=load_condition_callback,
            )

        with save_col:
            save_button_placeholder = st.empty()
            condition_name = st.text_input(
                "condition name",
                key="condition_name",
                help=(
                    "This value is saved inside the JSON conditions, used as the "
                    "suggested filename, and appended to the browser title after "
                    "the condition is loaded or saved."
                ),
            )

        condition_save_name = clean_condition_name(condition_name)
        condition_values = current_condition_values(
            condition_name=condition_save_name
        )
        condition_payload = build_condition_payload(condition_values)
        condition_json = json.dumps(condition_payload, indent=2)

        save_button_placeholder.download_button(
            "Save condition",
            data=condition_json,
            file_name=f"{condition_save_name}.json",
            mime="application/json",
            key="save_condition_button",
            width="stretch",
        )

        if st.session_state.get("_condition_load_message"):
            st.success(st.session_state.pop("_condition_load_message"))
        if st.session_state.get("_condition_load_error"):
            st.error(
                "Could not load condition file: "
                + st.session_state.pop("_condition_load_error")
            )


# ============================================================
# Build model
# ============================================================

det1_abs, selected_transition1 = absolute_detuning_from_transition_choice(
    atom=atom,
    line=line1,
    transition_label=transition1,
    relative_detuning_MHz=det_rel1,
    n2_pressure_torr=n2_pressure_torr,
    n2_coeffs=n2_coeffs,
    allowed_only=show_allowed_only,
)

det2_abs, selected_transition2 = absolute_detuning_from_transition_choice(
    atom=atom,
    line=line2,
    transition_label=transition2,
    relative_detuning_MHz=det_rel2,
    n2_pressure_torr=n2_pressure_torr,
    n2_coeffs=n2_coeffs,
    allowed_only=show_allowed_only,
)

det3_abs, selected_transition3 = absolute_detuning_from_transition_choice(
    atom=atom,
    line=line3,
    transition_label=transition3,
    relative_detuning_MHz=det_rel3,
    n2_pressure_torr=n2_pressure_torr,
    n2_coeffs=n2_coeffs,
    allowed_only=show_allowed_only,
)

beam_inputs = [
    {
        "name": "Beam 1",
        "line": line1,
        "transition_label": transition1,
        "selected_transition": selected_transition1,
        "detuning_relative": det_rel1,
        "detuning": det1_abs,
        "rate": rate1,
        "k_axis": k1,
        "pol": pol1,
        "q_axis": q_axis,
    },
    {
        "name": "Beam 2",
        "line": line2,
        "transition_label": transition2,
        "selected_transition": selected_transition2,
        "detuning_relative": det_rel2,
        "detuning": det2_abs,
        "rate": rate2,
        "k_axis": k2,
        "pol": pol2,
        "q_axis": q_axis,
    },
    {
        "name": "Beam 3",
        "line": line3,
        "transition_label": transition3,
        "selected_transition": selected_transition3,
        "detuning_relative": det_rel3,
        "detuning": det3_abs,
        "rate": rate3,
        "k_axis": k3,
        "pol": pol3,
        "q_axis": q_axis,
    },
]

ground_states = build_ground_states(atom)
N = len(ground_states)

L_total = np.zeros((N, N), dtype=float)
diagnostics = []

for b in beam_inputs:
    if b["rate"] > 0:
        line = b["line"]

        Lb, info = build_optical_L(
            atom=atom,
            line=line,
            ground_states=ground_states,
            detuning_MHz=b["detuning"],
            pump_rate_s=b["rate"],
            selected_transition=b.get("selected_transition"),
            k_axis=b["k_axis"],
            pol=b["pol"],
            q_axis=q_axis,
            n2_pressure_torr=n2_pressure_torr,
            temperature_C=temperature_C,
            n2_width_MHz_per_torr=n2_coeffs[line]["width"],
            n2_shift_MHz_per_torr=n2_coeffs[line]["shift"],
            normalize_to_unpolarized=True,
        )

        L_total += Lb
        diagnostics.append((b, info))

# Preserve the optical-only generator before ER and SE terms are added.
L_optical = L_total.copy()

M_ER = build_ER_matrix(atom, ground_states)
L_linear = L_optical + gamma_ER * (M_ER - np.eye(N))

se_rate_info = spin_exchange_rate_info(atom_name, atom, temperature_C)
R_SE_inferred = se_rate_info["rate_s"]
R_SE = R_SE_inferred if include_spin_exchange else 0.0

if R_SE > 0:
    p_ss, se_solver_info = steady_state_with_spin_exchange(
        L_linear, atom, ground_states, R_SE
    )
    L_total = se_solver_info["L_effective"]
else:
    p_ss = steady_state_from_L(L_linear)
    M_SE, electron_marginal = build_spin_exchange_matrix(atom, ground_states, p_ss)
    se_solver_info = {
        "M_SE": M_SE,
        "L_effective": L_linear.copy(),
        "electron_marginal": electron_marginal,
        "iterations": 0,
        "converged": True,
        "residual": float(np.max(np.abs(L_linear @ p_ss))),
        "mirror_symmetry_enforced": False,
    }
    L_total = L_linear

light_shift_Hz, light_shift_available = total_light_shift_Hz_from_diagnostics(
    ground_states,
    beam_inputs,
    diagnostics,
)
pumping_rate_by_state_s = total_pumping_rate_by_ground_state(
    ground_states,
    diagnostics,
)
optical_repopulation_rate_by_state_s = optical_repopulation_fractional_rates(
    L_optical,
    p_ss,
    pumping_rate_by_state_s,
)

er_population_relaxation_s = er_population_fractional_relaxation_rates(
    M_ER, p_ss, gamma_ER
)
er_adjacent_coherence_relaxation_s = er_adjacent_coherence_self_relaxation_rates(
    atom, ground_states, gamma_ER
)

se_population_relaxation_s = spin_exchange_population_fractional_relaxation_rates(
    se_solver_info["M_SE"], p_ss, R_SE
)
se_adjacent_coherence_relaxation_s = (
    spin_exchange_adjacent_coherence_self_relaxation_rates(
        atom,
        ground_states,
        se_solver_info["electron_marginal"],
        R_SE,
    )
)

df_pop = pd.DataFrame({
    "F": [g["F"] for g in ground_states],
    "m": [g["m"] for g in ground_states],
    "population": p_ss,
    "light_shift_Hz": light_shift_Hz,
    "optical_repopulation_rate_s": optical_repopulation_rate_by_state_s,
    "pumping_rate_s": pumping_rate_by_state_s,
    "er_population_relaxation_s": er_population_relaxation_s,
    "er_adjacent_coherence_relaxation_s": er_adjacent_coherence_relaxation_s,
    "se_population_relaxation_s": se_population_relaxation_s,
    "se_adjacent_coherence_relaxation_s": se_adjacent_coherence_relaxation_s,
})
df_pop = add_population_difference_column(df_pop)
df_pop = add_light_shift_difference_column(df_pop)
df_pop = add_adjacent_pumping_relaxation_columns(df_pop)

df_F = population_by_F(df_pop)

# Show the total population P_F of each hyperfine manifold only on its m=0 row
# in the Zeeman sublevel properties table.
df_pop["hyperfine_population"] = np.nan
for _, f_row in df_F.iterrows():
    manifold_mask = (
        np.isclose(df_pop["F"].to_numpy(dtype=float), float(f_row["F"]))
        & np.isclose(df_pop["m"].to_numpy(dtype=float), 0.0)
    )
    df_pop.loc[manifold_mask, "hyperfine_population"] = float(f_row["population"])

df_trans = hyperfine_transition_table(
    atom=atom,
    n2_pressure_torr=n2_pressure_torr,
    n2_coeffs=n2_coeffs,
    allowed_only=show_allowed_only,
    pump_beams=beam_inputs,
)

labels = [g["label"] for g in ground_states]

df_pop_display = df_pop.rename(columns={
    "hyperfine_population": "P_F",
    "population": "Pₘ",
    "population_difference": "Dₘ",
    "light_shift_Hz": "νLS (Hz)",
    "light_shift_difference_Hz": "Δν (Hz)",
    "optical_repopulation_rate_s": "Aₘ (s⁻¹)",
    "pumping_rate_s": "Rₘ (s⁻¹)",
    "adjacent_pumping_relaxation_s": "Γ^R (s^-1)",
    "adjacent_pumping_relaxation_Hz": "Γ^R/2π (Hz)",
    "er_population_relaxation_s": "Γ^{ER}_{m} (s^-1)",
    "er_adjacent_coherence_relaxation_s": "Γ^{ER}_{m,m-1} (s^-1)",
    "se_population_relaxation_s": "Γ^{SE}_{m} (s^-1)",
    "se_adjacent_coherence_relaxation_s": "Γ^{SE}_{m,m-1} (s^-1)",
})
# Place the ER and SE population rates immediately before A_m, then R_m and the adjacent-coherence columns.
df_pop_display = df_pop_display[[
    "F",
    "m",
    "P_F",
    "Pₘ",
    "Dₘ",
    "νLS (Hz)",
    "Δν (Hz)",
    "Γ^{ER}_{m} (s^-1)",
    "Γ^{SE}_{m} (s^-1)",
    "Aₘ (s⁻¹)",
    "Rₘ (s⁻¹)",
    "Γ^R (s^-1)",
    "Γ^R/2π (Hz)",
    "Γ^{ER}_{m,m-1} (s^-1)",
    "Γ^{SE}_{m,m-1} (s^-1)",
]]
# Display the upper-F manifold first and order m from +F to -F
# within each manifold.
df_pop_display = df_pop_display.sort_values(
    by=["F", "m"],
    ascending=[False, False],
    kind="stable",
).reset_index(drop=True)


# ============================================================
# Main compact layout
# ============================================================

def compact_section_title(text):
    """Render a compact title at about half the size of st.header."""
    st.markdown(
        f"<div style='font-size:1.25rem; font-weight:600; line-height:1.25; "
        f"margin:0.25rem 0 0.45rem 0;'>{text}</div>",
        unsafe_allow_html=True,
    )

left, right = st.columns([0.62, 1.63], gap="small")

with left:
    compact_section_title(f"{atom_name} ground-state populations")

    # Separate the two ground-state hyperfine manifolds.  The manifold with
    # the larger hyperfine energy is displayed in the upper panel.
    manifold_energies = {}
    for state in ground_states:
        manifold_energies.setdefault(float(state["F"]), float(state["E"]))

    manifolds_by_energy = sorted(
        manifold_energies, key=lambda F_value: manifold_energies[F_value]
    )
    lower_F = manifolds_by_energy[0]
    upper_F = manifolds_by_energy[-1]

    fig, (ax_upper, ax_lower) = plt.subplots(
        2, 1, figsize=(4.6, 4.5), sharey=True
    )

    for ax, F_value, panel_name in [
        (ax_upper, upper_F, "Upper hyperfine level"),
        (ax_lower, lower_F, "Lower hyperfine level"),
    ]:
        indices = [
            index
            for index, state in enumerate(ground_states)
            if np.isclose(float(state["F"]), F_value)
        ]
        m_labels = [f"{ground_states[index]['m']:g}" for index in indices]
        populations = [p_ss[index] for index in indices]

        ax.bar(m_labels, populations)
        ax.set_ylabel("Population")
        ax.set_title(f"{panel_name}: F={F_value:g}", fontsize=10, pad=3)

    population_axis_max = max(0.01, 1.08 * float(np.max(p_ss)))
    ax_upper.set_ylim(0.0, population_axis_max)
    ax_lower.set_ylim(0.0, population_axis_max)
    ax_lower.set_xlabel(rf"$m$ along {q_axis}")

    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    summary_sum_p = p_ss.sum()
    summary_m = expectation_m(ground_states, p_ss)
    summary_m2 = expectation_m2(ground_states, p_ss)

    st.markdown(
        f"""
        <style>
        .population-summary-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.9rem;
            margin-top: 0rem;
            font-size: 0.9rem;
            line-height: 1.35;
        }}
        .population-summary-item {{
            display: inline-flex;
            align-items: baseline;
            gap: 0.25rem;
            white-space: nowrap;
        }}
        .population-summary-label {{
            font-weight: 600;
        }}
        .population-summary-value {{
            font-family: inherit;
            font-size: inherit;
            font-weight: 400;
            font-variant-numeric: tabular-nums;
        }}
        </style>
        <div class="population-summary-row">
            <div class="population-summary-item">
                <span class="population-summary-label">Σp =</span>
                <span class="population-summary-value">{summary_sum_p:.4f}</span>
            </div>
            <div class="population-summary-item">
                <span class="population-summary-label">〈m〉 =</span>
                <span class="population-summary-value">{summary_m:.4f}</span>
            </div>
            <div class="population-summary-item">
                <span class="population-summary-label">〈m²〉 =</span>
                <span class="population-summary-value">{summary_m2:.4f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    title,tip = st.columns([0.9,0.1])
    with title:
        compact_section_title("Zeeman sublevel properties")     
    with tip:
        with st.popover("❓"):
            st.markdown(
                "Dₘ = Pₘ - Pₘ₋₁ is the population difference between adjacent Zeeman sublevels of the same F.  \n"
                "Δν = νLSₘ - νLSₘ₋₁ is the adjacent-sublevel light-shift difference.  \n"
                "Γ^{ER}_{m} is the signed net fractional ER rate of population; positive means loss.  \n" 
                "Γ^{SE}_{m} is the signed net fractional SE rate of population at the steady state  \n"
                "Aₘ is the repopulation rate into |F,m⟩ divided by its steady-state population: Aₘ = [Σₙ Wₘ←ₙ Pₙ]/Pₘ.  \n"
                "Rₘ is the depopulation rate from |F,m⟩, summed over excited states and all active pump beams.  \n"
                "Γ^R = (Rₘ + Rₘ₋₁)/2 is the pump-induced adjacent-coherence decay rate, and Γ^R/2π is the corresponding broadening.  \n"
                "Γ^{ER}_{m,m-1} is the local adjacent-coherence self-decay rate due to ER.  \n"
                "Γ^{SE}_{m,m-1} is the adjacent-coherence self-decay rate under the steady-state mean-field SE."
            )
    st.markdown(
        render_zeeman_properties_table_html(df_pop_display),
        unsafe_allow_html=True,
    )
    if light_shift_available:
        st.caption("νLS is shown because all active pump-beam light-shift Hamiltonians commute with the selected quantization-axis spin component.")
    else:
        st.caption("νLS is blank because at least one active beam has multiple spherical polarization components relative to the quantization axis, so the light-shift Hamiltonian may not commute with the selected spin component.")



# compact_section_title("D1/D2 hyperfine transition detunings")
# st.caption(
#     "Detunings are relative to the corresponding zero-pressure D1 or D2 fine-structure line center. "
#     "Absolute optical frequencies are shown in MHz."
# )
with st.expander("D1/D2 hyperfine transition detunings", expanded=False):
    st.caption(
        "Detunings are relative to the corresponding zero-pressure D1 or D2 fine-structure line center. "
        "Absolute optical frequencies are shown in MHz."
    )
    st.markdown(
        render_transition_table_html(df_trans),
        unsafe_allow_html=True,
    )

if show_rate_matrices:
    with st.expander("Total rate matrix L", expanded=False):
        st.write("Columns are source states; rows are destination states. dp/dt = L p.")
        Ldf = pd.DataFrame(L_total, index=labels, columns=labels)
        st.dataframe(Ldf.style.format("{:.3e}"), width="stretch")

    with st.expander("ER redistribution matrix M_ER", expanded=False):
        st.write("After one ER collision: p → M_ER p.")
        Mdf = pd.DataFrame(M_ER, index=labels, columns=labels)
        st.dataframe(Mdf.style.format("{:.4f}"), width="stretch")

with st.expander("Spin-exchange diagnostics", expanded=False):
    st.write(
        f"Include spin exchange: {include_spin_exchange}.  "
        f"Applied R_SE = {R_SE:.6g} s⁻¹ "
        f"(temperature-inferred {R_SE_inferred:.6g} s⁻¹)."
    )
    st.write(
        f"Alkali vapor pressure = {se_rate_info['pressure_torr']:.3e} Torr; "
        f"density = {se_rate_info['density_cm3']:.3e} cm⁻³; "
        f"mean relative speed = {se_rate_info['vrel_cm_s']:.3e} cm/s; "
        f"σ_SE = {se_rate_info['sigma_cm2']:.3e} cm²."
    )
    ps_minus, ps_plus = se_solver_info["electron_marginal"]
    st.write(
        f"Electron-spin marginal used by the mean-field SE map: "
        f"P(mS=-1/2) = {ps_minus:.6f}, P(mS=+1/2) = {ps_plus:.6f}."
    )
    st.write(
        f"SE fixed-point iterations = {se_solver_info['iterations']}; "
        f"converged = {se_solver_info['converged']}; "
        f"max residual = {se_solver_info['residual']:.3e} s⁻¹."
    )
    st.write(
        "m→−m symmetry enforced in SE solve: "
        f"{se_solver_info.get('mirror_symmetry_enforced', False)}."
    )
    st.caption(
        "Spin exchange is included as a population-only mean-field collision map. "
        "It preserves the source atom's nuclear-spin marginal and replaces the "
        "electron spin by the ensemble electron-spin marginal, then projects back "
        "onto the displayed hyperfine populations. Coherences are not propagated."
    )

with st.expander("Light-shift calculation", expanded=False):
    st.write(
        "The light-shift column is calculated only when every active pump beam has "
        "a single spherical polarization component q=-1, 0, or +1 relative to the "
        "selected quantization axis. In that case the AC-Stark Hamiltonian is "
        "diagonal in the displayed |F,m> basis and commutes with the spin component "
        "along the quantization axis."
    )
    st.latex(
        r"""
        R_{F,m\rightarrow F',m'}
        \propto
        \operatorname{Re}\,w(z),
        \qquad
        \delta\omega_{F,m\rightarrow F',m'}
        \propto
        \frac{1}{2}\operatorname{Im}\,w(z)
        """
    )
    st.latex(
        r"""
        z=
        \frac{\Delta_{F,m\rightarrow F',m'}+i\Gamma_L/2}
        {\sigma_D\sqrt{2}}
        """
    )
    st.write(
        "The table reports νLS = δω/(2π) in Hz and νₘ − νₘ₋₁ in Hz. The sum over excited states gives "
        "the total diagonal AC-Stark shift, including scalar, vector, and tensor "
        "contributions. In the zero-Doppler or far-wing limit this reduces to "
        "δω = R Δ/Γ_L. This small shift is not fed back into the optical detunings."
    )


# ============================================================
# Notes shown at bottom
# ============================================================



with st.expander("Model and sign convention"):
    st.write("The solved equation is")

    st.latex(
        r"""
        \frac{d\mathbf p}{dt}
        =
        \left[
        L_{\mathrm{op},1}
        +
        L_{\mathrm{op},2}
        +
        L_{\mathrm{op},3}
        +
        \Gamma_{\mathrm{ER}}(M_{\mathrm{ER}}-\mathbb I)
        \right]\mathbf p
        +
        R_{\mathrm{SE}}(T)
        \left[M_{\mathrm{SE}}[\mathbf p]-\mathbb I\right]\mathbf p .
        """
    )

    st.write("The spin-exchange rate is inferred from the alkali vapor density at the selected temperature:")

    st.latex(
        r"""
        R_{\mathrm{SE}}(T)
        =
        n(T)\,\sigma_{\mathrm{SE}}\,\bar v_{\mathrm{rel}},
        \qquad
        \bar v_{\mathrm{rel}}=\sqrt{\frac{16k_BT}{\pi m}} .
        """
    )

    st.write("In this population-only app, M_SE[p] is a mean-field collision map. It does not propagate spin-exchange coherences or pair correlations.")

    st.write("In the interface, each laser detuning is set relative to a selected pressure-shifted hyperfine transition. The entered pump rate R_pump is defined as the total absorption rate for atoms that are unpolarized within the ground hyperfine level F of that selected transition:")

    st.latex(
        r"""
        R_{\mathrm{pump}}
        =
        \frac{1}{2F_0+1}
        \sum_{m=-F_0}^{F_0}
        \sum_{F',m'}
        R_{F_0,m\rightarrow F',m'} .
        """
    )

    st.write("Here F0 and F0' are the ground and excited hyperfine levels of the selected reference transition. R_pump is normalized using only absorption from the unpolarized F0 ground manifold into the selected F0' excited manifold. Nearby hyperfine transitions are still included in the dynamics after this selected-transition normalization fixes the optical scale. The relative detuning is")

    st.latex(
        r"""
        \Delta_{\mathrm{rel}}
        =
        \nu_L - \nu_{F_0\rightarrow F'_0}(P_{N_2}).
        """
    )

    st.write("The app internally converts that to a detuning from the zero-pressure D-line center:")

    st.latex(
        r"""
        \Delta_L
        =
        \delta_{\mathrm{hfs}}(F_0,F'_0)
        +
        \beta_{N_2}P_{N_2}
        +
        \Delta_{\mathrm{rel}} .
        """
    )

    st.write("For any other hyperfine transition in the pumping-rate sum, the actual detuning is")

    st.latex(
        r"""
        \Delta_{F,F'}
        =
        \Delta_L
        -
        \left[
        \delta_{\mathrm{hfs}}(F,F')
        +
        \beta_{N_2}P_{N_2}
        \right].
        """
    )

    st.write("A negative βN₂ means the optical resonance shifts to lower frequency.")