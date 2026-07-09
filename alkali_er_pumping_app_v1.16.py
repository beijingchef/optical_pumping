# alkali_er_pumping_app.py
#
# Streamlit app:
#   Steady-state ground-state population distribution of alkali vapors
#   with N2 buffer gas, three monochromatic optical-pumping beams,
#   pressure broadening, pressure shift, and electron-randomization relaxation.
#
# Run:
#   pip install streamlit numpy scipy sympy pandas matplotlib
#   streamlit run alkali_er_pumping_app.py
#
# Model:
#   dp/dt = [L_op,1 + L_op,2 + L_op,3 + Gamma_ER (M_ER - I)] p
#
# The app is population-only. It does not keep Zeeman coherences, excited-state
# coherences, spin-exchange coherences, or optical propagation effects.

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

    E_plus = -(Ex + 1j * Ey) / sqrt(2)
    E_zero = Ez
    E_minus = (Ex - 1j * Ey) / sqrt(2)

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

def voigt_profile_relative(delta_MHz, lorentz_fwhm_MHz, doppler_fwhm_MHz_val):
    """
    Dimensionless Voigt profile normalized to V(0)=1.
    delta_MHz is the laser detuning from the pressure-shifted hyperfine transition.
    """
    gamma_hwhm = lorentz_fwhm_MHz / 2.0
    sigma = doppler_fwhm_MHz_val / (2 * sqrt(2 * np.log(2)))

    if gamma_hwhm <= 0:
        gamma_hwhm = 1e-12

    if sigma <= 1e-12:
        return 1.0 / (1.0 + (delta_MHz / gamma_hwhm)**2)

    z = (delta_MHz + 1j * gamma_hwhm) / (sigma * sqrt(2))
    V = np.real(wofz(z)) / (sigma * sqrt(2 * pi))

    z0 = (1j * gamma_hwhm) / (sigma * sqrt(2))
    V0 = np.real(wofz(z0)) / (sigma * sqrt(2 * pi))

    if V0 <= 0:
        return 0.0
    return float(max(0.0, V / V0))


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

    Absolute optical frequencies are shown in THz.
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

        pump1_abs_THz = np.nan
        pump2_abs_THz = np.nan
        pump3_abs_THz = np.nan
        for beam in pump_beams:
            if beam.get("line") != line:
                continue
            pump_abs_THz = MHz_to_THz(line_center_MHz + float(beam.get("detuning", 0.0)))
            if beam.get("name") == "Beam 1":
                pump1_abs_THz = pump_abs_THz
            elif beam.get("name") == "Beam 2":
                pump2_abs_THz = pump_abs_THz
            elif beam.get("name") == "Beam 3":
                pump3_abs_THz = pump_abs_THz

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

                rows.append({
                    "Line": line,
                    "Fg": f"{Fg:g}",
                    "F'": f"{Fe:g}",
                    "nu_D_absolute": MHz_to_THz(line_center_MHz),
                    "detuning_zero_pressure": det0,
                    "N2_shift": pressure_shift,
                    "detuning_with_N2": detP,
                    "transition_frequency_with_N2": MHz_to_THz(transition_abs_MHz),
                    "pump_1_frequency": pump1_abs_THz,
                    "pump_2_frequency": pump2_abs_THz,
                    "pump_3_frequency": pump3_abs_THz,
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
            label = f"{line} F={Fg:g} → F'={Fe:g}"
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


def make_current_condition_dict():
    """Collect all user-facing simulation conditions from st.session_state."""
    keys = [
        "atom_name", "gamma_ER", "q_axis", "temperature_C_for_table", "n2_pressure_torr",
        "D1_width", "D2_width", "D1_shift", "D2_shift",
        "line1", "transition1", "det_rel1", "rate1", "k1", "pol1",
        "line2", "transition2", "det_rel2", "rate2", "k2", "pol2",
        "line3", "transition3", "det_rel3", "rate3", "k3", "pol3",
        "show_allowed_only", "show_rate_matrices",
    ]
    return {
        "app": "alkali_er_pumping_app",
        "version": 2,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "conditions": {k: st.session_state.get(k) for k in keys},
    }


def apply_loaded_condition_dict(payload):
    """Load a saved condition dictionary into st.session_state."""
    if not isinstance(payload, dict):
        raise ValueError("The uploaded file is not a JSON object.")

    conditions = payload.get("conditions", payload)
    if not isinstance(conditions, dict):
        raise ValueError("The uploaded JSON does not contain a valid conditions object.")

    for key, value in conditions.items():
        if value is not None:
            st.session_state[key] = value

    # Prevent atom-change default logic from overwriting loaded N2 coefficients.
    if conditions.get("atom_name") is not None:
        st.session_state["_last_atom_name_for_defaults"] = conditions["atom_name"]

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
    R_ge = np.zeros((N, len(excited_states)), dtype=float)
    delta_ge_MHz = np.zeros((N, len(excited_states)), dtype=float)

    for gi, g in enumerate(ground_states):
        for ei, e in enumerate(excited_states):
            hfs_shift = transition_shift_MHz(g, e)

            # Laser detuning relative to the actual pressure-shifted transition.
            delta_to_transition = detuning_MHz - (hfs_shift + pressure_shift_MHz)
            delta_ge_MHz[gi, ei] = delta_to_transition

            profile = voigt_profile_relative(
                delta_to_transition,
                lorentz_fwhm,
                doppler_fwhm,
            )

            strength_sum = 0.0
            for q, wq in q_weights.items():
                strength_sum += wq * dipole_strength(
                    I, Jg, Je,
                    g["F"], g["m"],
                    e["F"], e["m"],
                    q,
                )

            R_ge[gi, ei] = strength_sum * profile

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

    For each transition, the weak-drive two-level relation used here is

        δω_ge = R_ge * Δ_ge / Γ_L,

    where R_ge is the excitation/scattering rate in s^-1, Δ_ge is the detuning
    from that transition, and Γ_L is the Lorentzian FWHM in the same frequency
    units as Δ_ge. The result δω is an angular-frequency shift in rad/s, so the
    returned column is δω/(2π) in Hz. Summing over excited states gives the total
    diagonal AC-Stark shift of each |F,m> state; scalar, vector, and tensor parts
    are all included in this total diagonal shift.
    """
    if len(diagnostics) == 0:
        return np.zeros(len(ground_states), dtype=float), True

    # Require every active pump beam to be diagonal in the chosen quantization basis.
    for b, _info in diagnostics:
        if not light_shift_is_diagonal_for_beam(b["k_axis"], b["pol"], b["q_axis"]):
            return np.full(len(ground_states), np.nan, dtype=float), False

    shift_angular = np.zeros(len(ground_states), dtype=float)
    for _b, info in diagnostics:
        gamma_L_MHz = info["lorentz_fwhm_MHz"]
        if gamma_L_MHz <= 0:
            continue
        R_ge = info["R_ge"]
        delta_ge_MHz = info["delta_ge_MHz"]
        shift_angular += np.sum(R_ge * delta_ge_MHz / gamma_L_MHz, axis=1)

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
        ("nu_D_absolute", "ν<sub>D</sub> absolute", "THz", "9f"),
        ("detuning_zero_pressure", "ν(F→F′) − ν<sub>D</sub>, P=0", "MHz", "1f"),
        ("N2_shift", "N<sub>2</sub> shift βP", "MHz", "1f"),
        ("detuning_with_N2", "ν(F→F′) − ν<sub>D</sub>, with N<sub>2</sub>", "MHz", "1f"),
        ("transition_frequency_with_N2", "ν(F→F′), with N<sub>2</sub>", "THz", "9f"),
        ("pump_1_frequency", "ν<sub>pump,1</sub>", "THz", "9f"),
        ("pump_2_frequency", "ν<sub>pump,2</sub>", "THz", "9f"),
        ("pump_3_frequency", "ν<sub>pump,3</sub>", "THz", "9f"),
        ("lorentz_FWHM_total", "Lorentz FWHM total", "MHz", "1f"),
        ("doppler_FWHM", "Doppler FWHM", "MHz", "1f"),
        ("beta_width", "β<sub>width</sub>", "MHz/Torr", "1f"),
        ("beta_shift", "β<sub>shift</sub>", "MHz/Torr", "1f"),
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


# ============================================================
# 9. Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Alkali ER Optical Pumping",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Alkali steady-state populations: three pumps + N₂ + ER")

# ============================================================
# Sidebar: all input conditions
# ============================================================

with st.sidebar:
    # Defaults used before the Display widgets are rendered at the bottom.
    if "show_allowed_only" not in st.session_state:
        st.session_state["show_allowed_only"] = True
    if "show_rate_matrices" not in st.session_state:
        st.session_state["show_rate_matrices"] = False

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
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
            display: none;
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

    st.caption("Conditions")
    load_col, save_col = st.columns(2, gap="small")

    with load_col:
        uploaded_condition = st.file_uploader(
            "Load condition",
            type=["json"],
            key="condition_file_uploader",
            help="Choose a saved JSON condition file. It loads immediately after selection.",
            label_visibility="collapsed",
        )

    with save_col:
        condition_json = json.dumps(make_current_condition_dict(), indent=2)
        st.download_button(
            "Save condition",
            data=condition_json,
            file_name=f"alkali_er_condition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="save_condition_button",
            use_container_width=True,
        )

    if uploaded_condition is not None:
        load_signature = (uploaded_condition.name, uploaded_condition.size)
        if st.session_state.get("_loaded_condition_signature") != load_signature:
            try:
                payload = json.loads(uploaded_condition.getvalue().decode("utf-8"))
                apply_loaded_condition_dict(payload)
                st.session_state["_loaded_condition_signature"] = load_signature
                st.success("Condition loaded.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load condition file: {exc}")

    st.divider()
    st.header("Atom / cell")

    atom_row_col1, atom_row_col2 = st.columns(2, gap="small")
    with atom_row_col1:
        atom_name = st.selectbox("Alkali atom", list(ATOMS.keys()), index=0, key="atom_name")
    with atom_row_col2:
        n2_pressure_torr = st.number_input(
            "N₂ pressure (Torr)",
            value=0.0,
            min_value=0.0,
            step=10.0,
            key="n2_pressure_torr",
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
        temperature_C = st.number_input(
            "Temperature (°C)",
            value=25.0,
            step=1.0,
            key="temperature_C_for_table",
        )
    with cell_row_col2:
        gamma_ER = st.number_input(
            "ER rate ΓER (s⁻¹)",
            value=1.0,
            min_value=0.0,
            step=1.0,
            key="gamma_ER",
        )
    with cell_row_col3:
        q_axis = st.selectbox(
            "Quantization axis",
            ["z", "x", "y"],
            index=0,
            key="q_axis",
        )

    with st.expander("N₂ coefficients", expanded=False):
        c1, c2 = st.columns(2, gap="small")
        with c1:
            D1_width = st.number_input(
                "D1 width",
                value=float(DEFAULT_N2_COEFFS[atom_name]["D1"]["width"]),
                step=0.1,
                key="D1_width",
                help="N2 pressure broadening coefficient, FWHM, MHz/Torr",
            )
            D2_width = st.number_input(
                "D2 width",
                value=float(DEFAULT_N2_COEFFS[atom_name]["D2"]["width"]),
                step=0.1,
                key="D2_width",
                help="N2 pressure broadening coefficient, FWHM, MHz/Torr",
            )

        with c2:
            D1_shift = st.number_input(
                "D1 shift",
                value=float(DEFAULT_N2_COEFFS[atom_name]["D1"]["shift"]),
                step=0.1,
                key="D1_shift",
                help="N2 pressure shift coefficient, MHz/Torr",
            )
            D2_shift = st.number_input(
                "D2 shift",
                value=float(DEFAULT_N2_COEFFS[atom_name]["D2"]["shift"]),
                step=0.1,
                key="D2_shift",
                help="N2 pressure shift coefficient, MHz/Torr",
            )

        st.caption("Widths and shifts are in MHz/Torr. Negative shift = red shift.")

    n2_coeffs = {
        "D1": {"width": D1_width, "shift": D1_shift},
        "D2": {"width": D2_width, "shift": D2_shift},
    }

    show_allowed_only = st.session_state["show_allowed_only"]
    show_rate_matrices = st.session_state["show_rate_matrices"]

    def beam_config_ui(beam_number, default_line_index=0):
        st.header(f"Beam {beam_number}")
        line = st.selectbox("Line", ["D1", "D2"], index=default_line_index, key=f"line{beam_number}")

        transition_options = transition_choice_labels(
            atom, line, n2_pressure_torr, n2_coeffs, allowed_only=show_allowed_only
        )
        transition_key = f"transition{beam_number}"
        if transition_options and st.session_state.get(transition_key) not in transition_options:
            st.session_state[transition_key] = transition_options[0]
        transition = st.selectbox(
            "Reference hyperfine transition",
            transition_options,
            key=transition_key,
            help="Laser detuning is defined relative to this pressure-shifted hyperfine transition.",
        )
        det_rel = st.number_input(
            "Detuning from selected transition (MHz)",
            value=0.0,
            step=10.0,
            key=f"det_rel{beam_number}",
        )
        rate = st.number_input(
            "Rₚᵤₘₚ: pumping rate for selected transition (s⁻¹)",
            value=10.0,
            min_value=0.0,
            step=10.0,
            key=f"rate{beam_number}",
        )
        k_axis = st.selectbox(
            "Propagation direction",
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
        line1, transition1, det_rel1, rate1, k1, pol1 = beam_config_ui(1)
    with bcol2:
        line2, transition2, det_rel2, rate2, k2, pol2 = beam_config_ui(2)
    with bcol3:
        line3, transition3, det_rel3, rate3, k3, pol3 = beam_config_ui(3)

    st.divider()
    st.header("Display")
    show_allowed_only = st.checkbox(
        "Only show allowed hyperfine transitions",
        value=True,
        key="show_allowed_only",
    )
    show_rate_matrices = st.checkbox(
        "Show rate matrices",
        value=False,
        key="show_rate_matrices",
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

M_ER = build_ER_matrix(atom, ground_states)
L_total += gamma_ER * (M_ER - np.eye(N))

p_ss = steady_state_from_L(L_total)

light_shift_Hz, light_shift_available = total_light_shift_Hz_from_diagnostics(
    ground_states,
    beam_inputs,
    diagnostics,
)

df_pop = pd.DataFrame({
    "F": [g["F"] for g in ground_states],
    "m": [g["m"] for g in ground_states],
    "population": p_ss,
    "light_shift_Hz": light_shift_Hz,
})
df_pop = add_population_difference_column(df_pop)
df_pop = add_light_shift_difference_column(df_pop)

df_F = population_by_F(df_pop)

df_trans = hyperfine_transition_table(
    atom=atom,
    n2_pressure_torr=n2_pressure_torr,
    n2_coeffs=n2_coeffs,
    allowed_only=show_allowed_only,
    pump_beams=beam_inputs,
)

labels = [g["label"] for g in ground_states]

df_pop_display = df_pop.rename(columns={
    "population": "Pₘ",
    "population_difference": "Pₘ − Pₘ₋₁",
    "light_shift_Hz": "νLS (Hz)",
    "light_shift_difference_Hz": "νₘ − νₘ₋₁ (Hz)",
})
# Keep the light-shift columns visible and placed at the far right of the table.
df_pop_display = df_pop_display[["F", "m", "Pₘ", "Pₘ − Pₘ₋₁", "νLS (Hz)", "νₘ − νₘ₋₁ (Hz)"]]


# ============================================================
# Main compact layout
# ============================================================

left, right = st.columns([0.62, 1.63], gap="small")

with left:
    st.subheader(f"{atom_name} ground-state populations")

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.bar(labels, p_ss)
    ax.set_ylabel("Population")
    ax.set_xlabel(rf"$|F,m\rangle$ along {q_axis}")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

with right:
    # Give the Zeeman table more width so the final light-shift column is visible.
    cc1, cc2 = st.columns([1.33, 0.54], gap="small")

    with cc1:
        st.caption("Zeeman sublevel population & shift")
        st.dataframe(
            df_pop_display.style.format({
                "F": "{:g}",
                "m": "{:g}",
                "Pₘ": "{:.3f}",
                "Pₘ − Pₘ₋₁": "{:.3f}",
                "νLS (Hz)": "{:.1f}",
                "νₘ − νₘ₋₁ (Hz)": "{:.1f}",
            }, na_rep=""),
            use_container_width=True,
            height=315,
        )
        if light_shift_available:
            st.caption("νLS is shown because all active pump-beam light-shift Hamiltonians commute with the selected quantization-axis spin component.")
        else:
            st.caption("νLS is blank because at least one active beam has multiple spherical polarization components relative to the quantization axis, so the light-shift Hamiltonian may not commute with the selected spin component.")

    with cc2:
        st.caption("Population by F")
        st.dataframe(
            df_F.style.format({
                "F": "{:g}",
                "population": "{:.6f}",
            }),
            use_container_width=True,
            height=100,
        )

        m1 = expectation_m(ground_states, p_ss)
        m2 = expectation_m2(ground_states, p_ss)

        st.write(f"Σp = {p_ss.sum():.8f}")
        st.write(f"〈m〉 = {m1:.6f}")
        st.write(f"〈m²〉 = {m2:.6f}")

st.subheader("D1/D2 hyperfine transition detunings")
st.caption(
    "Detunings are relative to the corresponding zero-pressure D1 or D2 fine-structure line center. "
    "Absolute optical frequencies are shown in THz. Blank pump-frequency cells mean that pump beam is on the other D line."
)

st.markdown(
    render_transition_table_html(df_trans),
    unsafe_allow_html=True,
)

if show_rate_matrices:
    with st.expander("Total rate matrix L", expanded=False):
        st.write("Columns are source states; rows are destination states. dp/dt = L p.")
        Ldf = pd.DataFrame(L_total, index=labels, columns=labels)
        st.dataframe(Ldf.style.format("{:.3e}"), use_container_width=True)

    with st.expander("ER redistribution matrix M_ER", expanded=False):
        st.write("After one ER collision: p → M_ER p.")
        Mdf = pd.DataFrame(M_ER, index=labels, columns=labels)
        st.dataframe(Mdf.style.format("{:.4f}"), use_container_width=True)

with st.expander("Beam diagnostics", expanded=False):
    if len(diagnostics) == 0:
        st.info("Both optical pumping rates are zero.")
    else:
        for i, (b, info) in enumerate(diagnostics, start=1):
            qtxt = ", ".join(
                [f"q={q}: {w:.3f}" for q, w in info["q_weights"].items()]
            )
            st.markdown(f"**{b['name']}**")
            st.write(
                f"{b['line']}, k={b['k_axis']}, pol={b['pol']}"
            )
            st.write(f"Reference transition: {b['transition_label']}")
            st.write(
                f"Relative detuning = {b['detuning_relative']:.3f} MHz; "
                f"absolute detuning from zero-pressure {b['line']} center = {b['detuning']:.3f} MHz"
            )
            st.write(f"Spherical weights relative to quantization axis {q_axis}: {qtxt}")
            st.write(
                f"Natural FWHM = {info['gamma_nat_MHz']:.3f} MHz; "
                f"N2 FWHM = {info['pressure_width_MHz']:.3f} MHz; "
                f"total Lorentz FWHM = {info['lorentz_fwhm_MHz']:.3f} MHz"
            )
            st.write(
                f"Doppler FWHM = {info['doppler_fwhm_MHz']:.3f} MHz; "
                f"N2 pressure shift = {info['pressure_shift_MHz']:.3f} MHz"
            )
            if info.get("reference_Fg") is not None:
                st.write(
                    f"Rₚᵤₘₚ normalization: average total absorption from unpolarized "
                    f"F={info['reference_Fg']:g} ground sublevels = {b['rate']:.6g} s⁻¹."
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
        \delta\omega_{F,m}
        =
        \sum_{F',m'}
        R_{F,m\rightarrow F',m'}
        \frac{\Delta_{F,m\rightarrow F',m'}}{\Gamma_L}
        """
    )
    st.write(
        "The table reports νLS = δω/(2π) in Hz and νₘ − νₘ₋₁ in Hz. The sum over excited states gives "
        "the total diagonal AC-Stark shift, including scalar, vector, and tensor "
        "contributions. This small shift is not fed back into the optical detunings."
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
        \right]\mathbf p .
        """
    )

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