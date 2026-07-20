"""Doppler/Voigt profiles and hyperfine transition helpers."""

from math import pi, sqrt

import numpy as np
import pandas as pd
from scipy.special import wofz

from .angular_momentum import allowed_F, hfs_energy_MHz, hyperfine_transition_allowed

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
    temperature_C=23.5,
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
        doppler = doppler_fwhm_MHz(atom, line, float(temperature_C))

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

