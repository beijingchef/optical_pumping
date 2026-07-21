"""Reference atomic properties and collision helpers for the settings dialog."""

from fractions import Fraction
from math import pi, sqrt

from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_6j


K_B = 1.380649e-23
AMU_KG = 1.66053906660e-27
TORR_PA = 133.32236842105263


ATOMIC_PROPERTY_DATA = {
    "Na23": {
        "element": "Na",
        "label": "²³Na",
        "mass_amu": 22.9897692807,
        "I": 3 / 2,
        "melting_K": 370.95,
        "vapor": {
            "solid": {"A": 8.179, "B": 5603.0},
            "liquid": {"A": 7.585, "B": 5377.0},
        },
        "spin_exchange_cross_section_cm2": 1.0e-14,
    },
    "K39": {
        "element": "K",
        "label": "³⁹K",
        "mass_amu": 38.9637064864,
        "I": 3 / 2,
        "melting_K": 336.53,
        "vapor": {
            "solid": {"A": 7.842, "B": 4646.0},
            "liquid": {"A": 7.283, "B": 4453.0},
        },
        "spin_exchange_cross_section_cm2": 1.5e-14,
    },
    "K41": {
        "element": "K",
        "label": "⁴¹K",
        "mass_amu": 40.9618252579,
        "I": 3 / 2,
        "melting_K": 336.53,
        "vapor": {
            "solid": {"A": 7.842, "B": 4646.0},
            "liquid": {"A": 7.283, "B": 4453.0},
        },
        "spin_exchange_cross_section_cm2": 1.5e-14,
    },
    "Rb85": {
        "element": "Rb",
        "label": "⁸⁵Rb",
        "mass_amu": 84.9117897379,
        "I": 5 / 2,
        "melting_K": 312.46,
        "vapor": {
            "solid": {"A": 7.738, "B": 4215.0},
            "liquid": {"A": 7.193, "B": 4040.0},
        },
        "spin_exchange_cross_section_cm2": 1.9e-14,
    },
    "Rb87": {
        "element": "Rb",
        "label": "⁸⁷Rb",
        "mass_amu": 86.9091805310,
        "I": 3 / 2,
        "melting_K": 312.46,
        "vapor": {
            "solid": {"A": 7.738, "B": 4215.0},
            "liquid": {"A": 7.193, "B": 4040.0},
        },
        "spin_exchange_cross_section_cm2": 1.9e-14,
    },
    "Cs133": {
        "element": "Cs",
        "label": "¹³³Cs",
        "mass_amu": 132.9054519610,
        "I": 7 / 2,
        "melting_K": 301.59,
        "vapor": {
            "solid": {"A": 7.592, "B": 3999.0},
            "liquid": {"A": 7.046, "B": 3830.0},
        },
        "spin_exchange_cross_section_cm2": 2.1e-14,
    },
}


BUFFER_GASES = {
    "N2": {"label": "N₂", "mass_amu": 28.0134},
    "He4": {"label": "⁴He", "mass_amu": 4.00260325413},
    "CH4": {"label": "CH₄", "mass_amu": 16.04246},
}


# Ground-state electron-randomization cross sections from Table 10.3 of
# Happer, Jau, and Walker, Optically Pumped Atoms (2010). None means that the
# reference does not provide a value. The listed measurements are strongly
# temperature dependent, so the dialog exposes every value as an editable input.
ELECTRON_RANDOMIZATION_CROSS_SECTION_CM2 = {
    "Na": {"N2": None, "He4": 0.0036e-22, "CH4": None},
    "K": {"N2": 0.75e-22, "He4": 0.005e-22, "CH4": None},
    "Rb": {"N2": 1.0e-22, "He4": 0.087e-22, "CH4": None},
    "Cs": {"N2": 5.5e-22, "He4": 0.24e-22, "CH4": None},
}


# Optical Lorentzian FWHM broadening and line-shift coefficients in MHz/Torr.
# Entries are element based because isotope shifts are not resolved here.
PRESSURE_COEFFICIENTS_MHZ_TORR = {
    "Na": {},
    "K": {
        "D1": {
            "N2": (17.78, -6.80),
            "He4": (13.08, 1.63),
            "CH4": (29.35, -7.41),
        },
        "D2": {
            "N2": (18.98, -5.66),
            "He4": (19.84, 0.52),
            "CH4": (27.78, -8.38),
        },
    },
    "Rb": {
        "D1": {
            "N2": (17.8, -8.25),
            "He4": (20.80, 5.80),
            "CH4": (32.78, -6.96),
        },
        "D2": {"N2": (18.1, -5.90)},
    },
    "Cs": {
        "D1": {"N2": (19.51, -8.23)},
        "D2": {"N2": (22.68, -6.73)},
    },
}


def atomic_property_record(isotope):
    """Return the reference record for a supported isotope."""
    return ATOMIC_PROPERTY_DATA[isotope]


def alkali_thermal_properties(isotope, temperature_C):
    """Return saturated-vapor and thermal quantities for one isotope."""
    atom = atomic_property_record(isotope)
    temperature_K = float(temperature_C) + 273.15
    if temperature_K <= 0.0:
        raise ValueError("Temperature must be above absolute zero.")

    phase = "solid" if temperature_K < atom["melting_K"] else "liquid"
    coeff = atom["vapor"][phase]
    pressure_torr = 10.0 ** (coeff["A"] - coeff["B"] / temperature_K)
    density_cm3 = pressure_torr * TORR_PA / (K_B * temperature_K) / 1e6
    mass_kg = atom["mass_amu"] * AMU_KG
    rms_velocity_m_s = sqrt(3.0 * K_B * temperature_K / mass_kg)
    mean_relative_velocity_m_s = sqrt(
        16.0 * K_B * temperature_K / (pi * mass_kg)
    )
    sigma_cm2 = atom["spin_exchange_cross_section_cm2"]
    spin_exchange_rate_s = (
        density_cm3 * sigma_cm2 * mean_relative_velocity_m_s * 100.0
    )
    return {
        "temperature_K": temperature_K,
        "phase": phase,
        "pressure_torr": pressure_torr,
        "density_cm3": density_cm3,
        "rms_velocity_m_s": rms_velocity_m_s,
        "mean_relative_velocity_m_s": mean_relative_velocity_m_s,
        "spin_exchange_rate_s": spin_exchange_rate_s,
    }


def buffer_gas_collision_rate_s(
    isotope,
    gas_name,
    temperature_C,
    pressure_torr,
    cross_section_cm2,
):
    """Return n_gas * sigma * mean relative speed for an alkali-buffer pair."""
    temperature_K = float(temperature_C) + 273.15
    if temperature_K <= 0.0:
        raise ValueError("Temperature must be above absolute zero.")
    atom_mass = atomic_property_record(isotope)["mass_amu"] * AMU_KG
    gas_mass = BUFFER_GASES[gas_name]["mass_amu"] * AMU_KG
    reduced_mass = atom_mass * gas_mass / (atom_mass + gas_mass)
    mean_relative_velocity_cm_s = 100.0 * sqrt(
        8.0 * K_B * temperature_K / (pi * reduced_mass)
    )
    density_cm3 = float(pressure_torr) * TORR_PA / (K_B * temperature_K) / 1e6
    return density_cm3 * float(cross_section_cm2) * mean_relative_velocity_cm_s


def allowed_hyperfine_F(I, J):
    """Return all angular-momentum-allowed hyperfine F values."""
    lower = abs(float(I) - float(J))
    upper = float(I) + float(J)
    count = int(round(upper - lower)) + 1
    return [lower + index for index in range(count)]


def magnetic_sublevels(F):
    """Return m=-F,...,+F for integer or half-integer F."""
    return [-float(F) + index for index in range(int(round(2.0 * F)) + 1)]


def _half_integer(value):
    return Rational(int(round(2.0 * float(value))), 2)


def notebook_transition_strength(I, Je, Fg, mg, Fe, me):
    """Reproduce the transition-strength convention used in the notebook."""
    Jg = Rational(1, 2)
    I_r = _half_integer(I)
    Je_r = _half_integer(Je)
    Fg_r = _half_integer(Fg)
    Fe_r = _half_integer(Fe)
    mg_r = _half_integer(mg)
    me_r = _half_integer(me)
    q_r = me_r - mg_r

    if abs(float(q_r)) > 1.0 or (Fg_r == 0 and Fe_r == 0):
        return 0.0
    value = (
        3
        * (2 * Jg + 1)
        * (2 * Je_r + 1)
        * (2 * Fg_r + 1)
        * (2 * Fe_r + 1)
        * wigner_6j(1, 0, 1, Jg, Je_r, Rational(1, 2)) ** 2
        * wigner_6j(Je_r, Jg, 1, Fg_r, Fe_r, I_r) ** 2
        * wigner_3j(Fe_r, 1, Fg_r, -me_r, q_r, mg_r) ** 2
    )
    return float(value)


def format_transition_strength_fraction(value):
    """Format a transition strength as a reduced rational fraction."""
    fraction = Fraction(float(value)).limit_denominator(1_000_000)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def format_transition_strength_vertical_fraction(value):
    """Format a transition strength as a stacked MathText fraction."""
    fraction = Fraction(float(value)).limit_denominator(1_000_000)
    if fraction.denominator == 1:
        return rf"${fraction.numerator}$"
    return rf"$\frac{{{fraction.numerator}}}{{{fraction.denominator}}}$"


def grotrian_transitions(I, line, Fg, Fe, polarizations=(-1, 0, 1)):
    """Return all nonzero Zeeman transitions for one hyperfine branch."""
    Je = 0.5 if line == "D1" else 1.5
    selected_q = {int(q) for q in polarizations}
    transitions = []
    for mg in magnetic_sublevels(Fg):
        for me in magnetic_sublevels(Fe):
            q = int(round(me - mg))
            if abs((me - mg) - q) > 1e-9 or q not in selected_q:
                continue
            strength = notebook_transition_strength(I, Je, Fg, mg, Fe, me)
            if strength > 1e-14:
                transitions.append(
                    {"mg": mg, "me": me, "q": q, "strength": strength}
                )
    return transitions
