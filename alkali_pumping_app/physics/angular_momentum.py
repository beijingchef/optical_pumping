"""Angular-momentum state construction, basis transforms, and dipole strengths."""

from math import sqrt

import numpy as np
from sympy import S
from sympy.physics.wigner import wigner_3j, wigner_6j

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


def cg_coeff_F_to_mI_mS(I, F, m, mI, mS):
    """Return <I mI, S mS | F m> for electron spin S=1/2."""
    electron_spin = 0.5
    if abs(mI + mS - m) > 1e-9:
        return 0.0

    try:
        exponent = int(round(I - electron_spin + m))
        return float(
            (-1) ** exponent
            * sqrt(2 * F + 1)
            * float(
                wigner_3j(
                    S(I), S(electron_spin), S(F), S(mI), S(mS), S(-m)
                )
            )
        )
    except Exception:
        return 0.0


def coupled_basis_amplitudes(atom, ground_states):
    """Return real amplitudes <I mI, S mS | F m> for ground states."""
    mI_list = m_values(atom["I"])
    mS_list = [-0.5, 0.5]
    amplitudes = np.zeros(
        (len(ground_states), len(mI_list), len(mS_list)), dtype=float
    )
    for ai, state in enumerate(ground_states):
        for ii, mI in enumerate(mI_list):
            for si, mS in enumerate(mS_list):
                amplitudes[ai, ii, si] = cg_coeff_F_to_mI_mS(
                    atom["I"], state["F"], state["m"], mI, mS
                )
    return amplitudes


def hyperfine_uncoupled_probabilities(atom, ground_states):
    """Return |<I mI, S mS | F m>|^2 and the uncoupled basis axes."""
    mI_list = m_values(atom["I"])
    mS_list = [-0.5, 0.5]
    return coupled_basis_amplitudes(atom, ground_states) ** 2, mI_list, mS_list


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
