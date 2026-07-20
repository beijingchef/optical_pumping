"""Weak-RF adjacent-coherence response calculations."""

from math import sqrt

import numpy as np

# ============================================================
# 7c. Weak linearly polarized RF response
# ============================================================

def lab_axis_in_local_frame(q_axis, lab_axis):
    """Map a laboratory axis to the local frame whose z axis is q_axis.

    This uses the same right-handed local frames as local_components():
      q=z: (x_local,y_local,z_local) = (x,y,z)
      q=x: (x_local,y_local,z_local) = (y,z,x)
      q=y: (x_local,y_local,z_local) = (z,x,y)
    """
    mappings = {
        "z": {"x": "x", "y": "y", "z": "z"},
        "x": {"x": "z", "y": "x", "z": "y"},
        "y": {"x": "y", "y": "z", "z": "x"},
    }
    return mappings[q_axis][lab_axis]


def adjacent_spin_matrix_elements(local_axis, ladder_coefficient):
    """Return O_ab and O_ba for a=|F,m>, b=|F,m-1>."""
    C = float(ladder_coefficient)
    if local_axis == "x":
        return 0.5 * C, 0.5 * C
    if local_axis == "y":
        return -0.5j * C, 0.5j * C
    if local_axis == "z":
        return 0.0j, 0.0j
    raise ValueError(local_axis)


def weak_rf_observable_susceptibility(
    frequencies_hz,
    ground_states,
    populations,
    adjacent_transition_hz,
    gamma_op,
    gamma_er,
    gamma_se,
    q_axis,
    rf_axis,
    observable,
    target_F=None,
):
    """Return the zero-drive RF susceptibility of one hyperfine manifold.

    Only adjacent coherences within ``target_F`` contribute. If ``target_F``
    is omitted, the upper ground hyperfine manifold (largest F) is used. The
    returned observable is therefore the upper-manifold spin component

        <F_{+,i}> = Tr[rho_{F+} F_i^(F+)],

    not the coherent sum of the two ground hyperfine spin observables.

    The applied field is linearly polarized along the selected laboratory
    axis i = x, y, or z:

        H_rf / hbar = Omega_1 F_i cos(omega t).

    The density matrix is expanded directly to first order,

        rho = rho_0 + Omega_1 rho^(1) + O(Omega_1^2),

    and this function calculates rho^(1) without assigning any finite numerical
    value to Omega_1. Let chi_plus be the coefficient of exp(-i omega t).
    The real time-domain susceptibility is written as

        d<F_{+,i}(t)>/dOmega_1
        = X(omega) cos(omega t) + Y(omega) sin(omega t),

    with

        X = 2 Re[chi_plus],
        Y = 2 Im[chi_plus],
        A = sqrt(X^2 + Y^2) = 2 |chi_plus|.

    The function returns A, X, and Y within the linear weak-drive
    approximation.

    Every adjacent coherence is treated independently with its local rate

        Gamma_m = Gamma_OP,m + Gamma_ER,m + Gamma_SE,m.

    Both Fourier components of the real cosine field are retained. This is
    important when a negative signed transition frequency contributes to a
    positive-frequency resonance through the counter-rotating component.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    p = np.asarray(populations, dtype=float)
    transition_hz = np.asarray(adjacent_transition_hz, dtype=float)
    gamma_op = np.asarray(gamma_op, dtype=float)
    gamma_er = np.asarray(gamma_er, dtype=float)
    gamma_se = np.asarray(gamma_se, dtype=float)

    susceptibility_amplitude = np.zeros_like(frequencies_hz, dtype=float)
    susceptibility_in_phase = np.zeros_like(frequencies_hz, dtype=float)
    susceptibility_quadrature = np.zeros_like(frequencies_hz, dtype=float)
    susceptibility_phasor = np.zeros_like(frequencies_hz, dtype=complex)

    if len(ground_states) == 0:
        return (
            susceptibility_amplitude,
            susceptibility_in_phase,
            susceptibility_quadrature,
            {
                "used_transitions": 0,
                "nonpositive_linewidths": 0,
                "target_F": np.nan,
            },
        )

    if target_F is None:
        target_F = max(float(state["F"]) for state in ground_states)
    target_F = float(target_F)

    drive_local_axis = lab_axis_in_local_frame(q_axis, rf_axis)
    observable_lab_axis = observable[-1].lower()
    observable_local_axis = lab_axis_in_local_frame(q_axis, observable_lab_axis)

    state_index = {
        (float(state["F"]), float(state["m"])): idx
        for idx, state in enumerate(ground_states)
    }

    omega = 2.0 * np.pi * frequencies_hz
    used_transitions = 0
    nonpositive_linewidths = 0

    for a_idx, state in enumerate(ground_states):
        F = float(state["F"])
        if not np.isclose(F, target_F):
            continue
        m = float(state["m"])
        b_idx = state_index.get((F, m - 1.0))
        if b_idx is None or not np.isfinite(transition_hz[a_idx]):
            continue

        C2 = F * (F + 1.0) - m * (m - 1.0)
        if C2 <= 0:
            continue
        C = sqrt(C2)

        drive_ab, drive_ba = adjacent_spin_matrix_elements(drive_local_axis, C)
        obs_ab, obs_ba = adjacent_spin_matrix_elements(observable_local_axis, C)
        if abs(drive_ab) == 0.0 or (abs(obs_ab) == 0.0 and abs(obs_ba) == 0.0):
            continue

        gamma = gamma_op[a_idx] + gamma_er[a_idx] + gamma_se[a_idx]
        if not np.isfinite(gamma):
            continue
        if gamma <= 0.0:
            nonpositive_linewidths += 1
            gamma = 1e-12

        D_m = p[a_idx] - p[b_idx]
        omega_m = 2.0 * np.pi * transition_hz[a_idx]

        # Coefficients of exp(-i omega t) in d(rho_ab)/dOmega_1 and
        # d(rho_ba)/dOmega_1 at Omega_1=0. The factor 1/2 is the Fourier
        # amplitude of cos(omega t); no finite drive amplitude is introduced.
        drho_ab_plus_domega1 = (
            1j * 0.5 * drive_ab * D_m
            / (gamma + 1j * (omega_m - omega))
        )
        drho_ba_plus_domega1 = (
            -1j * 0.5 * drive_ba * D_m
            / (gamma + 1j * (-omega_m - omega))
        )

        susceptibility_phasor += (
            obs_ba * drho_ab_plus_domega1
            + obs_ab * drho_ba_plus_domega1
        )
        used_transitions += 1

    # With the RF reference chosen as cos(omega t),
    #
    #   d<O(t)>/dOmega_1 = X cos(omega t) + Y sin(omega t),
    #
    # where chi_plus is the coefficient of exp(-i omega t). Therefore
    # X = 2 Re(chi_plus), Y = 2 Im(chi_plus), and A = 2 |chi_plus|.
    susceptibility_in_phase = 2.0 * np.real(susceptibility_phasor)
    susceptibility_quadrature = 2.0 * np.imag(susceptibility_phasor)
    susceptibility_amplitude = np.hypot(
        susceptibility_in_phase,
        susceptibility_quadrature,
    )
    return (
        susceptibility_amplitude,
        susceptibility_in_phase,
        susceptibility_quadrature,
        {
            "used_transitions": used_transitions,
            "nonpositive_linewidths": nonpositive_linewidths,
            "drive_local_axis": drive_local_axis,
            "observable_local_axis": observable_local_axis,
            "target_F": target_F,
        },
    )


def largest_abs_Dm_relaxation_reference(df_pop, target_F=None):
    """Return the relaxation rate at the largest |D_m| in one F manifold.

    By default, the upper ground hyperfine manifold (largest F in ``df_pop``)
    is used. Here D_m = P_m - P_(m-1), as displayed in the Zeeman table, and
    the reference transition is selected by the largest absolute magnitude
    |D_m| within that same manifold.
    Rows without an adjacent lower-m partner are excluded. The total local
    relaxation rate is

        Gamma_m = Gamma_OP,m + Gamma_ER,m + Gamma_SE,m.
    """
    required = {
        "F", "m", "population_difference",
        "Gamma_OP", "Gamma_ER", "Gamma_SE",
    }
    if not required.issubset(df_pop.columns):
        return {
            "available": False,
            "reason": "required columns are missing",
        }

    if target_F is None:
        finite_F_values = df_pop.loc[
            np.isfinite(df_pop["F"].to_numpy(dtype=float)), "F"
        ].to_numpy(dtype=float)
        if finite_F_values.size == 0:
            return {
                "available": False,
                "reason": "no finite hyperfine manifold was found",
            }
        target_F = float(np.max(finite_F_values))
    target_F = float(target_F)

    candidate = df_pop.loc[:, list(required)].copy()
    candidate = candidate.loc[np.isclose(candidate["F"].to_numpy(dtype=float), target_F)]
    finite_D = np.isfinite(candidate["population_difference"].to_numpy(dtype=float))
    candidate = candidate.loc[finite_D]
    if candidate.empty:
        return {
            "available": False,
            "reason": f"no adjacent transition in F={target_F:g} has a finite D_m",
        }

    # Select the adjacent transition with the largest population-difference magnitude.
    row_index = candidate["population_difference"].abs().idxmax()
    row = df_pop.loc[row_index]
    gamma_total = float(row["Gamma_OP"] + row["Gamma_ER"] + row["Gamma_SE"])

    if not np.isfinite(gamma_total) or gamma_total <= 0.0:
        return {
            "available": False,
            "reason": "the selected total relaxation rate is not positive and finite",
            "F": float(row["F"]),
            "m": float(row["m"]),
            "D_m": float(row["population_difference"]),
            "Gamma_m": gamma_total,
        }

    return {
        "available": True,
        "F": float(row["F"]),
        "m": float(row["m"]),
        "D_m": float(row["population_difference"]),
        "Gamma_m": gamma_total,
        "Gamma_OP": float(row["Gamma_OP"]),
        "Gamma_ER": float(row["Gamma_ER"]),
        "Gamma_SE": float(row["Gamma_SE"]),
    }

