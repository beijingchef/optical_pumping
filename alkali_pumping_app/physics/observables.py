"""Steady-state solver and population-derived observables."""

import numpy as np
import pandas as pd
from scipy.linalg import lstsq

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


def add_nu_m_column(df_pop):
    """Add the total adjacent-transition frequency nu_m in Hz.

    For each adjacent coherence rho_(m,m-1) within one F manifold,

        nu_m = [nu_LS(F,m) + nu_B(F,m)]
             - [nu_LS(F,m-1) + nu_B(F,m-1)].

    The lowest-m state has no adjacent lower-m partner and is left blank. If
    the optical light-shift Hamiltonian is not diagonal, nu_LS is unavailable
    and the total nu_m is left blank even though the diagonal Zeeman column is
    still shown.
    """
    df = df_pop.copy()
    df["nu_m"] = np.nan

    if "nu_LS" not in df.columns or "nu_B" not in df.columns:
        return df

    for _F_value, group in df.groupby("F", sort=False):
        group_sorted = group.sort_values("m")
        total_frequency_by_m = {
            float(row["m"]): (
                float(row["nu_LS"]) + float(row["nu_B"])
                if pd.notna(row["nu_LS"]) and pd.notna(row["nu_B"])
                else np.nan
            )
            for _, row in group_sorted.iterrows()
        }
        for row_index, row in group_sorted.iterrows():
            m_value = float(row["m"])
            previous_m = m_value - 1.0
            if previous_m not in total_frequency_by_m:
                continue
            current_nu = total_frequency_by_m[m_value]
            previous_nu = total_frequency_by_m[previous_m]
            if np.isfinite(current_nu) and np.isfinite(previous_nu):
                df.loc[row_index, "nu_m"] = current_nu - previous_nu

    return df


def total_G_OP_by_ground_state(ground_states, diagnostics):
    """Return G^OP for every displayed |F,m> ground state.

    For each active beam, R_ge[ground, excited] is the state-resolved optical
    excitation rate after the beam intensity has been normalized to its selected
    reference-transition pumping rate. Summing over all excited states and all
    active beams gives the total optical depopulation rate G^OP of each ground
    Zeeman sublevel.
    """
    rates = np.zeros(len(ground_states), dtype=float)
    for _beam, info in diagnostics:
        R_ge = np.asarray(info["R_ge"], dtype=float)
        if R_ge.shape[0] != len(ground_states):
            raise ValueError("Optical-pumping diagnostic has an incompatible ground-state dimension.")
        rates += np.sum(R_ge, axis=1)
    return rates


def optical_Lambda_fractional_rates(
    optical_generator, populations, G_OP_values, population_floor=1e-15
):
    """Return the optical repopulation rate Lambda for each ground state.

    The optical generator can be decomposed as

        L_op = W_rep - diag(G_OP),

    where G^OP is the total excitation/depopulation rate from state m and
    W_rep contains spontaneous-emission repopulation into the ground states,
    including return to the same ground state. At the supplied population
    distribution p, the repopulation flow into state m is (W_rep p)_m.

    The table reports the corresponding fractional repopulation rate

        Lambda = (W_rep p)_m / p_m,

    in s^-1, so that the signed net optical fractional population relaxation
    rate is G^OP - Lambda. States with negligible population are left blank.
    """
    L_op = np.asarray(optical_generator, dtype=float)
    p = np.asarray(populations, dtype=float)
    G_OP = np.asarray(G_OP_values, dtype=float)

    if L_op.shape != (len(p), len(p)) or G_OP.shape != p.shape:
        raise ValueError("Incompatible dimensions in optical repopulation calculation.")

    W_rep = L_op + np.diag(G_OP)
    repopulation_flow = W_rep @ p
    # Remove only tiny negative roundoff; physical repopulation flow is nonnegative.
    repopulation_flow[np.abs(repopulation_flow) < 1e-14] = 0.0

    Lambda = np.full_like(p, np.nan, dtype=float)
    mask = p > population_floor
    Lambda[mask] = repopulation_flow[mask] / p[mask]
    return Lambda


def add_adjacent_optical_relaxation_columns(df_pop):
    """Add Gamma^OP and Gamma^OP/(2 pi) for adjacent Zeeman coherences.

    For the adjacent coherence rho_{m,m-1}, direct optical depopulation gives

        Gamma^OP = (G^OP_m + G^OP_{m-1}) / 2

    in s^-1. The corresponding ordinary-frequency linewidth is
    Gamma^OP/(2 pi) in Hz. The lowest-m state in each F manifold has no adjacent
    lower-m partner, so both entries are blank there.
    """
    df = df_pop.copy()
    rate_column = "Gamma_OP"
    hz_column = "Gamma_OP_over_2pi"
    df[rate_column] = np.nan
    df[hz_column] = np.nan

    if "G_OP" not in df.columns:
        return df

    for _F_value, group in df.groupby("F", sort=False):
        group_sorted = group.sort_values("m")
        rate_by_m = dict(zip(group_sorted["m"], group_sorted["G_OP"]))
        for row_index, row in group_sorted.iterrows():
            m_value = row["m"]
            previous_m = m_value - 1.0
            if previous_m in rate_by_m:
                Gamma_OP = 0.5 * (
                    row["G_OP"] + rate_by_m[previous_m]
                )
                df.loc[row_index, rate_column] = Gamma_OP
                df.loc[row_index, hz_column] = Gamma_OP / (2.0 * np.pi)

    return df

