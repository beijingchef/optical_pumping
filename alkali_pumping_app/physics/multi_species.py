"""Coupled population model for two different alkali species."""

import numpy as np

from .observables import steady_state_from_L
from .spin_exchange import (
    build_spin_exchange_matrix,
    coupled_basis_amplitudes,
    electron_marginal_from_population,
    generator_has_m_inversion_symmetry,
    hyperfine_uncoupled_probabilities,
    spin_exchange_population_jacobian,
    symmetrize_populations_under_m_inversion,
)


def build_cross_spin_exchange_matrix(
    target_atom,
    target_ground_states,
    partner_electron_marginal,
):
    """Build the target-species collision map for a fixed partner marginal."""
    probabilities, _mI, _mS = hyperfine_uncoupled_probabilities(
        target_atom, target_ground_states
    )
    electron = np.asarray(partner_electron_marginal, dtype=float)
    electron = np.clip(electron, 0.0, None)
    if electron.sum() > 0.0:
        electron /= electron.sum()
    else:
        electron = np.array([0.5, 0.5], dtype=float)

    nuclear_by_state = probabilities.sum(axis=2)
    matrix = np.einsum(
        "bi,s,ais->ab",
        nuclear_by_state,
        electron,
        probabilities,
    )
    column_sums = matrix.sum(axis=0)
    populated = column_sums > 0.0
    matrix[:, populated] /= column_sums[populated]
    return matrix


def cross_spin_exchange_population_jacobian(
    target_atom,
    target_ground_states,
    target_population,
    partner_atom,
    partner_ground_states,
    partner_population,
    rate_s,
):
    """Derivative of the target population flow with respect to the partner."""
    target_probabilities, _target_mI, _target_mS = (
        hyperfine_uncoupled_probabilities(target_atom, target_ground_states)
    )
    partner_probabilities, _partner_mI, _partner_mS = (
        hyperfine_uncoupled_probabilities(partner_atom, partner_ground_states)
    )

    target_p = np.asarray(target_population, dtype=float)
    target_p = np.clip(target_p, 0.0, None)
    target_p = (
        target_p / target_p.sum()
        if target_p.sum() > 0.0
        else np.ones(len(target_ground_states)) / len(target_ground_states)
    )
    partner_p = np.asarray(partner_population, dtype=float)
    partner_p = np.clip(partner_p, 0.0, None)
    partner_p = (
        partner_p / partner_p.sum()
        if partner_p.sum() > 0.0
        else np.ones(len(partner_ground_states)) / len(partner_ground_states)
    )

    target_nuclear_by_state = target_probabilities.sum(axis=2)
    target_nuclear = np.einsum("b,bi->i", target_p, target_nuclear_by_state)
    partner_electron_by_state = partner_probabilities.sum(axis=1)
    partner_electron = np.einsum(
        "b,bs->s", partner_p, partner_electron_by_state
    )
    electron_derivative = partner_electron_by_state - partner_electron[None, :]

    derivative = np.einsum(
        "ais,i,bs->ab",
        target_probabilities,
        target_nuclear,
        electron_derivative,
    )
    return float(rate_s) * derivative


def cross_spin_exchange_adjacent_coherence_self_relaxation_rates(
    target_atom,
    target_ground_states,
    partner_electron_marginal,
    rate_s,
):
    """Return local target-coherence loss caused by unlike-atom collisions."""
    amplitudes = coupled_basis_amplitudes(target_atom, target_ground_states)
    electron = np.asarray(partner_electron_marginal, dtype=float)
    electron = np.clip(electron, 0.0, None)
    if electron.sum() > 0.0:
        electron /= electron.sum()
    else:
        electron = np.array([0.5, 0.5], dtype=float)
    rho_electron = np.diag(electron)

    rates = np.full(len(target_ground_states), np.nan, dtype=float)
    state_index = {
        (float(state["F"]), float(state["m"])): idx
        for idx, state in enumerate(target_ground_states)
    }
    for a_idx, state in enumerate(target_ground_states):
        b_idx = state_index.get((float(state["F"]), float(state["m"] - 1.0)))
        if b_idx is None:
            continue
        C_a = amplitudes[a_idx]
        C_b = amplitudes[b_idx]
        delta_rho_nuclear = np.einsum("is,js->ij", C_a, C_b)
        retention = np.einsum(
            "is,ij,st,jt->",
            C_a,
            delta_rho_nuclear,
            rho_electron,
            C_b,
        )
        rates[a_idx] = float(rate_s) * (1.0 - float(retention))
    return rates


def steady_state_two_species(
    linear_A,
    atom_A,
    states_A,
    self_rate_A,
    cross_rate_A_from_B,
    linear_B,
    atom_B,
    states_B,
    self_rate_B,
    cross_rate_B_from_A,
    max_iter=300,
    tol=1e-12,
):
    """Solve two coupled nonlinear mean-field population equations."""
    linear_A = np.asarray(linear_A, dtype=float)
    linear_B = np.asarray(linear_B, dtype=float)
    p_A = steady_state_from_L(linear_A)
    p_B = steady_state_from_L(linear_B)
    symmetric_A = generator_has_m_inversion_symmetry(linear_A, states_A)
    symmetric_B = generator_has_m_inversion_symmetry(linear_B, states_B)
    enforce_symmetry = symmetric_A and symmetric_B
    if enforce_symmetry:
        p_A = symmetrize_populations_under_m_inversion(p_A, states_A)
        p_B = symmetrize_populations_under_m_inversion(p_B, states_B)

    damping = 0.65
    converged = False
    residual = np.inf
    identity_A = np.eye(len(states_A))
    identity_B = np.eye(len(states_B))

    for iteration in range(1, max_iter + 1):
        self_map_A, electron_A = build_spin_exchange_matrix(atom_A, states_A, p_A)
        self_map_B, electron_B = build_spin_exchange_matrix(atom_B, states_B, p_B)
        cross_map_A = build_cross_spin_exchange_matrix(atom_A, states_A, electron_B)
        cross_map_B = build_cross_spin_exchange_matrix(atom_B, states_B, electron_A)
        effective_A = (
            linear_A
            + float(self_rate_A) * (self_map_A - identity_A)
            + float(cross_rate_A_from_B) * (cross_map_A - identity_A)
        )
        effective_B = (
            linear_B
            + float(self_rate_B) * (self_map_B - identity_B)
            + float(cross_rate_B_from_A) * (cross_map_B - identity_B)
        )
        next_A = steady_state_from_L(effective_A)
        next_B = steady_state_from_L(effective_B)
        if enforce_symmetry:
            next_A = symmetrize_populations_under_m_inversion(next_A, states_A)
            next_B = symmetrize_populations_under_m_inversion(next_B, states_B)

        difference = max(
            float(np.max(np.abs(next_A - p_A))),
            float(np.max(np.abs(next_B - p_B))),
        )
        p_A = damping * next_A + (1.0 - damping) * p_A
        p_B = damping * next_B + (1.0 - damping) * p_B
        p_A /= p_A.sum()
        p_B /= p_B.sum()

        self_map_A, electron_A = build_spin_exchange_matrix(atom_A, states_A, p_A)
        self_map_B, electron_B = build_spin_exchange_matrix(atom_B, states_B, p_B)
        cross_map_A = build_cross_spin_exchange_matrix(atom_A, states_A, electron_B)
        cross_map_B = build_cross_spin_exchange_matrix(atom_B, states_B, electron_A)
        residual_A = (
            linear_A @ p_A
            + float(self_rate_A) * (self_map_A @ p_A - p_A)
            + float(cross_rate_A_from_B) * (cross_map_A @ p_A - p_A)
        )
        residual_B = (
            linear_B @ p_B
            + float(self_rate_B) * (self_map_B @ p_B - p_B)
            + float(cross_rate_B_from_A) * (cross_map_B @ p_B - p_B)
        )
        residual = max(
            float(np.max(np.abs(residual_A))),
            float(np.max(np.abs(residual_B))),
        )
        rate_scale = max(
            1.0,
            float(self_rate_A),
            float(self_rate_B),
            float(cross_rate_A_from_B),
            float(cross_rate_B_from_A),
        )
        if difference < tol and residual < max(1e-10, tol * rate_scale):
            converged = True
            break
    else:
        iteration = max_iter

    self_map_A, electron_A = build_spin_exchange_matrix(atom_A, states_A, p_A)
    self_map_B, electron_B = build_spin_exchange_matrix(atom_B, states_B, p_B)
    cross_map_A = build_cross_spin_exchange_matrix(atom_A, states_A, electron_B)
    cross_map_B = build_cross_spin_exchange_matrix(atom_B, states_B, electron_A)
    effective_A = (
        linear_A
        + float(self_rate_A) * (self_map_A - identity_A)
        + float(cross_rate_A_from_B) * (cross_map_A - identity_A)
    )
    effective_B = (
        linear_B
        + float(self_rate_B) * (self_map_B - identity_B)
        + float(cross_rate_B_from_A) * (cross_map_B - identity_B)
    )
    return p_A, p_B, {
        "A": {
            "self_map": self_map_A,
            "cross_map": cross_map_A,
            "electron_marginal": electron_A,
            "L_effective": effective_A,
        },
        "B": {
            "self_map": self_map_B,
            "cross_map": cross_map_B,
            "electron_marginal": electron_B,
            "L_effective": effective_B,
        },
        "iterations": iteration,
        "converged": converged,
        "residual": residual,
        "mirror_symmetry_enforced": enforce_symmetry,
    }


def coupled_population_jacobian(
    linear_A,
    atom_A,
    states_A,
    p_A,
    self_rate_A,
    cross_rate_A_from_B,
    linear_B,
    atom_B,
    states_B,
    p_B,
    self_rate_B,
    cross_rate_B_from_A,
):
    """Return the full A/B block population Jacobian and its four blocks."""
    electron_A = electron_marginal_from_population(atom_A, states_A, p_A)
    electron_B = electron_marginal_from_population(atom_B, states_B, p_B)
    cross_map_A = build_cross_spin_exchange_matrix(atom_A, states_A, electron_B)
    cross_map_B = build_cross_spin_exchange_matrix(atom_B, states_B, electron_A)
    J_AA = (
        np.asarray(linear_A, dtype=float)
        + spin_exchange_population_jacobian(
            atom_A, states_A, p_A, self_rate_A
        )
        + float(cross_rate_A_from_B) * (cross_map_A - np.eye(len(states_A)))
    )
    J_BB = (
        np.asarray(linear_B, dtype=float)
        + spin_exchange_population_jacobian(
            atom_B, states_B, p_B, self_rate_B
        )
        + float(cross_rate_B_from_A) * (cross_map_B - np.eye(len(states_B)))
    )
    J_AB = cross_spin_exchange_population_jacobian(
        atom_A,
        states_A,
        p_A,
        atom_B,
        states_B,
        p_B,
        cross_rate_A_from_B,
    )
    J_BA = cross_spin_exchange_population_jacobian(
        atom_B,
        states_B,
        p_B,
        atom_A,
        states_A,
        p_A,
        cross_rate_B_from_A,
    )
    full = np.block([[J_AA, J_AB], [J_BA, J_BB]])
    return full, {"AA": J_AA, "AB": J_AB, "BA": J_BA, "BB": J_BB}
