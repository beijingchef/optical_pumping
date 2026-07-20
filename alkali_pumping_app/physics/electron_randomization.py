"""Electron-randomization collision and relaxation models."""

import numpy as np

from .angular_momentum import coupled_basis_amplitudes


def build_ER_matrix(atom, ground_states):
    """Build the population map for electron-spin randomization."""
    probabilities = coupled_basis_amplitudes(atom, ground_states) ** 2
    nuclear_probabilities = probabilities.sum(axis=2)
    count = len(ground_states)
    matrix = np.zeros((count, count), dtype=float)

    for target in range(count):
        for source in range(count):
            matrix[target, source] = 0.5 * np.sum(
                nuclear_probabilities[source] * nuclear_probabilities[target]
            )

    column_sums = matrix.sum(axis=0)
    for column in range(count):
        if column_sums[column] > 0:
            matrix[:, column] /= column_sums[column]
    return matrix


def er_population_fractional_relaxation_rates(M_ER, p_steady, R_ER):
    """Return signed ER fractional rates for steady-state populations."""
    populations = np.asarray(p_steady, dtype=float)
    derivative = float(R_ER) * (
        np.asarray(M_ER, dtype=float) @ populations - populations
    )
    rates = np.full_like(populations, np.nan, dtype=float)
    populated = populations > 1e-15
    rates[populated] = -derivative[populated] / populations[populated]
    return rates


def er_adjacent_coherence_self_relaxation_rates(atom, ground_states, R_ER):
    """Return ER self-decay rates for adjacent within-manifold coherences."""
    amplitudes = coupled_basis_amplitudes(atom, ground_states)
    rates = np.full(len(ground_states), np.nan, dtype=float)
    state_index = {
        (float(state["F"]), float(state["m"])): index
        for index, state in enumerate(ground_states)
    }

    for target_index, target in enumerate(ground_states):
        source_index = state_index.get(
            (float(target["F"]), float(target["m"] - 1.0))
        )
        if source_index is None:
            continue

        target_amplitudes = amplitudes[target_index]
        source_amplitudes = amplitudes[source_index]
        nuclear_operator = np.einsum(
            "is,js->ij", target_amplitudes, source_amplitudes
        )
        retention = 0.5 * np.einsum(
            "is,ij,js->",
            target_amplitudes,
            nuclear_operator,
            source_amplitudes,
        )
        rates[target_index] = float(R_ER) * (1.0 - float(retention))
    return rates
