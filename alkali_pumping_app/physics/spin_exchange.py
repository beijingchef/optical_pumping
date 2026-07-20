"""Population-only mean-field spin-exchange models."""

from math import sqrt

import numpy as np
from sympy import S
from sympy.physics.wigner import wigner_3j

from .angular_momentum import m_values
from .observables import steady_state_from_L

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


def er_population_fractional_relaxation_rates(M_ER, p_steady, R_ER):
    """Return the signed ER fractional rate of each steady-state population.

    For a diagonal steady-state density matrix,

        (d p_a / dt)_ER = R_ER [(M_ER p)_a - p_a].

    The reported rate is

        G_a^(ER) = -(d p_a / dt)_ER / p_a.

    Positive values mean ER removes population from the state; negative values
    mean ER replenishes it. States with numerically zero population are blank.
    """
    p = np.asarray(p_steady, dtype=float)
    er_derivative = float(R_ER) * (np.asarray(M_ER, dtype=float) @ p - p)
    rates = np.full_like(p, np.nan, dtype=float)
    populated = p > 1e-15
    rates[populated] = -er_derivative[populated] / p[populated]
    return rates


def er_adjacent_coherence_self_relaxation_rates(atom, ground_states, R_ER):
    """Return ER self-decay rates for infinitesimal rho_(m,m-1) coherences.

    The electron-randomization channel is

        E_ER(rho) = Tr_S(rho) tensor I_S/2.

    For each adjacent coherence |a><b| within one F manifold, this function
    evaluates its self-retention coefficient

        k_ab = <a| E_ER(|a><b|) |b>,

    and reports R_ER (1-k_ab). ER can also couple coherences with the same
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
        rates[a_idx] = float(R_ER) * (1.0 - float(self_retention))

    return rates


# ============================================================
# 7b. Population-only spin-exchange matrix
# ============================================================

# Use the shared angular-momentum basis transformations below this boundary.
from .angular_momentum import (  # noqa: E402
    coupled_basis_amplitudes,
    hyperfine_uncoupled_probabilities,
)

__all__ = [
    "electron_marginal_from_population",
    "build_spin_exchange_matrix",
    "spin_exchange_population_jacobian",
    "spin_exchange_population_fractional_relaxation_rates",
    "spin_exchange_adjacent_coherence_self_relaxation_rates",
    "mirror_state_indices",
    "symmetrize_populations_under_m_inversion",
    "generator_has_m_inversion_symmetry",
    "steady_state_with_spin_exchange",
]

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


def spin_exchange_population_jacobian(atom, ground_states, p_reference, R_SE):
    """Return the full linearized SE population Jacobian about p_reference.

    The nonlinear population collision target is

        T[p] = populations of Tr_S(rho[p]) tensor Tr_I(rho[p]),

    after projection onto the displayed |F,m> populations.  Both the source
    nuclear marginal and the ensemble electron marginal depend on p.  Hence
    the derivative contains two retention/repopulation contributions:

        delta T = (delta rho_I) tensor rho_S
                + rho_I tensor (delta rho_S),

    plus the normalization correction required when the population vector is
    extended away from unit trace.  The returned matrix J_SE satisfies

        delta(dot p)_SE = J_SE delta p

    and includes the ensemble-electron feedback term.  Its columns sum to zero,
    so it preserves the trace for arbitrary infinitesimal perturbations.
    """
    probs, _mI_list, _mS_list = hyperfine_uncoupled_probabilities(
        atom, ground_states
    )

    p = np.asarray(p_reference, dtype=float)
    p = np.clip(p, 0.0, None)
    trace = float(p.sum())
    if trace > 0:
        p = p / trace
    else:
        p = np.ones(len(ground_states), dtype=float) / len(ground_states)

    # Marginal probabilities of the reference diagonal density matrix.
    nuclear_by_state = probs.sum(axis=2)  # [source state, m_I]
    electron_by_state = probs.sum(axis=1)  # [source state, m_S]
    nuclear_marginal = np.einsum("b,bi->i", p, nuclear_by_state)
    electron_marginal = np.einsum("b,bs->s", p, electron_by_state)

    # Reference postcollision target population T[p].
    target_population = np.einsum(
        "i,s,ais->a",
        nuclear_marginal,
        electron_marginal,
        probs,
    )

    # dT/dp from changing the source nuclear marginal.
    nuclear_feedback = np.einsum(
        "ais,bi,s->ab",
        probs,
        nuclear_by_state,
        electron_marginal,
    )

    # dT/dp from changing the ensemble electron marginal.
    electron_feedback = np.einsum(
        "ais,i,bs->ab",
        probs,
        nuclear_marginal,
        electron_by_state,
    )

    # The implemented nonlinear map is homogeneous and trace preserving:
    # T[p] = rho_I[p] tensor rho_S[p] / Tr(p).  At Tr(p)=1, differentiating
    # 1/Tr(p) contributes -T[p] to every source-state column.
    target_derivative = (
        nuclear_feedback
        + electron_feedback
        - target_population[:, None]
    )

    J_SE = float(R_SE) * (
        target_derivative - np.eye(len(ground_states), dtype=float)
    )
    return J_SE


def spin_exchange_population_fractional_relaxation_rates(M_SE, p_steady, R_SE):
    """Return the signed SE fractional rate of each steady-state population.

    For the final self-consistent mean-field spin-exchange map,

        (d p_a / dt)_SE = R_SE [(M_SE p)_a - p_a].

    The reported rate is

        G_a^(SE) = -(d p_a / dt)_SE / p_a.

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
    p_steady,
    electron_marginal,
    R_SE,
):
    """Return linearized SE self-decay rates for adjacent coherences.

    The population solver uses the nonlinear mean-field collision map

        E_SE(rho) = Tr_S(rho) tensor Tr_I(rho).

    Linearizing this map around the final diagonal steady state rho_0 gives

        delta E_SE = Tr_S(delta rho) tensor rho_S^(0)
                   + rho_I^(0) tensor Tr_I(delta rho).

    The first term retains coherence through the source atom's nuclear marginal.
    The second is the ensemble-electron feedback term: the collision partner's
    electron marginal changes when the infinitesimal coherence carries electron
    orientation. For each adjacent operator |a><b| within one F manifold, the
    function projects both first-order terms back onto the same coherence and
    reports

        Gamma_ab^(SE) = R_SE [1 - k_ab^(nuclear) - k_ab^(electron)].

    Spin exchange can also couple different coherences with the same Delta m.
    Therefore this is the local/self coefficient appropriate to a well-resolved
    transition, not a general Liouvillian eigenmode decay rate.
    """
    amplitudes = coupled_basis_amplitudes(atom, ground_states)

    p = np.asarray(p_steady, dtype=float)
    p = np.clip(p, 0.0, None)
    if p.sum() > 0:
        p = p / p.sum()
    else:
        p = np.ones(len(ground_states), dtype=float) / len(ground_states)

    electron_marginal = np.asarray(electron_marginal, dtype=float)
    if electron_marginal.sum() > 0:
        electron_marginal = electron_marginal / electron_marginal.sum()
    else:
        electron_marginal = np.array([0.5, 0.5], dtype=float)

    # rho_0 is diagonal in |F,m>. Its nuclear and electron marginals are then
    # diagonal in |m_I> and |m_S>, respectively.
    probabilities = amplitudes**2
    nuclear_marginal = np.einsum("a,ais->i", p, probabilities)
    if nuclear_marginal.sum() > 0:
        nuclear_marginal = nuclear_marginal / nuclear_marginal.sum()

    rho_I_0 = np.diag(nuclear_marginal)
    rho_S_0 = np.diag(electron_marginal)

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

        # Marginals of the infinitesimal operator delta rho = |a><b|.
        delta_rho_I = np.einsum("is,js->ij", C_a, C_b)
        delta_rho_S = np.einsum("is,it->st", C_a, C_b)

        # 1) Nuclear-memory retention:
        #    Tr_S(delta rho) tensor rho_S^(0).
        nuclear_retention = np.einsum(
            "is,ij,st,jt->",
            C_a,
            delta_rho_I,
            rho_S_0,
            C_b,
        )

        # 2) Ensemble-electron feedback:
        #    rho_I^(0) tensor Tr_I(delta rho).
        electron_feedback = np.einsum(
            "is,ij,st,jt->",
            C_a,
            rho_I_0,
            delta_rho_S,
            C_b,
        )

        total_retention = float(nuclear_retention + electron_feedback)
        # Clip only tiny floating-point excursions outside the physical range.
        if -1e-12 < total_retention < 0.0:
            total_retention = 0.0
        if 1.0 < total_retention < 1.0 + 1e-12:
            total_retention = 1.0

        rates[a_idx] = float(R_SE) * (1.0 - total_retention)

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
