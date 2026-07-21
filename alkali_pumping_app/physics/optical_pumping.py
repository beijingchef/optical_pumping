"""Optical-pumping generator and light-shift calculations."""

import numpy as np

from .angular_momentum import build_excited_states, dipole_strength
from .polarization import spherical_weights_relative_to_quant_axis
from .spectroscopy import (
    complex_voigt_response_relative,
    doppler_fwhm_MHz,
    transition_shift_MHz,
)

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
    normalize_to_selected_total=True,
    reference_at_resonance_center=False,
    normalize_to_unpolarized=None,
):
    """
    Build optical-pumping generator L for one beam.

    detuning_MHz:
        Laser detuning from the zero-pressure D1 or D2 fine-structure center.

    pump_rate_s:
        If normalize_to_selected_total=True, this is the total pumping rate for
        the selected hyperfine transition Fg -> Fe: the absorption/depopulation
        rates are summed over every ground Zeeman sublevel m and every selected
        excited Zeeman sublevel m'. Nearby hyperfine transitions are still
        included in the subsequent dynamics after this selected-transition
        normalization sets the optical intensity scale. If
        reference_at_resonance_center=True, the same total rate is defined at
        the selected transition's resonance center instead of at the laser
        detuning used for the dynamics.

    normalize_to_unpolarized:
        Backward-compatible alias for normalize_to_selected_total. New callers
        should use normalize_to_selected_total.
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
    angular_strength_ge = np.zeros_like(R_ge)
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
            angular_strength_ge[gi, ei] = strength_sum
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
        reference_indices = np.ix_(
            reference_ground_indices, reference_excited_indices
        )
        if reference_at_resonance_center:
            resonance_profile = max(
                0.0,
                complex_voigt_response_relative(
                    0.0, lorentz_fwhm, doppler_fwhm
                ).real,
            )
            reference_raw_total_selected_transition = (
                angular_strength_ge[reference_indices].sum()
                * resonance_profile
            )
        else:
            reference_raw_total_selected_transition = R_ge[reference_indices].sum()
    else:
        reference_raw_total_selected_transition = 0.0

    if normalize_to_unpolarized is not None:
        normalize_to_selected_total = bool(normalize_to_unpolarized)

    if normalize_to_selected_total:
        scale = (
            pump_rate_s / reference_raw_total_selected_transition
            if reference_raw_total_selected_transition > 0
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
        "reference_raw_total_selected_transition": reference_raw_total_selected_transition,
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


def total_nu_LS_from_diagnostics(ground_states, beam_inputs, diagnostics):
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

    nu_LS = shift_angular / (2.0 * np.pi)
    return nu_LS, True


def decompose_nu_LS_components(ground_states, nu_LS):
    """Decompose each hyperfine manifold into scalar, vector, and tensor shifts.

    Within a manifold the fitted basis is

        nu(F,m) = nu_scalar + c_vector*m
                  + c_tensor*[3*m**2 - F*(F+1)].

    The returned vector and tensor arrays contain the complete state-dependent
    contributions, rather than only their fitted coefficients. Unavailable
    total shifts remain unavailable in every component.
    """
    total = np.asarray(nu_LS, dtype=float)
    state_count = len(ground_states)
    if total.shape != (state_count,):
        raise ValueError("nu_LS must contain one value per ground state.")

    scalar = np.full(state_count, np.nan, dtype=float)
    vector = np.full(state_count, np.nan, dtype=float)
    tensor = np.full(state_count, np.nan, dtype=float)
    residual = np.full(state_count, np.nan, dtype=float)

    manifold_values = sorted({float(state["F"]) for state in ground_states})
    for F in manifold_values:
        indices = np.array(
            [
                index
                for index, state in enumerate(ground_states)
                if np.isclose(float(state["F"]), F)
            ],
            dtype=int,
        )
        manifold_shift = total[indices]
        if not np.all(np.isfinite(manifold_shift)):
            continue

        m = np.array([float(ground_states[index]["m"]) for index in indices])
        scalar_basis = np.ones_like(m)
        vector_basis = m
        tensor_basis = 3.0 * m**2 - F * (F + 1.0)

        basis_columns = [scalar_basis]
        component_names = ["scalar"]
        if np.linalg.norm(vector_basis) > 1e-14:
            basis_columns.append(vector_basis)
            component_names.append("vector")
        if np.linalg.norm(tensor_basis) > 1e-14:
            basis_columns.append(tensor_basis)
            component_names.append("tensor")

        design = np.column_stack(basis_columns)
        coefficients, *_ = np.linalg.lstsq(design, manifold_shift, rcond=None)
        fitted = design @ coefficients
        coefficient_by_name = dict(zip(component_names, coefficients))

        scalar[indices] = coefficient_by_name["scalar"] * scalar_basis
        vector[indices] = coefficient_by_name.get("vector", 0.0) * vector_basis
        tensor[indices] = coefficient_by_name.get("tensor", 0.0) * tensor_basis
        residual[indices] = manifold_shift - fitted

    return {
        "scalar": scalar,
        "vector": vector,
        "tensor": tensor,
        "residual": residual,
    }


# ============================================================
# 7. Electron-randomization matrix
