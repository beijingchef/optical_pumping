"""High-level construction of single- and dual-alkali model results."""

import numpy as np
import pandas as pd

from .angular_momentum import build_ground_states
from .constants import (
    ATOMS,
    alkali_vapor_density_cm3,
    cross_spin_exchange_rate_info,
    ground_zeeman_shifts_hz,
    self_spin_exchange_rate_info,
    upper_larmor_frequency_from_field_nT,
)
from .multi_species import (
    coupled_population_jacobian,
    cross_spin_exchange_adjacent_coherence_self_relaxation_rates,
    steady_state_two_species,
)
from .observables import (
    add_adjacent_optical_relaxation_columns,
    add_nu_m_column,
    add_population_difference_column,
    add_total_relaxation_columns,
    optical_Lambda_fractional_rates,
    population_by_F,
    steady_state_from_L,
    total_G_OP_by_ground_state,
)
from .optical_pumping import (
    build_optical_L,
    decompose_nu_LS_components,
    optical_rate_scale_from_intensity,
    total_nu_LS_from_diagnostics,
)
from .rf_response import (
    coupled_weak_rf_observable_susceptibilities,
    largest_abs_Dm_relaxation_reference,
    weak_rf_observable_susceptibility,
)
from .spin_exchange import (
    build_ER_matrix,
    build_spin_exchange_matrix,
    er_adjacent_coherence_self_relaxation_rates,
    er_population_fractional_relaxation_rates,
    spin_exchange_adjacent_coherence_self_relaxation_rates,
    spin_exchange_population_fractional_relaxation_rates,
    spin_exchange_population_jacobian,
    steady_state_with_spin_exchange,
)
from .spectroscopy import line_center_frequency_MHz


def resolve_alkali_densities(atom_A_name, atom_B_name, temperature_C, mode, ratio):
    """Resolve A/B number densities for the selected mixture convention."""
    density_A = alkali_vapor_density_cm3(atom_A_name, temperature_C)
    if atom_B_name in (None, "None") or atom_B_name == atom_A_name:
        return density_A, 0.0
    if mode == "Relative concentration":
        density_B = density_A * max(0.0, float(ratio))
    else:
        density_B = alkali_vapor_density_cm3(atom_B_name, temperature_C)
    return float(density_A), float(density_B)


def _nearest_line_and_detuning(atom, absolute_frequency_MHz):
    lines = ("D1", "D2")
    line = min(
        lines,
        key=lambda candidate: abs(
            float(absolute_frequency_MHz)
            - line_center_frequency_MHz(atom, candidate)
        ),
    )
    return line, float(absolute_frequency_MHz) - line_center_frequency_MHz(atom, line)


def build_species_linear_model(config, all_beams, common):
    """Build optical and ER generators for one species before spin exchange."""
    atom_name = config["atom_name"]
    atom = ATOMS[atom_name]
    ground_states = build_ground_states(atom)
    size = len(ground_states)
    optical = np.zeros((size, size), dtype=float)
    diagnostics = []
    species_beams = []
    light_shift_diagnostics = []

    for source_beam in all_beams:
        beam = dict(source_beam)
        line, detuning = _nearest_line_and_detuning(
            atom, beam["absolute_frequency_MHz"]
        )
        beam["line"] = line
        beam["detuning"] = detuning
        beam["q_axis"] = config["q_axis"]
        if beam["target_label"] != config["label"]:
            beam["selected_transition"] = None
            beam["transition_label"] = "off-target"
        species_beams.append(beam)

        if float(beam["intensity"]) <= 0.0:
            continue
        rate_scale = optical_rate_scale_from_intensity(
            atom=atom,
            line=line,
            intensity_uW_cm2=beam["intensity"],
            n2_pressure_torr=common["n2_pressure_torr"],
            temperature_C=common["temperature_C"],
            n2_width_MHz_per_torr=config["n2_coeffs"][line]["width"],
        )
        generator, info = build_optical_L(
            atom=atom,
            line=line,
            ground_states=ground_states,
            detuning_MHz=detuning,
            pump_rate_s=rate_scale,
            selected_transition=beam.get("selected_transition"),
            k_axis=beam["k_axis"],
            pol=beam["pol"],
            q_axis=config["q_axis"],
            n2_pressure_torr=common["n2_pressure_torr"],
            temperature_C=common["temperature_C"],
            n2_width_MHz_per_torr=config["n2_coeffs"][line]["width"],
            n2_shift_MHz_per_torr=config["n2_coeffs"][line]["shift"],
            normalize_to_selected_total=False,
        )
        optical += generator
        diagnostics.append((beam, info))
        # A pump remains in the optical-pumping generator for any physically
        # possible off-resonant absorption, but its AC-Stark shift is reported
        # only for the alkali species that the pump targets.  In particular,
        # the other species' beam geometry must not make this species' local
        # light-shift Hamiltonian appear non-diagonal and blank its table.
        if beam["target_label"] == config["label"]:
            light_shift_diagnostics.append((beam, info))

    er_map = build_ER_matrix(atom, ground_states)
    linear = optical + float(config["R_ER"]) * (er_map - np.eye(size))
    self_rate_info = self_spin_exchange_rate_info(
        atom_name,
        common["temperature_C"],
        config["density_cm3"],
    )
    return {
        **config,
        "atom": atom,
        "ground_states": ground_states,
        "size": size,
        "beams": species_beams,
        "diagnostics": diagnostics,
        "light_shift_diagnostics": light_shift_diagnostics,
        "L_optical": optical,
        "M_ER": er_map,
        "L_linear": linear,
        "self_rate_info": self_rate_info,
        "R_SE_self": (
            self_rate_info["rate_s"] if common["include_spin_exchange"] else 0.0
        ),
    }


def _finalize_species(
    model,
    population,
    self_map,
    cross_map,
    partner_electron,
    R_SE_cross,
    jacobian_local,
    common,
):
    atom = model["atom"]
    states = model["ground_states"]
    R_self = model["R_SE_self"]
    R_ER = float(model["R_ER"])

    nu_LS, light_shift_available = total_nu_LS_from_diagnostics(
        states,
        [beam for beam, _info in model["light_shift_diagnostics"]],
        model["light_shift_diagnostics"],
    )
    components = decompose_nu_LS_components(states, nu_LS)
    static_field_aligned = model["q_axis"] == common["static_field_axis"]
    field_parallel_nT = common["static_field_nT"] if static_field_aligned else 0.0
    bias_hz = upper_larmor_frequency_from_field_nT(
        model["atom_name"], field_parallel_nT
    )
    nu_B, bias_info = ground_zeeman_shifts_hz(
        model["atom_name"], atom, states, bias_hz
    )
    G_OP = total_G_OP_by_ground_state(states, model["diagnostics"])
    Lambda = optical_Lambda_fractional_rates(model["L_optical"], population, G_OP)
    G_ER = er_population_fractional_relaxation_rates(model["M_ER"], population, R_ER)
    Gamma_ER = er_adjacent_coherence_self_relaxation_rates(atom, states, R_ER)
    G_SE_self = spin_exchange_population_fractional_relaxation_rates(
        self_map, population, R_self
    )
    G_SE_cross = spin_exchange_population_fractional_relaxation_rates(
        cross_map, population, R_SE_cross
    )
    Gamma_SE_self = spin_exchange_adjacent_coherence_self_relaxation_rates(
        atom,
        states,
        population,
        model["electron_marginal"],
        R_self,
    )
    Gamma_SE_cross = cross_spin_exchange_adjacent_coherence_self_relaxation_rates(
        atom,
        states,
        partner_electron,
        R_SE_cross,
    )
    G_SE = G_SE_self + G_SE_cross
    Gamma_SE = Gamma_SE_self + Gamma_SE_cross

    df = pd.DataFrame({
        "F": [state["F"] for state in states],
        "m": [state["m"] for state in states],
        "population": population,
        "nu_VS": components["vector"],
        "nu_TS": components["tensor"],
        "nu_LS": nu_LS,
        "nu_B": nu_B,
        "Lambda": Lambda,
        "G_OP": G_OP,
        "G_ER": G_ER,
        "Gamma_ER": Gamma_ER,
        "G_SE_self": G_SE_self,
        "Gamma_SE_self": Gamma_SE_self,
        "G_SE_cross": G_SE_cross,
        "Gamma_SE_cross": Gamma_SE_cross,
        "G_SE": G_SE,
        "Gamma_SE": Gamma_SE,
    })
    df = add_population_difference_column(df)
    df = add_nu_m_column(df)
    df = add_adjacent_optical_relaxation_columns(df)
    df = add_total_relaxation_columns(df)

    upper_F = max(float(state["F"]) for state in states)
    frequencies = model["rf_frequencies_hz"]
    if light_shift_available:
        rf_amplitude, rf_in_phase, rf_quadrature, rf_info = (
            weak_rf_observable_susceptibility(
                frequencies_hz=frequencies,
                ground_states=states,
                populations=population,
                adjacent_transition_hz=df["nu_m"].to_numpy(dtype=float),
                gamma_op=df["Gamma_OP"].to_numpy(dtype=float),
                gamma_er=df["Gamma_ER"].to_numpy(dtype=float),
                gamma_se=df["Gamma_SE"].to_numpy(dtype=float),
                q_axis=model["q_axis"],
                rf_axis=model["rf_axis"],
                observable=model["rf_observable"],
                target_F=upper_F,
            )
        )
    else:
        rf_amplitude = np.full_like(frequencies, np.nan, dtype=float)
        rf_in_phase = np.full_like(frequencies, np.nan, dtype=float)
        rf_quadrature = np.full_like(frequencies, np.nan, dtype=float)
        rf_info = {"used_transitions": 0, "nonpositive_linewidths": 0}

    df_F = population_by_F(df)
    df["hyperfine_population"] = np.nan
    for _, row in df_F.iterrows():
        mask = (
            np.isclose(df["F"].to_numpy(dtype=float), float(row["F"]))
            & np.isclose(df["m"].to_numpy(dtype=float), 0.0)
        )
        df.loc[mask, "hyperfine_population"] = float(row["population"])

    model.update({
        "population": population,
        "self_map": self_map,
        "cross_map": cross_map,
        "R_SE_cross": float(R_SE_cross),
        "J_population": jacobian_local,
        "electron_marginal": model["electron_marginal"],
        "nu_LS": nu_LS,
        "light_shift_available": light_shift_available,
        "bias_larmor_hz": bias_hz,
        "static_field_aligned": static_field_aligned,
        "field_parallel_nT": field_parallel_nT,
        "bias_info": bias_info,
        "df_pop": df,
        "df_F": df_F,
        "rf_upper_F": upper_F,
        "rf_frequencies_hz": frequencies,
        "rf_amplitude": rf_amplitude,
        "rf_in_phase": rf_in_phase,
        "rf_quadrature": rf_quadrature,
        "rf_info": rf_info,
        "rf_relaxation_reference": largest_abs_Dm_relaxation_reference(
            df, target_F=upper_F
        ),
    })
    return model


def compute_alkali_system(species_A_config, species_B_config, all_beams, common):
    """Compute one active alkali or a coupled A/B system."""
    model_A = build_species_linear_model(species_A_config, all_beams, common)
    model_B = (
        build_species_linear_model(species_B_config, all_beams, common)
        if species_B_config is not None
        else None
    )

    if model_B is None:
        R_self = model_A["R_SE_self"]
        if R_self > 0.0:
            p_A, solve = steady_state_with_spin_exchange(
                model_A["L_linear"],
                model_A["atom"],
                model_A["ground_states"],
                R_self,
            )
            self_map = solve["M_SE"]
            electron_A = solve["electron_marginal"]
        else:
            p_A = steady_state_from_L(model_A["L_linear"])
            self_map, electron_A = build_spin_exchange_matrix(
                model_A["atom"], model_A["ground_states"], p_A
            )
            solve = {
                "iterations": 0,
                "converged": True,
                "residual": float(np.max(np.abs(model_A["L_linear"] @ p_A))),
                "mirror_symmetry_enforced": False,
            }
        model_A["electron_marginal"] = electron_A
        cross_map = np.eye(model_A["size"])
        J_A = model_A["L_linear"] + spin_exchange_population_jacobian(
            model_A["atom"], model_A["ground_states"], p_A, R_self
        )
        result_A = _finalize_species(
            model_A,
            p_A,
            self_map,
            cross_map,
            np.array([0.5, 0.5]),
            0.0,
            J_A,
            common,
        )
        return {"A": result_A, "B": None, "J_coupled": J_A, "solve": solve}

    cross_A_info = cross_spin_exchange_rate_info(
        model_A["atom_name"],
        model_B["atom_name"],
        common["temperature_C"],
        model_B["density_cm3"],
    )
    cross_B_info = cross_spin_exchange_rate_info(
        model_B["atom_name"],
        model_A["atom_name"],
        common["temperature_C"],
        model_A["density_cm3"],
    )
    R_A_cross = cross_A_info["rate_s"] if common["include_spin_exchange"] else 0.0
    R_B_cross = cross_B_info["rate_s"] if common["include_spin_exchange"] else 0.0
    p_A, p_B, solve = steady_state_two_species(
        model_A["L_linear"],
        model_A["atom"],
        model_A["ground_states"],
        model_A["R_SE_self"],
        R_A_cross,
        model_B["L_linear"],
        model_B["atom"],
        model_B["ground_states"],
        model_B["R_SE_self"],
        R_B_cross,
    )
    model_A["electron_marginal"] = solve["A"]["electron_marginal"]
    model_B["electron_marginal"] = solve["B"]["electron_marginal"]
    full_J, blocks = coupled_population_jacobian(
        model_A["L_linear"],
        model_A["atom"],
        model_A["ground_states"],
        p_A,
        model_A["R_SE_self"],
        R_A_cross,
        model_B["L_linear"],
        model_B["atom"],
        model_B["ground_states"],
        p_B,
        model_B["R_SE_self"],
        R_B_cross,
    )
    result_A = _finalize_species(
        model_A,
        p_A,
        solve["A"]["self_map"],
        solve["A"]["cross_map"],
        solve["B"]["electron_marginal"],
        R_A_cross,
        blocks["AA"],
        common,
    )
    result_B = _finalize_species(
        model_B,
        p_B,
        solve["B"]["self_map"],
        solve["B"]["cross_map"],
        solve["A"]["electron_marginal"],
        R_B_cross,
        blocks["BB"],
        common,
    )
    result_A["cross_rate_info"] = cross_A_info
    result_B["cross_rate_info"] = cross_B_info
    coupled_rf = coupled_weak_rf_observable_susceptibilities(result_A, result_B)
    if coupled_rf is not None:
        for label, result in (("A", result_A), ("B", result_B)):
            amplitude, in_phase, quadrature, info = coupled_rf[label]
            result["rf_amplitude"] = amplitude
            result["rf_in_phase"] = in_phase
            result["rf_quadrature"] = quadrature
            result["rf_info"] = info
    return {
        "A": result_A,
        "B": result_B,
        "J_coupled": full_J,
        "J_blocks": blocks,
        "solve": solve,
    }
