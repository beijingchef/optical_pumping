"""Streamlit entry point for Alkali Pumping v5.2.

The page body intentionally remains a direct script so Streamlit reruns it
from top to bottom on every interaction. Physics and reusable UI helpers live
in the alkali_pumping_app package.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from alkali_pumping_app.physics import *
from alkali_pumping_app.physics.optical_pumping import build_optical_L
from alkali_pumping_app.ui.captions import input_conditions_caption
from alkali_pumping_app.ui.atomic_settings import atomic_properties_dialog
from alkali_pumping_app.ui.conditions import *
from alkali_pumping_app.ui.exports import (
    dataframe_to_csv_bytes,
    weak_rf_export_dataframe,
)
from alkali_pumping_app.ui.rf_display import prepare_weak_rf_plot_values
from alkali_pumping_app.ui.tables import *
from alkali_pumping_app.version import DISPLAY_VERSION, __version__

# ============================================================

# Initialize every condition field before page configuration and widget creation.
# Later JSON loads still overwrite these values through apply_loaded_condition_dict().
for _key, _value in DEFAULT_STARTUP_CONDITION.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value
st.session_state.setdefault(
    "_last_atom_name_for_defaults",
    DEFAULT_STARTUP_CONDITION["atom_name"],
)

st.set_page_config(
    page_title=current_browser_title(),
    layout="wide",
    initial_sidebar_state="expanded",
)

title_column, settings_column = st.columns([0.84, 0.16], vertical_alignment="center")
with title_column:
    st.title(
        f"Alkali pumping v{DISPLAY_VERSION}: three pumps + ER + SE + weak RF"
    )
with settings_column:
    if st.button("⚙️ Settings", width="stretch"):
        atomic_properties_dialog()

# ============================================================
# Sidebar: all input condition values
# ============================================================

with st.sidebar:
    # Defaults used before widgets are rendered.
    # Put defaults in st.session_state, then create widgets using only key=...,
    # so Streamlit does not warn about competing default and session-state values.
    if "show_allowed_only" not in st.session_state:
        st.session_state["show_allowed_only"] = True
    if "show_rate_matrices" not in st.session_state:
        st.session_state["show_rate_matrices"] = False
    if "n2_pressure_torr" not in st.session_state:
        st.session_state["n2_pressure_torr"] = 0.0
    if "temperature_C_for_table" not in st.session_state:
        st.session_state["temperature_C_for_table"] = 23.5
    if "gamma_ER" not in st.session_state:
        st.session_state["gamma_ER"] = 4.0
    if "include_spin_exchange" not in st.session_state:
        st.session_state["include_spin_exchange"] = True

    st.header("condition")
    if "condition_name" not in st.session_state:
        st.session_state["condition_name"] = "default-ps400"

    # Keep the controls visually at the top, but populate this placeholder only
    # after every sidebar widget has been instantiated. This ensures that the
    # downloaded JSON contains the complete, current condition.
    condition_controls_placeholder = st.empty()

    st.header("Atom / cell")
    atom_name = st.selectbox("Alkali atom", list(ATOMS.keys()), index=0, key="atom_name")
    cell_condition_col1, cell_condition_col2 = st.columns(2, gap="small")
    with cell_condition_col1:
        n2_pressure_torr = st.number_input(
            "N₂ pressure (Torr)",
            min_value=0.0,
            step=10.0,
            format="%.1f",
            key="n2_pressure_torr",
        )
    with cell_condition_col2:
        temperature_C = st.number_input(
            "Temperature (°C)",
            step=1.0,
            format="%.1f",
            key="temperature_C_for_table",
        )

    atom = ATOMS[atom_name]

    # If the atom is changed manually, initialize the N2 coefficients to that atom's defaults.
    # Loading from a JSON condition overwrites these keys before widgets are created.
    if st.session_state.get("_last_atom_name_for_defaults") != atom_name:
        st.session_state["D1_width"] = DEFAULT_N2_COEFFS[atom_name]["D1"]["width"]
        st.session_state["D2_width"] = DEFAULT_N2_COEFFS[atom_name]["D2"]["width"]
        st.session_state["D1_shift"] = DEFAULT_N2_COEFFS[atom_name]["D1"]["shift"]
        st.session_state["D2_shift"] = DEFAULT_N2_COEFFS[atom_name]["D2"]["shift"]
        st.session_state["_last_atom_name_for_defaults"] = atom_name

    se_rate_preview = spin_exchange_rate_info(atom_name, atom, temperature_C)

    cell_row_col1, cell_row_col2 = st.columns(2, gap="small")
    with cell_row_col1:
        include_spin_exchange = st.checkbox(
            "Include spin exchange",
            key="include_spin_exchange",
        )
        st.caption(f"R_SE={se_rate_preview['rate_s']:.3g} s⁻¹")
    with cell_row_col2:
        # Keep the legacy session-state key for compatibility with existing
        # condition files, while using R_ER as the physical rate symbol.
        R_ER = st.number_input(
            r"$R_{\mathrm{ER}}$ (s⁻¹)",
            min_value=0.0,
            step=1.0,
            format="%.1f",
            key="gamma_ER",
        )

    # The coefficients remain part of condition state and the physics model,
    # but are no longer editable in the sidebar.
    D1_width = float(st.session_state["D1_width"])
    D2_width = float(st.session_state["D2_width"])
    D1_shift = float(st.session_state["D1_shift"])
    D2_shift = float(st.session_state["D2_shift"])

    axis_col, bias_col = st.columns(2, gap="small")
    with axis_col:
        q_axis = st.selectbox(
            "Quantization axis",
            ["z", "x", "y"],
            index=0,
            key="q_axis",
        )
    with bias_col:
        bias_larmor_hz = st.number_input(
            r"$B_q$ Larmor frequency (Hz)",
            step=1.0,
            format="%g",
            key="bias_larmor_hz",
            help=(
                "Signed adjacent-level Larmor frequency of the upper ground "
                "hyperfine manifold. The lower manifold uses its calculated "
                "hyperfine-g-factor ratio and therefore generally has the "
                "opposite Zeeman slope. In this model the static field is "
                "parallel to the quantization axis and enters only as a "
                "diagonal linear-Zeeman energy shift. It changes the displayed "
                "state shifts nu^B and the total adjacent-transition frequency "
                "nu_m, so it shifts the RF resonances and changes the response "
                "at a fixed RF frequency. It does not change the zero-RF "
                "steady-state populations, D_m, optical-pumping rates, ER or SE "
                "population flows, or the adjacent-coherence relaxation rates. "
                "The model does not include Zeeman corrections to optical "
                "detunings, nonlinear Zeeman shifts, Breit-Rabi mixing, or "
                "field-dependent relaxation."
            ),
        )

    n2_coeffs = {
        "D1": {"width": D1_width, "shift": D1_shift},
        "D2": {"width": D2_width, "shift": D2_shift},
    }

    show_allowed_only = st.session_state["show_allowed_only"]
    show_rate_matrices = st.session_state["show_rate_matrices"]

    def beam_config_ui(beam_number, default_line_index=0, default_Fg=None, default_Fe=None, default_rate=10.0):
        st.header(f"Beam {beam_number}")
        det_rel_key = f"det_rel{beam_number}"
        rate_reference_key = f"rate_reference{beam_number}"
        rate_key = f"rate{beam_number}"
        if det_rel_key not in st.session_state:
            st.session_state[det_rel_key] = 0.0
        if rate_key not in st.session_state:
            st.session_state[rate_key] = float(default_rate)

        line = st.selectbox("Reference Line", ["D1", "D2"], index=default_line_index, key=f"line{beam_number}")

        transition_options = transition_choice_labels(
            atom, line, n2_pressure_torr, n2_coeffs, allowed_only=show_allowed_only
        )
        transition_key = f"transition{beam_number}"
        if transition_options and st.session_state.get(transition_key) not in transition_options:
            preferred_transition = None
            if default_Fg is not None and default_Fe is not None:
                preferred_transition = default_transition_label(
                    atom, line, n2_pressure_torr, n2_coeffs, default_Fg, default_Fe, allowed_only=show_allowed_only
                )
            st.session_state[transition_key] = preferred_transition if preferred_transition in transition_options else transition_options[0]
        transition = st.selectbox(
            "hpf-transition",
            transition_options,
            key=transition_key,
        )
        det_rel = st.number_input(
            "Detuning (MHz)",
            step=10.0,
            format="%g",
            key=det_rel_key,
            help="Laser detuning is defined relative to this pressure-shifted selected hyperfine transition.",
        )
        rate_reference = st.selectbox(
            "Pump-rate reference",
            ["At resonance", "At detuning"],
            key=rate_reference_key,
            help=(
                "Choose whether the entered total rate describes the selected "
                "transition at the entered detuning or at its resonance center."
            ),
        )
        rate = st.number_input(
            "Pump rate (s⁻¹)",
            min_value=0.0,
            step=10.0,
            format="%.0f",
            key=rate_key,
            help=(
                "Sum of the selected-transition depopulation rates over all "
                "ground and excited Zeeman sublevels, using the reference above."
            ),
        )
        k_axis = st.selectbox(
            "beam direction",
            ["z", "x", "y"],
            index=0,
            key=f"k{beam_number}",
        )
        pol_options = allowed_polarizations(k_axis)
        pol_key = f"pol{beam_number}"
        if st.session_state.get(pol_key) not in pol_options:
            st.session_state[pol_key] = pol_options[0]
        pol = st.selectbox(
            "Polarization",
            pol_options,
            key=pol_key,
        )
        return line, transition, det_rel, rate_reference, rate, k_axis, pol

    bcol1, bcol2, bcol3 = st.columns(3, gap="small")
    with bcol1:
        line1, transition1, det_rel1, rate_reference1, rate1, k1, pol1 = beam_config_ui(1, default_Fg=1, default_Fe=2)
    with bcol2:
        line2, transition2, det_rel2, rate_reference2, rate2, k2, pol2 = beam_config_ui(2, default_Fg=2, default_Fe=2)
    with bcol3:
        line3, transition3, det_rel3, rate_reference3, rate3, k3, pol3 = beam_config_ui(3, default_Fg=2, default_Fe=2, default_rate=0.0)

    st.divider()
    st.header("Display")
    show_allowed_only = st.checkbox(
        "Only show allowed hyperfine transitions",
        key="show_allowed_only",
    )
    show_rate_matrices = st.checkbox(
        "Show rate matrices",
        key="show_rate_matrices",
    )

    # Populate the top placeholder after all condition widgets exist.
    # The condition name widget is evaluated before the download payload is
    # serialized, so the JSON always contains the value currently visible here.
    with condition_controls_placeholder.container():
        load_col, save_col, con_name_col = st.columns([0.3,0.2,0.5], gap="small")

        with load_col:
            st.file_uploader(
                "Load condition",
                type=["json"],
                key="condition_file_uploader",
                help="Choose an alkali_pumping v5 JSON condition file.",
                label_visibility="collapsed",
                on_change=load_condition_callback,
            )
        with save_col:
            save_button_placeholder = st.empty()
        with con_name_col:
            condition_name = st.text_input(
                "condition name",
                key="condition_name",
                help=(
                    "This value is saved inside the JSON conditions, used as the "
                    "suggested filename, and appended to the browser title after "
                    "the condition is loaded or saved."
                ),
            )

        condition_save_name = clean_condition_name(condition_name)
        condition_values = current_condition_values(
            condition_name=condition_save_name
        )
        condition_payload = build_condition_payload(condition_values)
        condition_json = json.dumps(condition_payload, indent=2)

        save_button_placeholder.download_button(
            "Save",
            data=condition_json,
            file_name=f"{condition_save_name}.json",
            mime="application/json",
            key="save_condition_button",
            width="stretch",
        )

        if st.session_state.get("_condition_load_message"):
            st.success(st.session_state.pop("_condition_load_message"))
        if st.session_state.get("_condition_load_error"):
            st.error(
                "Could not load condition file: "
                + st.session_state.pop("_condition_load_error")
            )


# RF display controls are rendered next to and below the response plot. Read
# their current session-state values here so the model can be built before the
# main output layout is rendered.
normalize_rf_frequency_bounds(prefer="lower")
rf_axis = st.session_state.get("rf_axis", "x")
rf_observable = st.session_state.get("rf_observable", "Fx")
rf_frequency_lower_hz = float(st.session_state.get("rf_frequency_lower_hz", 0.0))
rf_frequency_upper_hz = float(st.session_state.get("rf_frequency_upper_hz", 50.0))
rf_show_amplitude = bool(st.session_state.get("rf_show_amplitude", True))
rf_show_in_phase = bool(st.session_state.get("rf_show_in_phase", False))
rf_show_quadrature = bool(st.session_state.get("rf_show_quadrature", False))
rf_relaxation_normalized = bool(
    st.session_state.get("rf_relaxation_normalized", False)
)
rf_density_factor = bool(st.session_state.get("rf_density_factor", False))

# ============================================================
# Build model
# ============================================================

det1_abs, selected_transition1 = absolute_detuning_from_transition_choice(
    atom=atom,
    line=line1,
    transition_label=transition1,
    relative_detuning_MHz=det_rel1,
    n2_pressure_torr=n2_pressure_torr,
    n2_coeffs=n2_coeffs,
    allowed_only=show_allowed_only,
)

det2_abs, selected_transition2 = absolute_detuning_from_transition_choice(
    atom=atom,
    line=line2,
    transition_label=transition2,
    relative_detuning_MHz=det_rel2,
    n2_pressure_torr=n2_pressure_torr,
    n2_coeffs=n2_coeffs,
    allowed_only=show_allowed_only,
)

det3_abs, selected_transition3 = absolute_detuning_from_transition_choice(
    atom=atom,
    line=line3,
    transition_label=transition3,
    relative_detuning_MHz=det_rel3,
    n2_pressure_torr=n2_pressure_torr,
    n2_coeffs=n2_coeffs,
    allowed_only=show_allowed_only,
)

beam_inputs = [
    {
        "name": "Beam 1",
        "line": line1,
        "transition_label": transition1,
        "selected_transition": selected_transition1,
        "detuning_relative": det_rel1,
        "detuning": det1_abs,
        "rate": rate1,
        "rate_reference": rate_reference1,
        "k_axis": k1,
        "pol": pol1,
        "q_axis": q_axis,
    },
    {
        "name": "Beam 2",
        "line": line2,
        "transition_label": transition2,
        "selected_transition": selected_transition2,
        "detuning_relative": det_rel2,
        "detuning": det2_abs,
        "rate": rate2,
        "rate_reference": rate_reference2,
        "k_axis": k2,
        "pol": pol2,
        "q_axis": q_axis,
    },
    {
        "name": "Beam 3",
        "line": line3,
        "transition_label": transition3,
        "selected_transition": selected_transition3,
        "detuning_relative": det_rel3,
        "detuning": det3_abs,
        "rate": rate3,
        "rate_reference": rate_reference3,
        "k_axis": k3,
        "pol": pol3,
        "q_axis": q_axis,
    },
]

ground_states = build_ground_states(atom)
N = len(ground_states)

L_total = np.zeros((N, N), dtype=float)
diagnostics = []

for b in beam_inputs:
    if b["rate"] > 0:
        line = b["line"]

        Lb, info = build_optical_L(
            atom=atom,
            line=line,
            ground_states=ground_states,
            detuning_MHz=b["detuning"],
            pump_rate_s=b["rate"],
            selected_transition=b.get("selected_transition"),
            k_axis=b["k_axis"],
            pol=b["pol"],
            q_axis=q_axis,
            n2_pressure_torr=n2_pressure_torr,
            temperature_C=temperature_C,
            n2_width_MHz_per_torr=n2_coeffs[line]["width"],
            n2_shift_MHz_per_torr=n2_coeffs[line]["shift"],
            normalize_to_selected_total=True,
            reference_at_resonance_center=(
                b["rate_reference"] == "At resonance"
            ),
        )

        L_total += Lb
        diagnostics.append((b, info))

# Preserve the optical-only generator before ER and SE terms are added.
L_optical = L_total.copy()

M_ER = build_ER_matrix(atom, ground_states)
L_linear = L_optical + R_ER * (M_ER - np.eye(N))

se_rate_info = spin_exchange_rate_info(atom_name, atom, temperature_C)
R_SE_inferred = se_rate_info["rate_s"]
R_SE = R_SE_inferred if include_spin_exchange else 0.0

if R_SE > 0:
    p_ss, se_solver_info = steady_state_with_spin_exchange(
        L_linear, atom, ground_states, R_SE
    )
    L_total = se_solver_info["L_effective"]
else:
    p_ss = steady_state_from_L(L_linear)
    M_SE, electron_marginal = build_spin_exchange_matrix(atom, ground_states, p_ss)
    se_solver_info = {
        "M_SE": M_SE,
        "L_effective": L_linear.copy(),
        "electron_marginal": electron_marginal,
        "iterations": 0,
        "converged": True,
        "residual": float(np.max(np.abs(L_linear @ p_ss))),
        "mirror_symmetry_enforced": False,
    }
    L_total = L_linear

nu_LS, light_shift_available = total_nu_LS_from_diagnostics(
    ground_states,
    beam_inputs,
    diagnostics,
)
light_shift_components = decompose_nu_LS_components(ground_states, nu_LS)
nu_VS = light_shift_components["vector"]
nu_TS = light_shift_components["tensor"]
nu_B, bias_zeeman_info = ground_zeeman_shifts_hz(
    atom_name,
    atom,
    ground_states,
    bias_larmor_hz,
)
G_OP_by_state = total_G_OP_by_ground_state(
    ground_states,
    diagnostics,
)
Lambda_by_state = optical_Lambda_fractional_rates(
    L_optical,
    p_ss,
    G_OP_by_state,
)

G_ER = er_population_fractional_relaxation_rates(
    M_ER, p_ss, R_ER
)
Gamma_ER = er_adjacent_coherence_self_relaxation_rates(
    atom, ground_states, R_ER
)

G_SE = spin_exchange_population_fractional_relaxation_rates(
    se_solver_info["M_SE"], p_ss, R_SE
)
Gamma_SE = (
    spin_exchange_adjacent_coherence_self_relaxation_rates(
        atom,
        ground_states,
        p_ss,
        se_solver_info["electron_marginal"],
        R_SE,
    )
)

# Full small-signal population Jacobian.  Unlike the frozen map used inside
# the fixed-point iteration, this includes the response of the ensemble electron
# marginal to a population perturbation.
J_SE_population = spin_exchange_population_jacobian(
    atom,
    ground_states,
    p_ss,
    R_SE,
)
J_total_population = L_linear + J_SE_population

df_pop = pd.DataFrame({
    "F": [g["F"] for g in ground_states],
    "m": [g["m"] for g in ground_states],
    "population": p_ss,
    "nu_VS": nu_VS,
    "nu_TS": nu_TS,
    "nu_LS": nu_LS,
    "nu_B": nu_B,
    "Lambda": Lambda_by_state,
    "G_OP": G_OP_by_state,
    "G_ER": G_ER,
    "Gamma_ER": Gamma_ER,
    "G_SE": G_SE,
    "Gamma_SE": Gamma_SE,
})
df_pop = add_population_difference_column(df_pop)
df_pop = add_nu_m_column(df_pop)
df_pop = add_adjacent_optical_relaxation_columns(df_pop)

# The RF observable is restricted to the upper ground hyperfine manifold.
rf_upper_F = max(float(state["F"]) for state in ground_states)

if np.isclose(rf_frequency_lower_hz, rf_frequency_upper_hz):
    rf_frequencies_hz = np.array([rf_frequency_lower_hz], dtype=float)
else:
    rf_frequencies_hz = np.linspace(
        rf_frequency_lower_hz,
        rf_frequency_upper_hz,
        1201,
    )

if light_shift_available:
    (
        rf_susceptibility_amplitude,
        rf_susceptibility_in_phase,
        rf_susceptibility_quadrature,
        rf_response_info,
    ) = weak_rf_observable_susceptibility(
        frequencies_hz=rf_frequencies_hz,
        ground_states=ground_states,
        populations=p_ss,
        adjacent_transition_hz=df_pop["nu_m"].to_numpy(dtype=float),
        gamma_op=df_pop["Gamma_OP"].to_numpy(dtype=float),
        gamma_er=df_pop["Gamma_ER"].to_numpy(dtype=float),
        gamma_se=df_pop["Gamma_SE"].to_numpy(dtype=float),
        q_axis=q_axis,
        rf_axis=rf_axis,
        observable=rf_observable,
        target_F=rf_upper_F,
    )
else:
    rf_susceptibility_amplitude = np.full_like(rf_frequencies_hz, np.nan, dtype=float)
    rf_susceptibility_in_phase = np.full_like(rf_frequencies_hz, np.nan, dtype=float)
    rf_susceptibility_quadrature = np.full_like(rf_frequencies_hz, np.nan, dtype=float)
    rf_response_info = {
        "used_transitions": 0,
        "nonpositive_linewidths": 0,
    }

rf_relaxation_reference = largest_abs_Dm_relaxation_reference(df_pop, target_F=rf_upper_F)
rf_relaxation_normalization_applied = False
normalization_gamma = None
if rf_relaxation_normalized and rf_relaxation_reference.get("available", False):
    # Relaxation normalization removes the 1/Gamma response scale by
    # multiplying the susceptibility by the selected local coherence rate.
    normalization_gamma = rf_relaxation_reference["Gamma_m"]
    rf_relaxation_normalization_applied = True

# The density option converts the per-atom susceptibility into a response per
# unit volume using the same saturated-vapor density used for spin exchange.
rf_density_factor_applied = bool(rf_density_factor)
rf_density_cm3 = (
    float(se_rate_info["density_cm3"])
    if rf_density_factor_applied
    else None
)

# Display convention: flip both signed lock-in components by a common minus
# sign. Optional relaxation and density factors multiply all three curves.
(
    rf_plot_amplitude,
    rf_plot_in_phase,
    rf_plot_quadrature,
) = prepare_weak_rf_plot_values(
    rf_susceptibility_amplitude,
    rf_susceptibility_in_phase,
    rf_susceptibility_quadrature,
    relaxation_gamma_s_inv=normalization_gamma,
    density_cm3=rf_density_cm3,
)

df_F = population_by_F(df_pop)

# Show the total population P_F of each hyperfine manifold only on its m=0 row
# in the Zeeman sublevel properties table.
df_pop["hyperfine_population"] = np.nan
for _, f_row in df_F.iterrows():
    manifold_mask = (
        np.isclose(df_pop["F"].to_numpy(dtype=float), float(f_row["F"]))
        & np.isclose(df_pop["m"].to_numpy(dtype=float), 0.0)
    )
    df_pop.loc[manifold_mask, "hyperfine_population"] = float(f_row["population"])

df_trans = hyperfine_transition_table(
    atom=atom,
    n2_pressure_torr=n2_pressure_torr,
    n2_coeffs=n2_coeffs,
    allowed_only=show_allowed_only,
    pump_beams=beam_inputs,
    temperature_C=temperature_C,
)

labels = [g["label"] for g in ground_states]

df_pop_display = df_pop.rename(columns={
    "hyperfine_population": "P_F",
    "population": "Pₘ",
    "population_difference": "Dₘ",
    "nu_VS": "ν^{VS} (Hz)",
    "nu_TS": "ν^{TS} (Hz)",
    "nu_LS": "ν^{LS} (Hz)",
    "nu_B": "ν^{B} (Hz)",
    "nu_m": "ν_m (Hz)",
    "Lambda": "Λ (s⁻¹)",
    "G_OP": "G^{OP} (s^-1)",
    "Gamma_OP": "Γ^{OP} (s^-1)",
    "Gamma_OP_over_2pi": "Γ^{OP}/2π (Hz)",
    "G_ER": "G^{ER} (s^-1)",
    "Gamma_ER": "Γ^{ER} (s^-1)",
    "G_SE": "G^{SE} (s^-1)",
    "Gamma_SE": "Γ^{SE} (s^-1)",
})
# Place G^ER and G^SE immediately before Gamma^ER.
df_pop_display = df_pop_display[[
    "F",
    "m",
    "P_F",
    "Pₘ",
    "Dₘ",
    "ν^{VS} (Hz)",
    "ν^{TS} (Hz)",
    "ν^{LS} (Hz)",
    "ν^{B} (Hz)",
    "ν_m (Hz)",
    "Λ (s⁻¹)",
    "G^{OP} (s^-1)",
    "Γ^{OP} (s^-1)",
    "Γ^{OP}/2π (Hz)",
    "G^{ER} (s^-1)",
    "G^{SE} (s^-1)",
    "Γ^{ER} (s^-1)",
    "Γ^{SE} (s^-1)",
]]
# Display the upper-F manifold first and order m from +F to -F
# within each manifold.
df_pop_display = df_pop_display.sort_values(
    by=["F", "m"],
    ascending=[False, False],
    kind="stable",
).reset_index(drop=True)

zeeman_csv = dataframe_to_csv_bytes(df_pop_display)
rf_normalization_gamma = (
    float(rf_relaxation_reference["Gamma_m"])
    if rf_relaxation_normalization_applied
    else None
)
df_rf_export = weak_rf_export_dataframe(
    frequencies_hz=rf_frequencies_hz,
    susceptibility_amplitude=rf_susceptibility_amplitude,
    susceptibility_in_phase=rf_susceptibility_in_phase,
    susceptibility_quadrature=rf_susceptibility_quadrature,
    plotted_amplitude=rf_plot_amplitude,
    plotted_in_phase=rf_plot_in_phase,
    plotted_quadrature=rf_plot_quadrature,
    relaxation_normalized=rf_relaxation_normalization_applied,
    normalization_gamma_s_inv=rf_normalization_gamma,
    density_factored=rf_density_factor_applied,
    density_cm3=rf_density_cm3,
)
rf_csv = dataframe_to_csv_bytes(df_rf_export)


# ============================================================
# Main compact layout
# ============================================================

def compact_section_title(text):
    """Render a compact title at about half the size of st.header."""
    st.markdown(
        f"<div style='text-align: center; font-size:1.25rem; font-weight:600; line-height:1.25; "
        f"margin:0.25rem 0 0.45rem 0;'>{text}</div>",
        unsafe_allow_html=True,
    )

left, right = st.columns([0.62, 1.63], gap="small")

with left:
    compact_section_title(f"{atom_name} ground-state populations")

    # Separate the two ground-state hyperfine manifolds.  The manifold with
    # the larger hyperfine energy is displayed in the upper panel.
    manifold_energies = {}
    for state in ground_states:
        manifold_energies.setdefault(float(state["F"]), float(state["E"]))

    manifolds_by_energy = sorted(
        manifold_energies, key=lambda F_value: manifold_energies[F_value]
    )
    lower_F = manifolds_by_energy[0]
    upper_F = manifolds_by_energy[-1]

    fig, (ax_upper, ax_lower) = plt.subplots(
        2, 1, figsize=(4.6, 4.5), sharey=True
    )

    for ax, F_value, panel_name in [
        (ax_upper, upper_F, "Upper hyperfine level"),
        (ax_lower, lower_F, "Lower hyperfine level"),
    ]:
        indices = [
            index
            for index, state in enumerate(ground_states)
            if np.isclose(float(state["F"]), F_value)
        ]
        m_labels = [f"{ground_states[index]['m']:g}" for index in indices]
        populations = [p_ss[index] for index in indices]

        ax.bar(m_labels, populations)
        ax.set_ylabel("Population",fontsize=12)
        ax.set_title(f"{panel_name}: F={F_value:g}", fontsize=12, pad=3)

    population_axis_max = max(0.01, 1.08 * float(np.max(p_ss)))
    ax_upper.set_ylim(0.0, population_axis_max)
    ax_lower.set_xlim(ax_upper.get_xlim()[0]-1, ax_upper.get_xlim()[1]-1)
    ax_lower.set_ylim(0.0, population_axis_max)
    ax_lower.set_xlabel(rf"$m$ along {q_axis}", fontsize=12)

    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    summary_m = expectation_m(ground_states, p_ss)
    summary_m2 = expectation_m2(ground_states, p_ss)

    st.markdown(
        f"""
        <style>
        .population-summary-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 0rem;
            font-size: 0.9rem;
            line-height: 1.35;
        }}
        .population-summary-item {{
            display: inline-flex;
            align-items: baseline;
            gap: 0.25rem;
            white-space: nowrap;
        }}
        .population-summary-label {{
            font-weight: 600;
        }}
        .population-summary-value {{
            font-family: inherit;
            font-size: inherit;
            font-weight: 400;
            font-variant-numeric: tabular-nums;
        }}
        </style>
        <div class="population-summary-row">
            <div class="population-summary-item">
                <span class="population-summary-label">〈m〉 =</span>
                <span class="population-summary-value">{summary_m:.4f}</span>
            </div>
            <div class="population-summary-item">
                <span class="population-summary-label">〈m²〉 =</span>
                <span class="population-summary-value">{summary_m2:.4f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    title, download, tip = st.columns([0.70, 0.22, 0.08], gap="small")
    with title:
        compact_section_title("Zeeman sublevel properties")
    with download:
        st.download_button(
            "Download CSV",
            data=zeeman_csv,
            file_name=f"{condition_save_name}_zeeman-sublevels.csv",
            mime="text/csv; charset=utf-8",
            key="download_zeeman_csv",
            help="Download the full-precision values shown in the table.",
            width="stretch",
        )
    with tip:
        with st.popover("❓"):
            st.markdown(
                r"""
                $\small D_m=P_m-P_{m-1}$ is the population difference between adjacent Zeeman sublevels.  
                $\small \nu^{\mathrm{VS}}$ and $\small \nu^{\mathrm{TS}}$ are the vector and tensor contributions to the total light shift $\small \nu^{\mathrm{LS}}$; its scalar contribution is not shown separately.  
                $\small \nu_m=[\nu^{\mathrm{LS}}_{m}+\nu^{B}_{m}]-[\nu^{\mathrm{LS}}_{m-1}+\nu^{B}_{m-1}]$ is the total adjacent-sublevel resonance frequency.  
                $\small \nu^{B}_{F,m}=m(g_F/g_{F_+})\nu_{B,+}$ is the static-field Zeeman shift, where the entered $\nu_{B,+}$ is the signed upper-manifold Larmor frequency.  
                $\small G^{\mathrm{ER}}_{m}$ is the signed net fractional ER rate of population of sublevel m; positive means loss.  
                $\small G^{\mathrm{SE}}_{m}=-(\dot P_m)_{\mathrm{SE}}/P_m$ is the exact signed net fractional SE population flow evaluated with the full nonlinear map; positive means loss. It is not a small-signal eigenmode decay rate.  
                $\small \Lambda_{m}$ is the total repopulation rate into $\small \lvert F,m\rangle$ divided by its steady-state population.  
                $\small G^{\mathrm{OP}}_{m}$ is the depopulation rate from $\small\lvert F,m\rangle$, summed over excited states and active pumps.  
                $\small \Gamma^{\mathrm{OP}}_{m}=(G^{\mathrm{OP}}_m+G^{\mathrm{OP}}_{m-1})/2$ is the pump-induced decay rate of adjacent coherence $\small\rho_{m,m-1}$  
                $\small \Gamma^{\mathrm{ER}}_{m}$ is the ER-induced self-decay rate of the local adjacent coherence $\small\rho_{m,m-1}$.  
                $\small \Gamma^{\mathrm{SE}}_{m}$ is the linearized SE-induced self-decay rate of the well-resolved local adjacent coherence $\small \rho_{m,m-1}$, including ensemble-electron feedback.
                """
            )
    st.markdown(
        render_zeeman_properties_table_html(df_pop_display),
        unsafe_allow_html=True,
    )
    if light_shift_available:
        st.caption(r"$\nu^{\mathrm{VS}}$, $\nu^{\mathrm{TS}}$, and $\nu^{\mathrm{LS}}$ are shown because all active pump-beam light-shift Hamiltonians commute with the selected quantization-axis spin component.")
    else:
        st.caption(r"$\nu^{\mathrm{VS}}$, $\nu^{\mathrm{TS}}$, and $\nu^{\mathrm{LS}}$ are blank because at least one active beam has multiple spherical polarization components relative to the quantization axis, so the light-shift Hamiltonian may not commute with the selected spin component.")
rf_title_left, rf_title, rf_download = st.columns(
    [0.16, 0.68, 0.16], gap="small"
)
with rf_title:
    compact_section_title(
        f"Upper-hyperfine weak-RF susceptibility (F={rf_upper_F:g})"
    )
with rf_download:
    st.download_button(
        "Download CSV",
        data=rf_csv,
        file_name=f"{condition_save_name}_weak-rf-susceptibility.csv",
        mime="text/csv; charset=utf-8",
        key="download_weak_rf_csv",
        help=(
            "Download frequency samples plus the raw and currently plotted "
            "amplitude, in-phase, and quadrature values."
        ),
        disabled=(
            not light_shift_available
            or rf_response_info.get("used_transitions", 0) == 0
        ),
        width="stretch",
    )

# RF response region: a narrow control column and a wide plot column.
# The plot shows amplitude, in-phase, and quadrature susceptibility components.
rf_control_col, rf_plot_col = st.columns([0.15, 0.85], gap="small")

with rf_control_col:
    rf_axis = st.selectbox(
        "RF axis",
        ["x", "y", "z"],
        key="rf_axis",
        help="Laboratory axis of the linearly polarized RF magnetic field.",
    )
    rf_observable = st.selectbox(
        "Observable",
        ["Fx", "Fy", "Fz"],
        key="rf_observable",
        format_func=lambda value: {"Fx": "F_x", "Fy": "F_y", "Fz": "F_z"}[value],
        help=f"Laboratory-frame spin component of the upper hyperfine manifold F={rf_upper_F:g}.",
    )
    rf_lower_col, rf_upper_col = st.columns(2, gap="xxsmall")
    with rf_lower_col:
        rf_frequency_lower_hz = st.number_input(
            "Lower (Hz)",
            min_value=0.0,
            step=1.0,
            format="%g",
            key="rf_frequency_lower_hz",
            on_change=normalize_rf_frequency_bounds,
            args=("lower",),
        )
    with rf_upper_col:
        rf_frequency_upper_hz = st.number_input(
            "Upper (Hz)",
            min_value=0.0,
            step=1.0,
            format="%g",
            key="rf_frequency_upper_hz",
            on_change=normalize_rf_frequency_bounds,
            args=("upper",),
        )

    rf_show_amplitude = st.checkbox(
        "Amplitude",
        key="rf_show_amplitude",
        help="Show the nonnegative RF susceptibility amplitude.",
    )
    rf_show_in_phase = st.checkbox(
        "In phase",
        key="rf_show_in_phase",
        help="Show the signed in-phase RF susceptibility component.",
    )
    rf_show_quadrature = st.checkbox(
        "Quadrature",
        key="rf_show_quadrature",
        help="Show the signed quadrature RF susceptibility component.",
    )
    rf_relaxation_normalized = st.checkbox(
        "Relaxation normalized",
        key="rf_relaxation_normalized",
        help=(
            "Multiply the plotted susceptibility by the total adjacent-coherence "
            f"relaxation rate Gamma_m for the F={rf_upper_F:g} transition with the largest |D_m|."
        ),
    )
    rf_density_factor = st.checkbox(
        "Density factor",
        key="rf_density_factor",
        help=(
            "Multiply every plotted RF susceptibility component by the "
            "calculated alkali vapor number density n(T) in cm⁻³."
        ),
    )

with rf_plot_col:
    if not light_shift_available:
        st.warning(
            "The RF response is unavailable because at least one active optical "
            "field produces a non-diagonal light-shift Hamiltonian in the selected "
            "quantization basis."
        )
    elif rf_response_info.get("used_transitions", 0) == 0:
        if lab_axis_in_local_frame(q_axis, rf_axis) == "z":
            st.info(
                f"The laboratory-{rf_axis} RF field is longitudinal for the selected "
                "quantization axis, so it does not drive adjacent Zeeman coherences "
                "to first order."
            )
        elif lab_axis_in_local_frame(q_axis, rf_observable[-1].lower()) == "z":
            st.info(
                f"⟨{rf_observable}⟩ is longitudinal in the selected quantization "
                "frame and has no first-order weak-drive response from adjacent "
                "coherences."
            )
        else:
            st.info("No valid adjacent Zeeman transitions were available for the RF response.")
    elif not any((rf_show_amplitude, rf_show_in_phase, rf_show_quadrature)):
        st.info("Select at least one RF curve: Amplitude, In phase, or Quadrature.")
    else:
        rf_fig, rf_ax = plt.subplots(figsize=(8.6, 5))
        if len(rf_frequencies_hz) == 1:
            if rf_show_amplitude:
                rf_ax.plot(
                    rf_frequencies_hz,
                    rf_plot_amplitude,
                    marker="o",
                    linestyle="none",
                    label="Amplitude",
                )
            if rf_show_in_phase:
                rf_ax.plot(
                    rf_frequencies_hz,
                    rf_plot_in_phase,
                    marker="s",
                    linestyle="none",
                    label="In phase",
                )
            if rf_show_quadrature:
                rf_ax.plot(
                    rf_frequencies_hz,
                    rf_plot_quadrature,
                    marker="^",
                    linestyle="none",
                    label="Quadrature",
                )
        else:
            if rf_show_amplitude:
                rf_ax.plot(
                    rf_frequencies_hz,
                    rf_plot_amplitude,
                    label="Amplitude",
                )
            if rf_show_in_phase:
                rf_ax.plot(
                    rf_frequencies_hz,
                    rf_plot_in_phase,
                    linestyle="--",
                    label="In phase",
                )
            if rf_show_quadrature:
                rf_ax.plot(
                    rf_frequencies_hz,
                    rf_plot_quadrature,
                    linestyle=":",
                    label="Quadrature",
                )
        rf_ax.axhline(0.0, linewidth=0.8, alpha=0.45)
        rf_ax.set_xlabel("RF frequency (Hz)")
        if rf_relaxation_normalization_applied and rf_density_factor_applied:
            rf_ax.set_ylabel(
                rf"$n(T)\Gamma_{{m_*}}\langle F_{{+,{rf_observable[-1].lower()}}}\rangle/\Omega_1$ "
                rf"($\hbar/\mathrm{{cm}}^3$)"
            )
        elif rf_relaxation_normalization_applied:
            rf_ax.set_ylabel(
                rf"$\Gamma_{{m_*}}\langle F_{{+,{rf_observable[-1].lower()}}}\rangle/\Omega_1$ "
                rf"($\hbar$/atom)"
            )
        elif rf_density_factor_applied:
            rf_ax.set_ylabel(
                rf"$n(T)\langle F_{{+,{rf_observable[-1].lower()}}}\rangle/\Omega_1$ "
                rf"($\hbar\,\mathrm{{s}}/\mathrm{{cm}}^3$)"
            )
        else:
            rf_ax.set_ylabel(
                rf"$\langle F_{{+,{rf_observable[-1].lower()}}}\rangle/\Omega_1$ "
                rf"($\hbar\,\mathrm{{s}}$/atom)"
            )
        rf_ax.set_title(rf"$F_+={rf_upper_F:g},\quad B_{{\mathrm{{rf}}}}\parallel {rf_axis}$")
        rf_ax.legend(loc="best", frameon=False)
        rf_ax.grid(True, alpha=0.25)
        rf_fig.subplots_adjust(
            left=0.105,
            right=0.995,
            bottom=0.205,
            top=0.895,
        )
        st.pyplot(rf_fig, width="stretch")

        input_caption_col, response_tip_col = st.columns(
            [0.92, 0.08], gap="small"
        )
        with input_caption_col:
            st.caption(
                input_conditions_caption(
                    atom_name=atom_name,
                    temperature_C=temperature_C,
                    n2_pressure_torr=n2_pressure_torr,
                    R_ER=R_ER,
                    include_spin_exchange=include_spin_exchange,
                    R_SE=R_SE_inferred,
                    q_axis=q_axis,
                    bias_larmor_hz=bias_larmor_hz,
                    beam_inputs=beam_inputs,
                ),
                text_alignment="center",
            )
            if (
                rf_relaxation_normalized
                and not rf_relaxation_normalization_applied
            ):
                st.warning(
                    "Relaxation normalization could not be applied: "
                    + rf_relaxation_reference.get("reason", "unknown reason")
                    + "."
                )

        # The default Streamlit popover is about 20 rem wide.  The RF-response
        # explanation contains several equations, so make only this popover 50%
        # wider (30 rem) while leaving the Zeeman-table popover unchanged.
        st.markdown(
            """
            <style>
            [data-baseweb="popover"]:has(.rf-response-popover-marker),
            [data-baseweb="popover"]:has(.rf-response-popover-marker) > div {
                width: 30rem !important;
                min-width: 30rem !important;
                max-width: min(90vw, 30rem) !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        with response_tip_col:
            with st.popover("❓"):
                st.markdown(
                    "<span class='rf-response-popover-marker'></span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    r"""
                    The calculated lock-in components use the RF drive
                    $\cos(\omega t)$ as their phase reference:
                    $X=2\operatorname{Re}\chi_+$ and
                    $Y=2\operatorname{Im}\chi_+$.

                    For display, the graph plots
                    $X_{\rm plot}=-X$ and $Y_{\rm plot}=-Y$. Multiplying both
                    signed components by $-1$ is a common $180^\circ$ phase or
                    detector-polarity change. It leaves the amplitude
                    $A=\sqrt{X^2+Y^2}$, resonance positions, linewidths, and all
                    physical conclusions unchanged. The sign flip is used only
                    to make the three-curve plot more compact.
                    """
                )

        if rf_response_info.get("nonpositive_linewidths", 0) > 0:
            st.warning(
                "At least one summed local coherence linewidth was nonpositive and "
                "was replaced by a small numerical floor in the response plot."
            )

with st.expander("Weak-RF response model", expanded=False):
    st.markdown(
        r"""
        The population graph and Zeeman table are the zero-RF steady state. The
        RF plot is a first-order response around that state for a linearly
        polarized field along the selected laboratory RF axis. Only the upper
        ground hyperfine manifold $F_+=I+1/2$ contributes to the plotted
        observable; the lower-manifold spin response is excluded.

        For each adjacent coherence $\rho_{m,m-1}$, the signed transition
        frequency and total local linewidth are
        """
    )
    st.latex(
        rf"\mathbf{{B}}_{{\rm rf}}=B_1\cos(\omega t)\,\hat{{\mathbf{{{rf_axis}}}}}"
    )
    st.latex(
        r"""
        \omega_m=2\pi\nu_m,
        \qquad
        \nu_m=\Delta\nu_m^{\rm LS}+\Delta\nu_m^B,
        \qquad
        \Gamma_m=\Gamma_m^{\rm OP}+\Gamma_m^{\rm ER}+\Gamma_m^{\rm SE}.
        """
    )
    st.markdown(
        r"""
        The calculation retains both $e^{-i\omega t}$ and $e^{+i\omega t}$
        components of the real linearly polarized field. Thus a transition with
        negative signed $\omega_m$ can contribute to a positive-frequency
        resonance. The code expands the density matrix as
        $\rho=\rho_0+\Omega_1\rho^{(1)}+O(\Omega_1^2)$ and solves directly
        for $\rho^{(1)}$. It therefore never assigns a finite numerical value
        to $\Omega_1$. If $\chi_+$ denotes the coefficient of
        $e^{-i\omega t}$, the real response is written as
        $\partial\langle F_{+,i}(t)\rangle/\partial\Omega_1
        =X(\omega)\cos\omega t+Y(\omega)\sin\omega t$, with
        $X=2\operatorname{Re}\chi_+$,
        $Y=2\operatorname{Im}\chi_+$, and amplitude
        $A=\sqrt{X^2+Y^2}=2|\chi_+|$. The graph displays the unchanged
        amplitude together with $X_{\rm plot}=-X$ and $Y_{\rm plot}=-Y$.
        This common sign reversal is a $180^\circ$ phase/polarity convention
        used only to make the multi-curve plot more compact; it does not change
        the response amplitude or resonance physics. The underlying convention
        takes the RF drive $\cos\omega t$ as the in-phase reference. The
        expectation value is taken with the upper-manifold block
        of the full density matrix, so it retains the actual upper-manifold
        population rather than being renormalized to unit population within $F_+$.
        When **Relaxation normalized** is selected, this susceptibility is
        multiplied by $\Gamma_{m_*}$, where $m_*$ is the adjacent transition
        within $F_+$ with the largest $|D_m|$, with $D_m=P_m-P_{m-1}$ and
        $\Gamma_{m_*}=\Gamma_{m_*}^{\rm OP}+\Gamma_{m_*}^{\rm ER}
        +\Gamma_{m_*}^{\rm SE}$.
        When **Density factor** is selected, every displayed component is also
        multiplied by the saturated alkali vapor number density $n(T)$ in
        cm$^{-3}$. This converts the plotted response from a per-atom quantity
        to a response per unit volume; it does not alter the steady-state or RF
        susceptibility calculation itself.
        The adjacent transition matrix elements retain their full factors
        $C_m=\sqrt{F(F+1)-m(m-1)}$.

        This is a well-resolved local-coherence approximation. It does not feed
        RF-induced populations or coherences back into the optical-pumping or
        spin-exchange steady-state solver.
        """
    )


with st.expander("Static bias-field convention", expanded=False):
    upper_F = bias_zeeman_info["upper_F"]
    st.markdown(
        rf"""
        The entered bias-field value is the signed adjacent-level Larmor
        frequency $\nu_{{B,+}}$ of the upper ground hyperfine manifold
        $F_+={upper_F:g}$, with the field directed along the selected
        quantization axis. The linear Zeeman shift used in the table is
        """
    )
    st.latex(
        r"""
        \nu^B_{F,m}
        =m\,\frac{g_F}{g_{F_+}}\,\nu_{B,+},
        \qquad
        \Delta\nu_m^B
        =\nu^B_{F,m}-\nu^B_{F,m-1}
        =\frac{g_F}{g_{F_+}}\,\nu_{B,+}.
        """
    )
    ratio_text = ", ".join(
        f"F={F:g}: g_F/g_(F+)={ratio:.8g}"
        for F, ratio in sorted(bias_zeeman_info["ratio_by_F"].items())
    )
    st.write("Hyperfine Zeeman-slope ratios: " + ratio_text + ".")
    st.markdown(
        r"""
        Because this field is parallel to the quantization axis, its Hamiltonian
        is diagonal in the displayed $|F,m\rangle$ basis. It therefore does not
        change the zero-RF population steady state, optical depopulation rates,
        or the listed relaxation rates in this population-only secular model.
        It does shift every adjacent-coherence resonance and therefore changes
        the RF susceptibility spectrum through the updated $\nu_m$. The small
        Zeeman corrections to optical transition detunings, and excited-state
        Zeeman shifts, are not included in the optical-pumping calculation.
        """
    )


# compact_section_title("D1/D2 hyperfine transition detunings")
# st.caption(
#     "Detunings are relative to the corresponding zero-pressure D1 or D2 fine-structure line center. "
#     "Absolute optical frequencies are shown in MHz."
# )
with st.expander("D1/D2 hyperfine transition detunings", expanded=False):
    st.caption(
        "Detunings are relative to the corresponding zero-pressure D1 or D2 fine-structure line center. "
        "Absolute optical frequencies are shown in MHz."
    )
    st.markdown(
        render_transition_table_html(df_trans),
        unsafe_allow_html=True,
    )

if show_rate_matrices:
    with st.expander("Linearized population Jacobian J", expanded=False):
        st.write(
            "Columns are source-state perturbations and rows are destination "
            "population derivatives. Near the calculated steady state, "
            "d(δp)/dt = J δp. The SE part is the derivative of the full "
            "nonlinear map and includes ensemble-electron feedback."
        )
        Jdf = pd.DataFrame(J_total_population, index=labels, columns=labels)
        st.dataframe(Jdf.style.format("{:.3e}"), width="stretch")

    with st.expander("Frozen SE collision map M_SE", expanded=False):
        st.write(
            "For the final steady-state electron marginal, one collision maps "
            "p → M_SE p. This frozen map is used in the fixed-point solver, but "
            "R_SE(M_SE−I) is not the full population Jacobian because M_SE itself "
            "changes when the ensemble electron marginal changes."
        )
        MSEdf = pd.DataFrame(
            se_solver_info["M_SE"], index=labels, columns=labels
        )
        st.dataframe(MSEdf.style.format("{:.4f}"), width="stretch")

    with st.expander("ER redistribution matrix M_ER", expanded=False):
        st.write("After one ER collision: p → M_ER p.")
        Mdf = pd.DataFrame(M_ER, index=labels, columns=labels)
        st.dataframe(Mdf.style.format("{:.4f}"), width="stretch")

with st.expander("Spin-exchange diagnostics", expanded=False):
    st.write(
        f"Include spin exchange: {include_spin_exchange}.  "
        f"Applied R_SE = {R_SE:.6g} s⁻¹ "
        f"(temperature-inferred {R_SE_inferred:.6g} s⁻¹)."
    )
    st.write(
        f"Alkali vapor pressure = {se_rate_info['pressure_torr']:.3e} Torr; "
        f"density = {se_rate_info['density_cm3']:.3e} cm⁻³; "
        f"mean relative speed = {se_rate_info['vrel_cm_s']:.3e} cm/s; "
        f"σ_SE = {se_rate_info['sigma_cm2']:.3e} cm²."
    )
    ps_minus, ps_plus = se_solver_info["electron_marginal"]
    st.write(
        f"Electron-spin marginal used by the mean-field SE map: "
        f"P(mS=-1/2) = {ps_minus:.6f}, P(mS=+1/2) = {ps_plus:.6f}."
    )
    st.write(
        f"SE fixed-point iterations = {se_solver_info['iterations']}; "
        f"converged = {se_solver_info['converged']}; "
        f"max residual = {se_solver_info['residual']:.3e} s⁻¹."
    )
    st.write(
        "m→−m symmetry enforced in SE solve: "
        f"{se_solver_info.get('mirror_symmetry_enforced', False)}."
    )
    se_map_column_error = float(
        np.max(np.abs(np.sum(se_solver_info["M_SE"], axis=0) - 1.0))
    )
    se_jacobian_trace_error = float(
        np.max(np.abs(np.sum(J_SE_population, axis=0)))
    )
    m_vector = np.array([state["m"] for state in ground_states], dtype=float)
    se_jacobian_m_error = float(
        np.max(np.abs(m_vector @ J_SE_population))
    )
    st.write(
        "SE numerical conservation checks: "
        f"max |Σ_a M_SE(a,b)−1| = {se_map_column_error:.3e}; "
        f"max |Σ_a J_SE(a,b)| = {se_jacobian_trace_error:.3e}; "
        f"max |mᵀJ_SE| = {se_jacobian_m_error:.3e}."
    )
    st.caption(
        "Spin exchange is included as a population-only mean-field collision map. "
        "It preserves the source atom's nuclear-spin marginal and replaces the "
        "electron spin by the ensemble electron-spin marginal, then projects back "
        "onto the displayed hyperfine populations. Coherences are not propagated; "
        "the displayed coherence rates are obtained by linearizing the full nonlinear "
        "mean-field map, including ensemble-electron feedback."
    )

with st.expander("Light-shift calculation", expanded=False):
    st.markdown(
        r"""
        The light-shift column is calculated only when every active pump beam has
        one spherical polarization component $q=-1,0,+1$ relative to the selected
        quantization axis. In that case the AC-Stark Hamiltonian is diagonal in
        the displayed $\lvert F,m\rangle$ basis and commutes with the spin
        component along the quantization axis.
        """
    )
    st.latex(
        r"""
        R_{F,m\rightarrow F',m'}
        \propto
        \operatorname{Re}\!\left[w(z)\right],
        \qquad
        \delta\omega_{F,m\rightarrow F',m'}
        \propto
        \frac{1}{2}\operatorname{Im}\!\left[w(z)\right]
        """
    )
    st.latex(
        r"""
        z=
        \frac{\Delta_{F,m\rightarrow F',m'}+i\Gamma_L/2}
        {\sigma_D\sqrt{2}},
        \qquad
        \sigma_D=
        \frac{\Delta\nu_{D,\mathrm{FWHM}}}
        {2\sqrt{2\ln 2}}
        """
    )
    st.markdown(
        r"""
        **Definitions**

        The polarization index $q=+1,0,-1$ denotes the $\sigma^+$, $\pi$, and
        $\sigma^-$ spherical components, respectively, relative to the selected
        quantization axis. 

        $R_{F,m\rightarrow F',m'}$ is the optical excitation or depopulation rate
        for the indicated Zeeman transition. The quantity
        $\delta\omega_{F,m\rightarrow F',m'}$ is that transition's AC-Stark shift
        in angular-frequency units.

        $w(z)$ is the complex Faddeeva function used to describe the Voigt line
        shape. Its real part gives the Doppler-averaged absorption profile, and
        its imaginary part gives the corresponding dispersive profile.

        $\Delta_{F,m\rightarrow F',m'}$ is the laser-frequency detuning from the
        pressure-shifted Zeeman transition $\lvert F,m\rangle\rightarrow
        \lvert F',m'\rangle$. A positive value means that the laser frequency is
        above the transition frequency. $\Gamma_L$ is the total Lorentzian full width at half maximum, including
        the natural linewidth and N$_2$ pressure broadening. The quantity
        $\sigma_D$ is the Gaussian standard deviation of the Doppler distribution,
        and $\Delta\nu_{D,\mathrm{FWHM}}$ is its Doppler full width at half maximum.

        The table quantity $\nu^{\mathrm{LS}}=\delta\omega/(2\pi)$ is the total
        light shift of a ground-state Zeeman sublevel in Hz after summing over all
        excited states and active beams. The table also shows the static-field
        shift $\nu^B_{F,m}$. The total adjacent-sublevel resonance frequency is
        $\nu_m=[\nu^{\mathrm{LS}}_m+\nu^B_m]-[\nu^{\mathrm{LS}}_{m-1}+\nu^B_{m-1}]$.
        Finally, $G^{\mathrm{OP}}$ is the total optical depopulation rate of the
        relevant ground-state sublevel, summed over excited states and active
        pump beams.
        
        In the zero-Doppler or far-wing limit, each isolated two-level
        contribution reduces to
        $\delta\omega=G^{\mathrm{OP}}\Delta/\Gamma_L$.
        This small shift is not fed back into the optical detunings.
 
        The total diagonal AC-Stark shift contains scalar, vector, and tensor
        contributions. Within one hyperfine manifold, their dependence on the
        Zeeman quantum number can be written schematically as
        $\nu_m^{\mathrm{LS}}=\nu^{(0)}+C_V\mathcal{P}_1m
        +C_T\mathcal{P}_2[3m^2-F(F+1)]$.  The coefficients $C_V$ and $C_T$ contain the optical intensity,
        detunings, atomic line strengths, and normalization factors. 
        
        The scalar term $\nu^{(0)}$ is independent of $m$ and therefore cancels from the adjacent-sublevel
        difference $\nu_m$.

        The vector contribution is
        $\nu_m^{(V)}=C_V\mathcal{P}_1m$. It is odd in $m$ and therefore acts
        like an effective magnetic field along the quantization axis. Here
        $\mathcal{P}_1=|\epsilon_{+1}|^2-|\epsilon_{-1}|^2$ measures the optical
        helicity relative to that axis, where $\epsilon_q$ is the normalized
        spherical electric-field amplitude. Thus $\mathcal{P}_1=+1$ for pure
        $\sigma^+$ light, $-1$ for pure $\sigma^-$ light, and $0$ for pure
        $\pi$ light. Its contribution to adjacent levels is independent of $m$:
        $\nu_m^{(V)}-\nu_{m-1}^{(V)}=C_V\mathcal{P}_1$.

        The tensor contribution is
        $\nu_m^{(T)}=C_T\mathcal{P}_2[3m^2-F(F+1)]$. It is even in $m$ and
        produces a quadratic, generally nonuniform spacing across the Zeeman
        manifold. With the normalization used here,
        $\mathcal{P}_2=(3|\epsilon_0|^2-1)/2$; hence $\mathcal{P}_2=1$ for pure
        $\pi$ light and $\mathcal{P}_2=-1/2$ for pure $\sigma^+$ or $\sigma^-$
        light. The adjacent-level tensor contribution is linear in the transition
        label $m$:
        $\nu_m^{(T)}-\nu_{m-1}^{(T)}=3C_T\mathcal{P}_2(2m-1)$.
        The tensor term is absent for $F<1$.

        The app does not assume the schematic polynomial form when calculating the table; it
        sums the state-resolved Clebsch-Gordan-weighted dispersive shifts over all
        excited hyperfine states and active beams. The formulas above explain the
        resulting characteristic linear and quadratic dependence on $m$.
        """
    )


# ============================================================
# Notes shown at bottom
# ============================================================



with st.expander("Model and sign convention"):
    st.write("The solved equation is")

    st.latex(
        r"""
        \frac{d\mathbf p}{dt}
        =
        \left[
        L_{\mathrm{op},1}
        +
        L_{\mathrm{op},2}
        +
        L_{\mathrm{op},3}
        +
        R_{\mathrm{ER}}(M_{\mathrm{ER}}-\mathbb I)
        \right]\mathbf p
        +
        R_{\mathrm{SE}}(T)
        \left[M_{\mathrm{SE}}[\mathbf p]-\mathbb I\right]\mathbf p .
        """
    )

    st.write("The spin-exchange rate is inferred from the alkali vapor density at the selected temperature:")

    st.latex(
        r"""
        R_{\mathrm{SE}}(T)
        =
        n(T)\,\sigma_{\mathrm{SE}}\,\bar v_{\mathrm{rel}},
        \qquad
        \bar v_{\mathrm{rel}}=\sqrt{\frac{16k_BT}{\pi m}} .
        """
    )

    st.write("For the zero-RF steady state, M_SE[p] is a nonlinear mean-field collision map. The population solver recomputes the ensemble electron marginal self-consistently, so population feedback is included. The RF plot then calculates adjacent coherences only to first order around that steady state; it does not propagate a finite-RF density matrix or pair correlations.")

    st.write("In the interface, each laser detuning is set relative to a selected pressure-shifted hyperfine transition. The entered total pump rate R_pump is the selected-transition absorption rate summed over all ground and excited Zeeman sublevels:")

    st.latex(
        r"""
        R_{\mathrm{pump}}
        =
        \sum_{m=-F_0}^{F_0}
        \sum_{m'=-F'_0}^{F'_0}
        R_{F_0,m\rightarrow F'_0,m'} .
        """
    )

    st.write("Here F0 and F0' are the ground and excited hyperfine levels of the selected reference transition. The pump-rate reference selector determines whether this total is evaluated at the entered relative detuning or at the resonance center (Δrel = 0). In the resonance-center mode, R_pump fixes the beam intensity at line center while the optical-pumping dynamics are still evaluated at the entered detuning. Nearby hyperfine transitions remain included after this normalization fixes the optical scale. The relative detuning is")

    st.latex(
        r"""
        \Delta_{\mathrm{rel}}
        =
        \nu_L - \nu_{F_0\rightarrow F'_0}(P_{N_2}).
        """
    )

    st.write("The app internally converts that to a detuning from the zero-pressure D-line center:")

    st.latex(
        r"""
        \Delta_L
        =
        \delta_{\mathrm{hfs}}(F_0,F'_0)
        +
        \beta_{N_2}P_{N_2}
        +
        \Delta_{\mathrm{rel}} .
        """
    )

    st.write("For any other hyperfine transition in the pumping-rate sum, the actual detuning is")

    st.latex(
        r"""
        \Delta_{F,F'}
        =
        \Delta_L
        -
        \left[
        \delta_{\mathrm{hfs}}(F,F')
        +
        \beta_{N_2}P_{N_2}
        \right].
        """
    )

    st.write("A negative βN₂ means the optical resonance shifts to lower frequency.")
