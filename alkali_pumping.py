"""Streamlit entry point for the dual-alkali Alkali Pumping application."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from alkali_pumping_app.physics import *
from alkali_pumping_app.physics.optical_pumping import build_optical_L
from alkali_pumping_app.ui.atomic_settings import atomic_properties_dialog
from alkali_pumping_app.ui.conditions import *
from alkali_pumping_app.ui.exports import dataframe_to_csv_bytes, weak_rf_export_dataframe
from alkali_pumping_app.ui.rf_display import prepare_weak_rf_plot_values
from alkali_pumping_app.ui.tables import (
    render_transition_table_html,
    render_zeeman_properties_table_html,
)


for _key, _value in DEFAULT_STARTUP_CONDITION.items():
    st.session_state.setdefault(_key, _value)
for _key in ("q_axis_A", "q_axis_B", *RF_CONDITION_KEYS):
    # These condition keys were widget-bound before v6.1.6. Reassigning them
    # protects existing sessions from Streamlit's lazy-widget cleanup.
    st.session_state[_key] = st.session_state[_key]
st.session_state.setdefault(
    "_last_atom_names_for_defaults",
    {
        "A": DEFAULT_STARTUP_CONDITION["atom_A_name"],
        "B": DEFAULT_STARTUP_CONDITION["atom_B_name"],
    },
)

st.set_page_config(
    page_title=current_browser_title(),
    layout="wide",
    initial_sidebar_state="expanded",
)

_, settings_column = st.columns([0.84, 0.16], vertical_alignment="center")
with settings_column:
    if st.button("⚙️ Settings", width="stretch"):
        atomic_properties_dialog()


def _n2_coefficients(label):
    return {
        "D1": {
            "width": float(st.session_state[f"D1_width_{label}"]),
            "shift": float(st.session_state[f"D1_shift_{label}"]),
        },
        "D2": {
            "width": float(st.session_state[f"D2_width_{label}"]),
            "shift": float(st.session_state[f"D2_shift_{label}"]),
        },
    }


def _initialize_atom_coefficients(label, atom_name):
    """Apply atom defaults only when a real selector changes manually."""
    previous = st.session_state["_last_atom_names_for_defaults"].get(label)
    if atom_name == "None" or previous == atom_name:
        return
    for line in ("D1", "D2"):
        st.session_state[f"{line}_width_{label}"] = DEFAULT_N2_COEFFS[atom_name][line]["width"]
        st.session_state[f"{line}_shift_{label}"] = DEFAULT_N2_COEFFS[atom_name][line]["shift"]
    st.session_state["_last_atom_names_for_defaults"][label] = atom_name


def _migrate_legacy_beam_intensity(
    prefix, atom, n2_coeffs, line, transition, det_rel, q_axis_value
):
    legacy_inputs = st.session_state.get("_legacy_pump_inputs", {})
    legacy = legacy_inputs.get(prefix)
    if legacy is None:
        return
    relative_detuning = 0.0 if legacy.get("rate_reference") == "At resonance" else det_rel
    detuning, selected = absolute_detuning_from_transition_choice(
        atom=atom,
        line=line,
        transition_label=transition,
        relative_detuning_MHz=relative_detuning,
        n2_pressure_torr=n2_pressure_torr,
        n2_coeffs=n2_coeffs,
        allowed_only=show_allowed_only,
    )
    k_axis = st.session_state.get(f"k_{prefix}", "x")
    pol_options = allowed_polarizations(k_axis)
    pol = st.session_state.get(f"pol_{prefix}", pol_options[0])
    if pol not in pol_options:
        pol = pol_options[0]
    states = build_ground_states(atom)
    rate_scale = optical_rate_scale_from_intensity(
        atom=atom,
        line=line,
        intensity_uW_cm2=1.0,
        n2_pressure_torr=n2_pressure_torr,
        temperature_C=temperature_C,
        n2_width_MHz_per_torr=n2_coeffs[line]["width"],
    )
    _, info = build_optical_L(
        atom=atom,
        line=line,
        ground_states=states,
        detuning_MHz=detuning,
        pump_rate_s=rate_scale,
        selected_transition=selected,
        k_axis=k_axis,
        pol=pol,
        q_axis=q_axis_value,
        n2_pressure_torr=n2_pressure_torr,
        temperature_C=temperature_C,
        n2_width_MHz_per_torr=n2_coeffs[line]["width"],
        n2_shift_MHz_per_torr=n2_coeffs[line]["shift"],
        normalize_to_selected_total=False,
    )
    indices = np.ix_(info["reference_ground_indices"], info["reference_excited_indices"])
    rate_per_uW = float(info["R_ge"][indices].sum())
    st.session_state[f"intensity_{prefix}"] = (
        max(0.0, float(legacy.get("rate", 0.0))) / rate_per_uW
        if rate_per_uW > 0.0
        else 0.0
    )
    del legacy_inputs[prefix]
    if legacy_inputs:
        st.session_state["_legacy_pump_inputs"] = legacy_inputs
    else:
        st.session_state.pop("_legacy_pump_inputs", None)


def _prepare_beam_state(prefix, atom_name, n2_coeffs, default_Fg, q_axis_value):
    """Normalize a pump's state before any tab-local widgets are instantiated."""
    atom = ATOMS[atom_name]
    line_key = f"line_{prefix}"
    if st.session_state.get(line_key) not in ("D1", "D2"):
        st.session_state[line_key] = "D1"
    line = st.session_state[line_key]

    k_key = f"k_{prefix}"
    if st.session_state.get(k_key) not in ("z", "x", "y"):
        st.session_state[k_key] = "x"
    pol_key = f"pol_{prefix}"
    pol_options = allowed_polarizations(st.session_state[k_key])
    if st.session_state.get(pol_key) not in pol_options:
        st.session_state[pol_key] = pol_options[0]

    transition_options = transition_choice_labels(
        atom, line, n2_pressure_torr, n2_coeffs, allowed_only=show_allowed_only
    )
    transition_key = f"transition_{prefix}"
    if st.session_state.get(transition_key) not in transition_options:
        st.session_state[transition_key] = default_transition_label(
            atom,
            line,
            n2_pressure_torr,
            n2_coeffs,
            default_Fg,
            default_Fg,
            allowed_only=show_allowed_only,
        )
    _migrate_legacy_beam_intensity(
        prefix,
        atom,
        n2_coeffs,
        line,
        st.session_state[transition_key],
        float(st.session_state[f"det_rel_{prefix}"]),
        q_axis_value,
    )
    # These keys were widget-bound before v6.1.2. Reassigning them detaches
    # existing sessions from Streamlit's stale-widget cleanup during upgrade.
    for field in ("line", "transition", "det_rel", "intensity", "k", "pol"):
        state_key = f"{field}_{prefix}"
        st.session_state[state_key] = st.session_state[state_key]


def _beam_from_state(
    prefix, target_label, atom_name, n2_coeffs, active, q_axis_value, placeholder=None
):
    atom = ATOMS[atom_name]
    line = st.session_state[f"line_{prefix}"]
    transition = st.session_state[f"transition_{prefix}"]
    det_rel = float(st.session_state[f"det_rel_{prefix}"])
    detuning, selected = absolute_detuning_from_transition_choice(
        atom=atom,
        line=line,
        transition_label=transition,
        relative_detuning_MHz=det_rel,
        n2_pressure_torr=n2_pressure_torr,
        n2_coeffs=n2_coeffs,
        allowed_only=show_allowed_only,
    )
    return {
        "name": f"Pump{prefix}",
        "target_label": target_label,
        "target_atom": atom_name,
        "line": line,
        "transition_label": transition,
        "selected_transition": selected,
        "detuning_relative": det_rel,
        "detuning": float(detuning),
        "absolute_frequency_MHz": line_center_frequency_MHz(atom, line) + float(detuning),
        "intensity": float(st.session_state[f"intensity_{prefix}"]),
        "k_axis": st.session_state[f"k_{prefix}"],
        "pol": st.session_state[f"pol_{prefix}"],
        "q_axis": q_axis_value,
        "rate_placeholder": placeholder,
        "active": active,
    }


def _pump_widget_key(prefix, field):
    return f"_pump_widget_{field}_{prefix}"


def _store_pump_widget_value(prefix, field):
    """Copy a visible pump widget value into its persistent condition state."""
    previous_intensity = float(st.session_state[f"intensity_{prefix}"])
    st.session_state[f"{field}_{prefix}"] = st.session_state[
        _pump_widget_key(prefix, field)
    ]
    if field == "intensity" or previous_intensity > 0.0:
        st.session_state["_pump_requires_app_rerun"] = True


def _prime_pump_widget(prefix, field):
    """Restore a lazy tab widget from persistent state before it is rendered."""
    widget_key = _pump_widget_key(prefix, field)
    st.session_state[widget_key] = st.session_state[f"{field}_{prefix}"]
    return widget_key


def _beam_config_ui(
    prefix, target_label, atom_name, n2_coeffs, active, default_Fg, q_axis_value
):
    st.markdown(f"#### Pump{prefix}")
    k_axis = st.selectbox(
        "Beam direction",
        ["z", "x", "y"],
        key=_prime_pump_widget(prefix, "k"),
        on_change=_store_pump_widget_value,
        args=(prefix, "k"),
    )
    pol_options = allowed_polarizations(k_axis)
    st.selectbox(
        "Polarization",
        pol_options,
        key=_prime_pump_widget(prefix, "pol"),
        on_change=_store_pump_widget_value,
        args=(prefix, "pol"),
    )
    line = st.selectbox(
        "Reference line",
        ["D1", "D2"],
        key=_prime_pump_widget(prefix, "line"),
        on_change=_store_pump_widget_value,
        args=(prefix, "line"),
    )
    transition_options = transition_choice_labels(
        ATOMS[atom_name], line, n2_pressure_torr, n2_coeffs,
        allowed_only=show_allowed_only,
    )
    st.selectbox(
        "Hyperfine transition",
        transition_options,
        key=_prime_pump_widget(prefix, "transition"),
        on_change=_store_pump_widget_value,
        args=(prefix, "transition"),
    )
    st.number_input(
        "Detuning (MHz)",
        step=10.0,
        format="%g",
        key=_prime_pump_widget(prefix, "det_rel"),
        on_change=_store_pump_widget_value,
        args=(prefix, "det_rel"),
        help="Detuning relative to the selected pressure-shifted hyperfine transition.",
    )
    st.number_input(
        "Intensity (µW/cm²)",
        min_value=0.0,
        step=1.0,
        format="%.1f",
        key=_prime_pump_widget(prefix, "intensity"),
        on_change=_store_pump_widget_value,
        args=(prefix, "intensity"),
    )
    rate_placeholder = st.empty()
    if not active:
        rate_placeholder.caption("Stored but ignored while Alkali B is inactive.")
    else:
        saved_caption = st.session_state.get(f"_pump_rate_caption_{prefix}")
        if saved_caption:
            rate_placeholder.caption(saved_caption)
    return _beam_from_state(
        prefix, target_label, atom_name, n2_coeffs, active, q_axis_value,
        rate_placeholder,
    )


@st.fragment
def _pump_configuration_ui(
    atom_A_name,
    atom_B_name,
    active_B,
    n2_coeffs_A,
    n2_coeffs_B,
    q_axis_A,
    q_axis_B,
):
    """Render pump controls independently until a physical beam changes."""
    pump_B_atom_name = atom_B_name if atom_B_name != "None" else atom_A_name
    pump_B_coeffs = n2_coeffs_B if atom_B_name != "None" else n2_coeffs_A
    for prep_args in (
        ("A1", atom_A_name, n2_coeffs_A, 1, q_axis_A),
        ("A2", atom_A_name, n2_coeffs_A, 2, q_axis_A),
        ("B1", pump_B_atom_name, pump_B_coeffs, 1, q_axis_B),
        ("B2", pump_B_atom_name, pump_B_coeffs, 2, q_axis_B),
    ):
        _prepare_beam_state(*prep_args)

    st.header("Pump configuration")
    pump_tab_A, pump_tab_B = st.tabs(
        ["Alkali A", "Alkali B"],
        key="pump_configuration_tab",
        on_change="rerun",
    )
    beam_A1 = _beam_from_state("A1", "A", atom_A_name, n2_coeffs_A, True, q_axis_A)
    beam_A2 = _beam_from_state("A2", "A", atom_A_name, n2_coeffs_A, True, q_axis_A)
    beam_B1 = _beam_from_state(
        "B1", "B", pump_B_atom_name, pump_B_coeffs, active_B, q_axis_B
    )
    beam_B2 = _beam_from_state(
        "B2", "B", pump_B_atom_name, pump_B_coeffs, active_B, q_axis_B
    )
    if pump_tab_A.open:
        with pump_tab_A:
            pump_A_col_1, pump_A_col_2 = st.columns(2, gap="xsmall")
            with pump_A_col_1:
                beam_A1 = _beam_config_ui(
                    "A1", "A", atom_A_name, n2_coeffs_A, True, 1, q_axis_A
                )
            with pump_A_col_2:
                beam_A2 = _beam_config_ui(
                    "A2", "A", atom_A_name, n2_coeffs_A, True, 2, q_axis_A
                )
    elif pump_tab_B.open:
        with pump_tab_B:
            pump_B_col_1, pump_B_col_2 = st.columns(2, gap="xsmall")
            with pump_B_col_1:
                beam_B1 = _beam_config_ui(
                    "B1", "B", pump_B_atom_name, pump_B_coeffs, active_B, 1, q_axis_B
                )
            with pump_B_col_2:
                beam_B2 = _beam_config_ui(
                    "B2", "B", pump_B_atom_name, pump_B_coeffs, active_B, 2, q_axis_B
                )
    if st.session_state.pop("_pump_requires_app_rerun", False):
        st.rerun()
    return beam_A1, beam_A2, beam_B1, beam_B2


with st.sidebar:
    st.header("Condition")
    condition_controls_placeholder = st.empty()

    st.header("Atom / cell")
    atom_col_A, atom_col_B = st.columns(2, gap="xsmall")
    with atom_col_A:
        atom_A_name = st.selectbox("Alkali A", list(ATOMS), key="atom_A_name")
    with atom_col_B:
        atom_B_name = st.selectbox("Alkali B", ["None", *list(ATOMS)], key="atom_B_name")
    active_B = atom_B_name != "None" and atom_B_name != atom_A_name
    if atom_B_name == "None":
        st.caption("Alkali B is inactive.")
    elif atom_B_name == atom_A_name:
        st.caption("Alkali B matches Alkali A, so its physical effects are ignored.")

    _initialize_atom_coefficients("A", atom_A_name)
    _initialize_atom_coefficients("B", atom_B_name)
    n2_coeffs_A = _n2_coefficients("A")
    n2_coeffs_B = _n2_coefficients("B")

    density_model_col, density_ratio_col = st.columns([0.63, 0.37], gap="xsmall")
    with density_model_col:
        density_mode = st.selectbox(
            "Mixture density model",
            ["Independent saturated-vapor curves", "Relative concentration"],
            key="density_mode",
        )
    with density_ratio_col:
        density_ratio = st.number_input(
            "n(B) / n(A)",
            min_value=0.0,
            step=0.1,
            format="%.3g",
            key="density_ratio_B_to_A",
            disabled=(not active_B or density_mode != "Relative concentration"),
        )

    cell_col_1, cell_col_2 = st.columns(2, gap="xsmall")
    with cell_col_1:
        n2_pressure_torr = st.number_input(
            "N₂ pressure (Torr)", min_value=0.0, step=10.0, format="%.1f", key="n2_pressure_torr"
        )
    with cell_col_2:
        temperature_C = st.number_input(
            "Temperature (°C)", step=1.0, format="%.1f", key="temperature_C_for_table"
        )

    density_A, density_B = resolve_alkali_densities(
        atom_A_name, atom_B_name, temperature_C, density_mode, density_ratio
    )
    include_spin_exchange = st.checkbox("Include spin exchange", key="include_spin_exchange")
    er_col_A, er_col_B = st.columns(2, gap="xsmall")
    with er_col_A:
        R_ER_A = st.number_input(
            r"$R_{\mathrm{ER},A}$ (s⁻¹)", min_value=0.0, step=1.0, format="%.1f", key="gamma_ER_A"
        )
    with er_col_B:
        R_ER_B = st.number_input(
            r"$R_{\mathrm{ER},B}$ (s⁻¹)", min_value=0.0, step=1.0, format="%.1f", key="gamma_ER_B"
        )
    if active_B:
        cross_A = cross_spin_exchange_rate_info(atom_A_name, atom_B_name, temperature_C, density_B)
        cross_B = cross_spin_exchange_rate_info(atom_B_name, atom_A_name, temperature_C, density_A)
        st.caption(
            f"nA={density_A:.3g} cm⁻³ · nB={density_B:.3g} cm⁻³ · "
            f"R(A←B)={cross_A['rate_s']:.3g} s⁻¹ · R(B←A)={cross_B['rate_s']:.3g} s⁻¹"
        )
    else:
        st.caption(f"nA={density_A:.3g} cm⁻³")

    field_col_1, field_col_2 = st.columns(2, gap="xsmall")
    with field_col_1:
        static_field_axis = st.selectbox(
            "Static field direction", ["z", "x", "y"], key="static_field_axis"
        )
    with field_col_2:
        static_field_nT = st.number_input(
            "Static field strength (nT)",
            step=1.0,
            format="%g",
            key="static_field_nT",
            help="A negative value reverses the selected field direction.",
        )

    q_axis_A = st.session_state["q_axis_A"]
    q_axis_B = st.session_state["q_axis_B"]

    show_allowed_only = bool(st.session_state["show_allowed_only"])
    beam_A1, beam_A2, beam_B1, beam_B2 = _pump_configuration_ui(
        atom_A_name,
        atom_B_name,
        active_B,
        n2_coeffs_A,
        n2_coeffs_B,
        q_axis_A,
        q_axis_B,
    )

    st.header("Display")
    show_allowed_only = st.checkbox("Only show allowed hyperfine transitions", key="show_allowed_only")
    show_rate_matrices = st.checkbox("Show rate matrices", key="show_rate_matrices")

    with condition_controls_placeholder.container():
        load_col, save_col, name_col = st.columns([0.30, 0.20, 0.50], gap="xsmall")
        with load_col:
            st.file_uploader(
                "Load condition",
                type=["json"],
                key="condition_file_uploader",
                label_visibility="collapsed",
                help="Load a v6.1 condition or migrate a v6.0/v5.0 condition.",
                on_change=load_condition_callback,
            )
        with save_col:
            save_placeholder = st.empty()
        with name_col:
            condition_name = st.text_input("Condition name", key="condition_name")
        condition_save_name = clean_condition_name(condition_name)
        payload = build_condition_payload(current_condition_values(condition_save_name))
        save_placeholder.download_button(
            "Save",
            data=json.dumps(payload, indent=2),
            file_name=f"{condition_save_name}.json",
            mime="application/json",
            key="save_condition_button",
            width="stretch",
        )
        if st.session_state.get("_condition_load_message"):
            st.success(st.session_state.pop("_condition_load_message"))
        if st.session_state.get("_condition_load_error"):
            st.error("Could not load condition file: " + st.session_state.pop("_condition_load_error"))


def _rf_frequency_samples(label):
    normalize_rf_frequency_bounds(label)
    lower = float(st.session_state[f"rf_frequency_lower_hz_{label}"])
    upper = float(st.session_state[f"rf_frequency_upper_hz_{label}"])
    if np.isclose(lower, upper):
        return np.array([lower], dtype=float)
    return np.linspace(lower, upper, 1201)


rf_frequencies_A = _rf_frequency_samples("A")
rf_frequencies_B = _rf_frequency_samples("B")

all_beams = [beam_A1, beam_A2]
if active_B:
    all_beams.extend([beam_B1, beam_B2])


def _physical_beam_config(beam):
    """Strip UI-only objects from an active beam before solver caching."""
    return {
        key: beam[key]
        for key in (
            "name",
            "target_label",
            "absolute_frequency_MHz",
            "intensity",
            "k_axis",
            "pol",
            "selected_transition",
            "transition_label",
        )
    }


@st.cache_data(max_entries=24, show_spinner=False)
def _compute_alkali_system_cached(
    species_A_config, species_B_config, physical_beams, common
):
    return compute_alkali_system(
        species_A_config, species_B_config, physical_beams, common
    )

species_A_config = {
    "label": "A",
    "atom_name": atom_A_name,
    "density_cm3": density_A,
    "R_ER": R_ER_A,
    "n2_coeffs": n2_coeffs_A,
    "q_axis": q_axis_A,
    "rf_axis": st.session_state["rf_axis_A"],
    "rf_observable": st.session_state["rf_observable_A"],
    "rf_frequencies_hz": rf_frequencies_A,
}
species_B_config = None
if active_B:
    species_B_config = {
        "label": "B",
        "atom_name": atom_B_name,
        "density_cm3": density_B,
        "R_ER": R_ER_B,
        "n2_coeffs": n2_coeffs_B,
        "q_axis": q_axis_B,
        "rf_axis": st.session_state["rf_axis_B"],
        "rf_observable": st.session_state["rf_observable_B"],
        "rf_frequencies_hz": rf_frequencies_B,
    }
common = {
    "temperature_C": temperature_C,
    "n2_pressure_torr": n2_pressure_torr,
    "include_spin_exchange": include_spin_exchange,
    "static_field_axis": static_field_axis,
    "static_field_nT": static_field_nT,
}
physical_beams = [
    _physical_beam_config(beam)
    for beam in all_beams
    if float(beam["intensity"]) > 0.0
]
system = _compute_alkali_system_cached(
    species_A_config, species_B_config, physical_beams, common
)


def _populate_rate_captions():
    for beam in (beam_A1, beam_A2, beam_B1, beam_B2):
        if not beam["active"]:
            continue
        result = system[beam["target_label"]]
        diagnostic = next(
            ((candidate, info) for candidate, info in result["diagnostics"] if candidate["name"] == beam["name"]),
            None,
        )
        if diagnostic is None:
            caption = "Configured target pump rate: 0 s⁻¹"
            st.session_state[f"_pump_rate_caption_{beam['name'][4:]}"] = caption
            if beam["rate_placeholder"] is not None:
                beam["rate_placeholder"].caption(caption)
            continue
        _, info = diagnostic
        selected_rows = info["reference_ground_indices"]
        selected_rate = float(info["R_ge"][selected_rows, :].sum())
        caption = f"F={info['reference_Fg']:g} target rate: {selected_rate:.3g} s⁻¹"
        st.session_state[f"_pump_rate_caption_{beam['name'][4:]}"] = caption
        if beam["rate_placeholder"] is not None:
            beam["rate_placeholder"].caption(caption)


_populate_rate_captions()


def _zeeman_display_dataframe(result):
    renamed = result["df_pop"].rename(columns={
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
        "G_ER": "G^{ER} (s^-1)",
        "Gamma_ER": "Γ^{ER} (s^-1)",
        "G_SE_self": "G^{SE,self} (s^-1)",
        "Gamma_SE_self": "Γ^{SE,self} (s^-1)",
        "G_SE_cross": "G^{SE,cross} (s^-1)",
        "Gamma_SE_cross": "Γ^{SE,cross} (s^-1)",
        "G_SE": "G^{SE} (s^-1)",
        "Gamma_SE": "Γ^{SE} (s^-1)",
        "G_total": "G (s^-1)",
        "Gamma_total": "Γ (s^-1)",
        "Gamma_total_over_2pi": "Γ/2π (Hz)",
    })
    columns = [
        "F", "m", "P_F", "Pₘ", "Dₘ", "ν^{VS} (Hz)", "ν^{TS} (Hz)",
        "ν^{LS} (Hz)", "ν^{B} (Hz)", "ν_m (Hz)", "Λ (s⁻¹)",
        "G^{OP} (s^-1)", "G^{ER} (s^-1)", "Γ^{ER} (s^-1)",
        "G^{SE,self} (s^-1)", "Γ^{SE,self} (s^-1)",
        "G^{SE,cross} (s^-1)", "Γ^{SE,cross} (s^-1)",
        "G^{SE} (s^-1)", "Γ^{SE} (s^-1)", "G (s^-1)",
        "Γ (s^-1)", "Γ/2π (Hz)",
    ]
    return renamed[columns].sort_values(["F", "m"], ascending=[False, False], kind="stable").reset_index(drop=True)


def _compact_title(text):
    st.markdown(
        f"<div style='text-align:center;font-size:1.25rem;font-weight:600;margin:.25rem 0 .45rem'>{text}</div>",
        unsafe_allow_html=True,
    )


def _render_population_plot(result):
    states = result["ground_states"]
    population = result["population"]
    energies = {}
    for state in states:
        energies.setdefault(float(state["F"]), float(state["E"]))
    manifolds = sorted(energies, key=energies.get, reverse=True)
    fig, axes = plt.subplots(len(manifolds), 1, figsize=(4.6, 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, F in zip(axes, manifolds):
        indices = [i for i, state in enumerate(states) if np.isclose(float(state["F"]), F)]
        axis.bar([f"{states[i]['m']:g}" for i in indices], population[indices])
        axis.set_title(f"F={F:g}", fontsize=11, pad=3)
        axis.set_ylabel("Population")
    axes[-1].set_xlabel(f"m along {result['q_axis']}")
    axis_max = max(0.01, 1.08 * float(np.max(population)))
    for axis in axes:
        axis.set_ylim(0.0, axis_max)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        f"⟨m⟩={expectation_m(states, population):.4f} · "
        f"⟨m²⟩={expectation_m2(states, population):.4f} · "
        f"n={result['density_cm3']:.3g} cm⁻³"
    )


def _result_widget_key(state_key):
    return f"_result_widget_{state_key}"


def _store_result_widget_value(state_key):
    """Copy a lazy result-tab widget into persistent condition state."""
    st.session_state[state_key] = st.session_state[_result_widget_key(state_key)]


def _prime_result_widget(state_key):
    """Restore a result-tab widget from persistent state before rendering."""
    widget_key = _result_widget_key(state_key)
    st.session_state[widget_key] = st.session_state[state_key]
    return widget_key


def _render_rf(result, label):
    rf_axis = st.session_state[f"rf_axis_{label}"]
    rf_observable = st.session_state[f"rf_observable_{label}"]
    rf_show_amplitude = bool(st.session_state[f"rf_show_amplitude_{label}"])
    rf_show_in_phase = bool(st.session_state[f"rf_show_in_phase_{label}"])
    rf_show_quadrature = bool(st.session_state[f"rf_show_quadrature_{label}"])
    rf_relaxation_normalized = bool(
        st.session_state[f"rf_relaxation_normalized_{label}"]
    )
    rf_density_factor = bool(st.session_state[f"rf_density_factor_{label}"])
    reference = result["rf_relaxation_reference"]
    gamma = reference.get("Gamma_m") if rf_relaxation_normalized and reference.get("available", False) else None
    density = result["density_cm3"] if rf_density_factor else None
    plotted = prepare_weak_rf_plot_values(
        result["rf_amplitude"], result["rf_in_phase"], result["rf_quadrature"],
        relaxation_gamma_s_inv=gamma, density_cm3=density,
    )
    export = weak_rf_export_dataframe(
        frequencies_hz=result["rf_frequencies_hz"],
        susceptibility_amplitude=result["rf_amplitude"],
        susceptibility_in_phase=result["rf_in_phase"],
        susceptibility_quadrature=result["rf_quadrature"],
        plotted_amplitude=plotted[0], plotted_in_phase=plotted[1], plotted_quadrature=plotted[2],
        relaxation_normalized=gamma is not None,
        normalization_gamma_s_inv=gamma,
        density_factored=rf_density_factor,
        density_cm3=density,
    )
    title_col, download_col = st.columns([0.82, 0.18], gap="small")
    with title_col:
        _compact_title(
            f"Alkali {label} upper-hyperfine weak-RF susceptibility "
            f"(F={result['rf_upper_F']:g})"
        )
    with download_col:
        st.download_button(
            "Download CSV", dataframe_to_csv_bytes(export),
            file_name=f"{condition_save_name}_alkali-{label}_weak-rf.csv",
            mime="text/csv; charset=utf-8", key=f"download_rf_{label}", width="stretch",
        )
    control_col, plot_col = st.columns([0.18, 0.82], gap="small")
    with control_col:
        st.caption(f"RF-{label} applied; the other RF drive is zero.")
        rf_axis_key = f"rf_axis_{label}"
        st.selectbox(
            "RF axis",
            ["x", "y", "z"],
            key=_prime_result_widget(rf_axis_key),
            on_change=_store_result_widget_value,
            args=(rf_axis_key,),
        )
        rf_observable_key = f"rf_observable_{label}"
        st.selectbox(
            "Observable",
            ["Fx", "Fy", "Fz"],
            key=_prime_result_widget(rf_observable_key),
            on_change=_store_result_widget_value,
            args=(rf_observable_key,),
        )
        lower_col, upper_col = st.columns(2, gap="xxsmall")
        with lower_col:
            lower_key = f"rf_frequency_lower_hz_{label}"
            st.number_input(
                "Lower (Hz)", min_value=0.0, step=1.0, format="%g",
                key=_prime_result_widget(lower_key),
                on_change=_store_result_widget_value,
                args=(lower_key,),
            )
        with upper_col:
            upper_key = f"rf_frequency_upper_hz_{label}"
            st.number_input(
                "Upper (Hz)", min_value=0.0, step=1.0, format="%g",
                key=_prime_result_widget(upper_key),
                on_change=_store_result_widget_value,
                args=(upper_key,),
            )
        for checkbox_label, state_key in (
            ("Amplitude", f"rf_show_amplitude_{label}"),
            ("In phase", f"rf_show_in_phase_{label}"),
            ("Quadrature", f"rf_show_quadrature_{label}"),
            ("Relaxation normalized", f"rf_relaxation_normalized_{label}"),
            ("Density factor", f"rf_density_factor_{label}"),
        ):
            st.checkbox(
                checkbox_label,
                key=_prime_result_widget(state_key),
                on_change=_store_result_widget_value,
                args=(state_key,),
            )

    with plot_col:
        if not result["static_field_aligned"] and abs(static_field_nT) > 0.0:
            st.warning(
                "The static field is transverse to this quantization axis. "
                "The current population model includes only the field component "
                "parallel to the quantization axis, so transverse static-field "
                "mixing is omitted."
            )
        elif not result["light_shift_available"]:
            st.warning(
                "RF response is unavailable because an active optical field has "
                "a non-diagonal light shift."
            )
        elif result["rf_info"].get("used_transitions", 0) == 0:
            st.info("No driven adjacent Zeeman transitions are available for this RF geometry.")
        else:
            selected_curves = []
            if rf_show_amplitude:
                selected_curves.append((plotted[0], "Amplitude", "-"))
            if rf_show_in_phase:
                selected_curves.append((plotted[1], "In phase", "--"))
            if rf_show_quadrature:
                selected_curves.append((plotted[2], "Quadrature", ":"))
            if not selected_curves:
                st.info("Select at least one RF curve.")
            else:
                fig, axis = plt.subplots(figsize=(8.6, 4.2))
                for values, curve_label, linestyle in selected_curves:
                    axis.plot(
                        result["rf_frequencies_hz"], values,
                        linestyle=linestyle, label=curve_label,
                    )
                axis.axhline(0.0, linewidth=0.8, alpha=0.45)
                axis.set_xlabel(f"RF-{label} frequency (Hz)")
                axis.set_ylabel(f"Alkali {label} weak-RF susceptibility")
                axis.grid(True, alpha=0.25)
                axis.legend(frameon=False)
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)


def _matrix_dataframe(matrix, result, prefix=""):
    labels = [f"{prefix}{state['label']}" for state in result["ground_states"]]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def _render_species_result(result, label):
    q_control, q_caption = st.columns([0.22, 0.78], gap="small")
    with q_control:
        q_axis_key = f"q_axis_{label}"
        st.selectbox(
            f"Alkali {label} quantization axis",
            ["z", "x", "y"],
            key=_prime_result_widget(q_axis_key),
            on_change=_store_result_widget_value,
            args=(q_axis_key,),
        )
    with q_caption:
        st.caption(
            f"Shared static field: {static_field_nT:g} nT along {static_field_axis}; "
            f"Alkali {label} upper-manifold Larmor frequency: "
            f"{result['bias_larmor_hz']:.6g} Hz."
        )
    left, right = st.columns([0.62, 1.63], gap="small")
    with left:
        _compact_title(f"{result['atom_name']} ground-state populations")
        _render_population_plot(result)
        self_rate = result["self_rate_info"]["rate_s"] if include_spin_exchange else 0.0
        st.caption(
            f"R_SE,self={self_rate:.3g} s⁻¹ · R_SE,cross={result['R_SE_cross']:.3g} s⁻¹ · "
            f"R_ER={result['R_ER']:.3g} s⁻¹"
        )
    display_df = _zeeman_display_dataframe(result)
    with right:
        title_col, download_col = st.columns([0.80, 0.20], gap="small")
        with title_col:
            _compact_title("Zeeman sublevel properties")
        with download_col:
            st.download_button(
                "Download CSV", dataframe_to_csv_bytes(display_df),
                file_name=f"{condition_save_name}_alkali-{label}_zeeman.csv",
                mime="text/csv; charset=utf-8", key=f"download_zeeman_{label}", width="stretch",
            )
        st.markdown(render_zeeman_properties_table_html(display_df), unsafe_allow_html=True)
        st.caption(
            "SE,self is same-species spin exchange; SE,cross is exchange with the other active alkali. "
            "G and Γ are the sums of all displayed population and adjacent-coherence relaxation mechanisms."
        )
    _render_rf(result, label)
    with st.expander("Optical transition frequencies"):
        transition_df = hyperfine_transition_table(
            atom=result["atom"], n2_pressure_torr=n2_pressure_torr,
            n2_coeffs=result["n2_coeffs"], allowed_only=show_allowed_only,
            pump_beams=[
                beam for beam in all_beams if beam["target_label"] == label
            ],
            temperature_C=temperature_C,
        )
        st.markdown(render_transition_table_html(transition_df), unsafe_allow_html=True)
    if show_rate_matrices:
        with st.expander(f"Alkali {label} rate matrices", expanded=False):
            st.markdown("**Optical population generator**")
            st.dataframe(_matrix_dataframe(result["L_optical"], result), width="stretch")
            st.markdown("**Electron-randomization map**")
            st.dataframe(_matrix_dataframe(result["M_ER"], result), width="stretch")
            st.markdown("**Self spin-exchange map**")
            st.dataframe(_matrix_dataframe(result["self_map"], result), width="stretch")
            st.markdown("**Cross spin-exchange map**")
            st.dataframe(_matrix_dataframe(result["cross_map"], result), width="stretch")
            st.markdown("**Local small-signal population Jacobian block**")
            st.dataframe(_matrix_dataframe(result["J_population"], result), width="stretch")


if active_B:
    result_labels = [f"Alkali A — {atom_A_name}", f"Alkali B — {atom_B_name}"]
    if st.session_state.get("result_species_tab") not in result_labels:
        st.session_state["result_species_tab"] = result_labels[0]
    result_tab_A, result_tab_B = st.tabs(
        result_labels,
        key="result_species_tab",
        on_change="rerun",
    )
    if result_tab_A.open:
        with result_tab_A:
            _render_species_result(system["A"], "A")
    elif result_tab_B.open:
        with result_tab_B:
            _render_species_result(system["B"], "B")
else:
    _render_species_result(system["A"], "A")

if show_rate_matrices and active_B:
    with st.expander("Coupled A/B population Jacobian", expanded=False):
        labels_A = [f"A:{state['label']}" for state in system["A"]["ground_states"]]
        labels_B = [f"B:{state['label']}" for state in system["B"]["ground_states"]]
        labels = labels_A + labels_B
        st.dataframe(pd.DataFrame(system["J_coupled"], index=labels, columns=labels), width="stretch")
        st.caption("Block order: J_AA, J_AB; J_BA, J_BB. Off-diagonal blocks are cross-species spin-exchange feedback.")

solve = system["solve"]
if not solve.get("converged", True):
    st.warning(
        f"The nonlinear spin-exchange fixed-point iteration did not fully converge "
        f"(residual={solve.get('residual', float('nan')):.3g})."
    )
