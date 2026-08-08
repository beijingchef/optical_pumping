"""Condition-file serialization and Streamlit session-state helpers."""

import json
from datetime import datetime

import streamlit as st

from ..physics.constants import field_nT_from_upper_larmor_frequency
from ..version import CONDITION_SCHEMA_VERSION


RF_FIELDS = (
    "axis",
    "observable",
    "frequency_lower_hz",
    "frequency_upper_hz",
    "show_amplitude",
    "show_in_phase",
    "show_quadrature",
    "relaxation_normalized",
    "density_factor",
)


def _rf_condition_keys(label):
    return tuple(f"rf_{field}_{label}" for field in RF_FIELDS)


RF_CONDITION_KEYS = (*_rf_condition_keys("A"), *_rf_condition_keys("B"))


def _pump_condition_keys(prefix):
    return tuple(
        f"{field}_{prefix}"
        for field in ("line", "transition", "det_rel", "intensity", "k", "pol")
    )


PUMP_CONDITION_KEYS = (
    *_pump_condition_keys("A1"),
    *_pump_condition_keys("A2"),
    *_pump_condition_keys("B1"),
    *_pump_condition_keys("B2"),
)

CONDITION_KEYS = (
    "condition_name",
    "atom_A_name",
    "atom_B_name",
    "density_mode",
    "density_ratio_B_to_A",
    "gamma_ER_A",
    "gamma_ER_B",
    "static_field_axis",
    "static_field_nT",
    "q_axis_A",
    "q_axis_B",
    "temperature_C_for_table",
    "n2_pressure_torr",
    "include_spin_exchange",
    "D1_width_A",
    "D2_width_A",
    "D1_shift_A",
    "D2_shift_A",
    "D1_width_B",
    "D2_width_B",
    "D1_shift_B",
    "D2_shift_B",
    *PUMP_CONDITION_KEYS,
    *RF_CONDITION_KEYS,
    "show_allowed_only",
    "show_rate_matrices",
)

DEFAULT_STARTUP_CONDITION = {
    "condition_name": "default-dual-alkali",
    "atom_A_name": "Rb87",
    "atom_B_name": "None",
    "density_mode": "Independent saturated-vapor curves",
    "density_ratio_B_to_A": 1.0,
    "gamma_ER_A": 4.0,
    "gamma_ER_B": 4.0,
    "static_field_axis": "z",
    "static_field_nT": 0.0,
    "q_axis_A": "z",
    "q_axis_B": "z",
    "temperature_C_for_table": 23.5,
    "n2_pressure_torr": 0.0,
    "include_spin_exchange": True,
    "D1_width_A": 17.8,
    "D2_width_A": 18.1,
    "D1_shift_A": -8.25,
    "D2_shift_A": -5.9,
    "D1_width_B": 17.8,
    "D2_width_B": 18.1,
    "D1_shift_B": -8.25,
    "D2_shift_B": -5.9,
    "line_A1": "D1",
    "transition_A1": "1→2",
    "det_rel_A1": 0.0,
    "intensity_A1": 5.0,
    "k_A1": "x",
    "pol_A1": "linear z",
    "line_A2": "D1",
    "transition_A2": "2→2",
    "det_rel_A2": 400.0,
    "intensity_A2": 5.0,
    "k_A2": "x",
    "pol_A2": "linear z",
    "line_B1": "D1",
    "transition_B1": "1→2",
    "det_rel_B1": 0.0,
    "intensity_B1": 0.0,
    "k_B1": "x",
    "pol_B1": "linear z",
    "line_B2": "D1",
    "transition_B2": "2→2",
    "det_rel_B2": 0.0,
    "intensity_B2": 0.0,
    "k_B2": "x",
    "pol_B2": "linear z",
    "rf_axis_A": "x",
    "rf_observable_A": "Fx",
    "rf_frequency_lower_hz_A": 0.0,
    "rf_frequency_upper_hz_A": 50.0,
    "rf_show_amplitude_A": True,
    "rf_show_in_phase_A": False,
    "rf_show_quadrature_A": False,
    "rf_relaxation_normalized_A": False,
    "rf_density_factor_A": False,
    "rf_axis_B": "x",
    "rf_observable_B": "Fx",
    "rf_frequency_lower_hz_B": 0.0,
    "rf_frequency_upper_hz_B": 50.0,
    "rf_show_amplitude_B": True,
    "rf_show_in_phase_B": False,
    "rf_show_quadrature_B": False,
    "rf_relaxation_normalized_B": False,
    "rf_density_factor_B": False,
    "show_allowed_only": True,
    "show_rate_matrices": False,
}


def clean_condition_name(value):
    name = str(value or "").strip()
    if name.lower().endswith(".json"):
        name = name[:-5].rstrip()
    return name or "default"


def build_condition_payload(values):
    conditions = {key: values.get(key) for key in CONDITION_KEYS}
    conditions["condition_name"] = clean_condition_name(conditions.get("condition_name"))
    return {
        "app": "alkali_pumping",
        "format": "alkali_pumping_conditions",
        "version": CONDITION_SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "conditions": conditions,
    }


def current_condition_values(condition_name=None):
    values = {key: st.session_state.get(key) for key in CONDITION_KEYS}
    if condition_name is not None:
        values["condition_name"] = condition_name
    return values


def normalize_rf_frequency_bounds(label="A", prefer="lower"):
    lower_key = f"rf_frequency_lower_hz_{label}"
    upper_key = f"rf_frequency_upper_hz_{label}"
    lower = max(0.0, float(st.session_state.get(lower_key, 0.0)))
    upper = max(0.0, float(st.session_state.get(upper_key, lower)))
    if lower > upper:
        if prefer == "upper":
            lower = upper
        else:
            upper = lower
    st.session_state[lower_key] = lower
    st.session_state[upper_key] = upper


def _copy_legacy_rf(conditions, migrated):
    for field in RF_FIELDS:
        old_key = f"rf_{field}"
        if old_key not in conditions:
            continue
        for label in ("A", "B"):
            migrated[f"rf_{field}_{label}"] = conditions[old_key]


def _migrate_v6_conditions(conditions):
    """Translate a v6.0 dual-alkali condition into the species-control schema."""
    migrated = dict(DEFAULT_STARTUP_CONDITION)
    for key in CONDITION_KEYS:
        if key in conditions:
            migrated[key] = conditions[key]
    old_q = conditions.get("q_axis", "z")
    migrated["static_field_axis"] = old_q
    migrated["q_axis_A"] = old_q
    migrated["q_axis_B"] = old_q
    atom_A_name = conditions.get("atom_A_name", migrated["atom_A_name"])
    migrated["static_field_nT"] = field_nT_from_upper_larmor_frequency(
        atom_A_name, conditions.get("bias_larmor_hz_A", 0.0)
    )
    _copy_legacy_rf(conditions, migrated)
    return migrated, {}


def _migrate_v5_conditions(conditions):
    """Translate a v5 single-alkali condition into the v6.1 A/B schema."""
    migrated = dict(DEFAULT_STARTUP_CONDITION)
    direct_map = {
        "condition_name": "condition_name",
        "atom_name": "atom_A_name",
        "gamma_ER": "gamma_ER_A",
        "temperature_C_for_table": "temperature_C_for_table",
        "n2_pressure_torr": "n2_pressure_torr",
        "include_spin_exchange": "include_spin_exchange",
        "D1_width": "D1_width_A",
        "D2_width": "D2_width_A",
        "D1_shift": "D1_shift_A",
        "D2_shift": "D2_shift_A",
        "show_allowed_only": "show_allowed_only",
        "show_rate_matrices": "show_rate_matrices",
    }
    for old_key, new_key in direct_map.items():
        if old_key in conditions:
            migrated[new_key] = conditions[old_key]
    old_q = conditions.get("q_axis", "z")
    migrated["static_field_axis"] = old_q
    migrated["q_axis_A"] = old_q
    migrated["q_axis_B"] = old_q
    migrated["static_field_nT"] = field_nT_from_upper_larmor_frequency(
        migrated["atom_A_name"], conditions.get("bias_larmor_hz", 0.0)
    )
    _copy_legacy_rf(conditions, migrated)

    legacy_pump_inputs = {}
    for old_number, new_prefix in ((1, "A1"), (2, "A2")):
        for field in ("line", "transition", "det_rel", "intensity", "k", "pol"):
            old_key = f"{field}{old_number}"
            if old_key in conditions:
                migrated[f"{field}_{new_prefix}"] = conditions[old_key]
        if f"intensity{old_number}" not in conditions and f"rate{old_number}" in conditions:
            legacy_pump_inputs[new_prefix] = {
                "rate": conditions[f"rate{old_number}"],
                "rate_reference": conditions.get(f"rate_reference{old_number}", "At detuning"),
            }
    migrated["atom_B_name"] = "None"
    return migrated, legacy_pump_inputs


def apply_loaded_condition_dict(payload):
    if not isinstance(payload, dict):
        raise ValueError("The loaded file is not a JSON object.")
    if payload.get("app") != "alkali_pumping":
        raise ValueError("This is not an alkali_pumping condition file.")
    if payload.get("format") != "alkali_pumping_conditions":
        raise ValueError("The JSON file is not an alkali_pumping condition file.")
    conditions = payload.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("The JSON file does not contain a conditions object.")

    version = payload.get("version")
    if version == CONDITION_SCHEMA_VERSION:
        loaded_conditions = dict(conditions)
        missing = [key for key in CONDITION_KEYS if key not in loaded_conditions]
        if missing:
            raise ValueError("The condition file is missing required fields: " + ", ".join(missing))
        legacy_pump_inputs = {}
    elif version == "6.0":
        loaded_conditions, legacy_pump_inputs = _migrate_v6_conditions(conditions)
    elif version == "5.0":
        loaded_conditions, legacy_pump_inputs = _migrate_v5_conditions(conditions)
    else:
        raise ValueError(
            "Unsupported condition-file version. Expected "
            f"{CONDITION_SCHEMA_VERSION}, legacy 6.0, or legacy 5.0."
        )

    loaded_name = clean_condition_name(loaded_conditions["condition_name"])
    for key in CONDITION_KEYS:
        value = loaded_conditions.get(key)
        if value is not None:
            st.session_state[key] = value
    if legacy_pump_inputs:
        st.session_state["_legacy_pump_inputs"] = legacy_pump_inputs
    else:
        st.session_state.pop("_legacy_pump_inputs", None)
    normalize_rf_frequency_bounds("A")
    normalize_rf_frequency_bounds("B")
    st.session_state["_last_atom_names_for_defaults"] = {
        "A": loaded_conditions["atom_A_name"],
        "B": loaded_conditions["atom_B_name"],
    }
    return loaded_name


def load_condition_callback():
    uploaded = st.session_state.get("condition_file_uploader")
    if uploaded is None:
        return
    try:
        payload = json.loads(uploaded.getvalue().decode("utf-8"))
        loaded_name = apply_loaded_condition_dict(payload)
        st.session_state["_condition_load_message"] = f"Loaded condition: {loaded_name}"
        st.session_state.pop("_condition_load_error", None)
    except Exception as exc:
        st.session_state["_condition_load_error"] = str(exc)
        st.session_state.pop("_condition_load_message", None)


APP_BASE_TITLE = "alkali pumping"


def current_browser_title():
    raw_name = st.session_state.get("condition_name", "")
    if not str(raw_name or "").strip():
        return APP_BASE_TITLE
    return f"{APP_BASE_TITLE}: {clean_condition_name(raw_name)}"
