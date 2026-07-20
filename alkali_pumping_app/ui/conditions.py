"""Condition-file serialization and Streamlit session-state helpers."""

import json
from datetime import datetime

import streamlit as st

from ..version import CONDITION_SCHEMA_VERSION

RF_CONDITION_KEYS = (
    "rf_axis",
    "rf_observable",
    "rf_frequency_lower_hz",
    "rf_frequency_upper_hz",
    "rf_show_amplitude",
    "rf_show_in_phase",
    "rf_show_quadrature",
    "rf_relaxation_normalized",
)
CONDITION_KEYS = (
    "condition_name",
    "atom_name", "gamma_ER", "q_axis", "bias_larmor_hz", "temperature_C_for_table", "n2_pressure_torr",
    "include_spin_exchange",
    "D1_width", "D2_width", "D1_shift", "D2_shift",
    "line1", "transition1", "det_rel1", "rate_reference1", "rate1", "k1", "pol1",
    "line2", "transition2", "det_rel2", "rate_reference2", "rate2", "k2", "pol2",
    "line3", "transition3", "det_rel3", "rate_reference3", "rate3", "k3", "pol3",
    *RF_CONDITION_KEYS,
    "show_allowed_only", "show_rate_matrices",
)
# Built-in startup condition: default-ps400.
DEFAULT_STARTUP_CONDITION = {
    "condition_name": "default-ps400",
    "atom_name": "Rb87",
    "gamma_ER": 4.0,
    "q_axis": "z",
    "bias_larmor_hz": 0.0,
    "temperature_C_for_table": 23.5,
    "n2_pressure_torr": 0.0,
    "include_spin_exchange": True,
    "D1_width": 17.8,
    "D2_width": 18.1,
    "D1_shift": -8.25,
    "D2_shift": -5.9,
    "line1": "D1",
    "transition1": "1→2",
    "det_rel1": 0.0,
    "rate_reference1": "At detuning",
    "rate1": 1200.0,
    "k1": "x",
    "pol1": "linear z",
    "line2": "D1",
    "transition2": "2→2",
    "det_rel2": 400.0,
    "rate_reference2": "At detuning",
    "rate2": 400.0,
    "k2": "x",
    "pol2": "linear z",
    "line3": "D1",
    "transition3": "2→2",
    "det_rel3": 0.0,
    "rate_reference3": "At detuning",
    "rate3": 0.0,
    "k3": "x",
    "pol3": "linear z",
    "rf_axis": "x",
    "rf_observable": "Fx",
    "rf_frequency_lower_hz": 0.0,
    "rf_frequency_upper_hz": 50.0,
    "rf_show_amplitude": True,
    "rf_show_in_phase": False,
    "rf_show_quadrature": False,
    "rf_relaxation_normalized": False,
    "show_allowed_only": True,
    "show_rate_matrices": False,
}


def clean_condition_name(value):
    """Return a nonempty condition name without a .json extension."""
    name = str(value or "").strip()
    if name.lower().endswith(".json"):
        name = name[:-5].rstrip()
    return name or "default"



def build_condition_payload(values):
    """Build the current JSON condition payload."""
    conditions = {key: values.get(key) for key in CONDITION_KEYS}
    conditions["condition_name"] = clean_condition_name(
        conditions.get("condition_name")
    )
    return {
        "app": "alkali_pumping",
        "format": "alkali_pumping_conditions",
        "version": CONDITION_SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "conditions": conditions,
    }


def current_condition_values(condition_name=None):
    """Collect all current sidebar conditions, including condition name."""
    values = {key: st.session_state.get(key) for key in CONDITION_KEYS}
    if condition_name is not None:
        values["condition_name"] = condition_name
    return values


def normalize_rf_frequency_bounds(prefer="lower"):
    """Keep the RF plotting interval nonnegative and ordered."""
    lower = max(0.0, float(st.session_state.get("rf_frequency_lower_hz", 0.0)))
    upper = max(0.0, float(st.session_state.get("rf_frequency_upper_hz", lower)))

    if lower > upper:
        if prefer == "upper":
            lower = upper
        else:
            upper = lower

    st.session_state["rf_frequency_lower_hz"] = lower
    st.session_state["rf_frequency_upper_hz"] = upper


def apply_loaded_condition_dict(payload):
    """Apply a complete condition file for the current schema."""
    if not isinstance(payload, dict):
        raise ValueError("The loaded file is not a JSON object.")
    if payload.get("app") != "alkali_pumping":
        raise ValueError("This is not an alkali_pumping condition file.")
    if payload.get("format") != "alkali_pumping_conditions":
        raise ValueError("The JSON file is not an alkali_pumping condition file.")
    if payload.get("version") != CONDITION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported condition-file version. Expected "
            f"{CONDITION_SCHEMA_VERSION}."
        )

    conditions = payload.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("The JSON file does not contain a conditions object.")

    missing = [key for key in CONDITION_KEYS if key not in conditions]
    if missing:
        raise ValueError(
            "The condition file is missing required fields: " + ", ".join(missing)
        )

    loaded_name = clean_condition_name(conditions["condition_name"])
    for key in CONDITION_KEYS:
        value = conditions[key]
        if value is not None:
            st.session_state[key] = value

    normalize_rf_frequency_bounds(prefer="lower")

    # Prevent atom-change default logic from overwriting loaded N2 coefficients.
    st.session_state["_last_atom_name_for_defaults"] = conditions["atom_name"]
    return loaded_name


def load_condition_callback():
    """Load a selected JSON in a callback, before keyed widgets are instantiated."""
    uploaded = st.session_state.get("condition_file_uploader")
    if uploaded is None:
        return

    try:
        payload = json.loads(uploaded.getvalue().decode("utf-8"))
        loaded_name = apply_loaded_condition_dict(payload)
        st.session_state["_condition_load_message"] = (
            f"Loaded condition: {loaded_name}"
        )
        st.session_state.pop("_condition_load_error", None)
    except Exception as exc:
        st.session_state["_condition_load_error"] = str(exc)
        st.session_state.pop("_condition_load_message", None)


APP_BASE_TITLE = "alkali pumping"


def current_browser_title():
    """Return the browser title from the live condition name field."""
    raw_name = st.session_state.get("condition_name", "")
    if not str(raw_name or "").strip():
        return APP_BASE_TITLE
    name = clean_condition_name(raw_name)
    return f"{APP_BASE_TITLE}: {name}"


# ============================================================
