"""Modal atomic-properties explorer used by the page Settings button."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D

from ..physics.atomic_properties import (
    ATOMIC_PROPERTY_DATA,
    BUFFER_GASES,
    ELECTRON_RANDOMIZATION_CROSS_SECTION_CM2,
    PRESSURE_COEFFICIENTS_MHZ_TORR,
    alkali_thermal_properties,
    allowed_hyperfine_F,
    atomic_property_record,
    buffer_gas_collision_rate_s,
    format_transition_strength_vertical_fraction,
    grotrian_transitions,
    magnetic_sublevels,
)


def _format_F(value):
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))
    return f"{int(round(2 * value))}/2"


def _transition_label_position(mg, me, q, line_bottom, line_top):
    """Return a label point that lies exactly on its transition line."""
    label_y = 0.62 if q == 0 else 0.30
    line_fraction = (label_y - line_bottom) / (line_top - line_bottom)
    label_x = mg + line_fraction * (me - mg)
    return label_x, label_y


def _grotrian_figure(isotope, line, Fg, Fe, polarizations, scale_width, labels):
    atom = atomic_property_record(isotope)
    transitions = grotrian_transitions(atom["I"], line, Fg, Fe, polarizations)
    ground_m = magnetic_sublevels(Fg)
    excited_m = magnetic_sublevels(Fe)
    x_min = min(ground_m + excited_m) - 0.65
    x_max = max(ground_m + excited_m) + 0.65

    fig, axis = plt.subplots(figsize=(9.0, 5.4))
    quantum_number_fontsize = 10
    fraction_fontsize = 1.5 * quantum_number_fontsize
    transition_line_bottom = 0.055
    transition_line_top = 0.945
    level_half_width = 0.17
    level_label_gap = 0.10
    for m in ground_m:
        axis.plot([m - level_half_width, m + level_half_width], [0, 0], color="black", lw=2)
    for m in excited_m:
        axis.plot([m - level_half_width, m + level_half_width], [1, 1], color="black", lw=2)

    colors = {-1: "#2575d8", 0: "#1f9d55", 1: "#d64545"}
    max_strength = max((row["strength"] for row in transitions), default=1.0)
    for row in transitions:
        normalized = row["strength"] / max_strength
        linewidth = 0.8 + (5.5 * normalized if scale_width else 1.4)
        axis.plot(
            [row["mg"], row["me"]],
            [transition_line_bottom, transition_line_top],
            color=colors[row["q"]],
            lw=linewidth,
            alpha=0.82,
            solid_capstyle="round",
        )
        if labels:
            label_x, label_y = _transition_label_position(
                row["mg"],
                row["me"],
                row["q"],
                transition_line_bottom,
                transition_line_top,
            )
            axis.text(
                label_x,
                label_y,
                format_transition_strength_vertical_fraction(row["strength"]),
                fontsize=fraction_fontsize,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1},
            )

    ground_label_x = min(ground_m) - level_half_width - level_label_gap
    excited_label_x = min(excited_m) - level_half_width - level_label_gap
    axis.text(
        ground_label_x,
        0,
        f"F={_format_F(Fg)}",
        ha="right",
        va="center",
        fontsize=10,
    )
    axis.text(
        excited_label_x,
        1,
        f"F′={_format_F(Fe)}",
        ha="right",
        va="center",
        fontsize=10,
    )
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(-0.16, 1.16)
    axis.set_xlabel("Magnetic quantum number m", fontsize=quantum_number_fontsize)
    axis.tick_params(axis="x", labelsize=quantum_number_fontsize)
    axis.set_yticks([])
    axis.set_title(f'{atom["label"]} {line} hyperfine–Zeeman transition strengths')
    axis.spines[["left", "right", "top"]].set_visible(False)
    axis.grid(axis="x", alpha=0.16)
    legend_handles = [
        Line2D([0], [0], color=colors[-1], lw=3, label="σ−  (Δm = −1)"),
        Line2D([0], [0], color=colors[0], lw=3, label="π  (Δm = 0)"),
        Line2D([0], [0], color=colors[1], lw=3, label="σ+  (Δm = +1)"),
    ]
    axis.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout()
    return fig, transitions


def _thermal_tab(isotope):
    st.session_state.setdefault(
        "atomic_properties_temperature_C",
        float(st.session_state.get("temperature_C_for_table", 23.5)),
    )
    temperature_C = st.slider(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=300.0,
        step=0.5,
        key="atomic_properties_temperature_C",
    )
    atom = atomic_property_record(isotope)
    values = alkali_thermal_properties(isotope, temperature_C)
    st.metric("Self spin-exchange cross section", f'{atom["spin_exchange_cross_section_cm2"]:.2e} cm²')
    row1 = st.columns(2)
    row1[0].metric("Saturated-vapor number density", f'{values["density_cm3"]:.4e} cm⁻³')
    row1[1].metric("Self spin-exchange rate", f'{values["spin_exchange_rate_s"]:.4e} s⁻¹')
    row2 = st.columns(2)
    row2[0].metric("RMS atomic velocity", f'{values["rms_velocity_m_s"]:.3f} m/s')
    row2[1].metric("Mean relative velocity", f'{values["mean_relative_velocity_m_s"]:.3f} m/s')
    st.caption(
        f'Saturated vapor is treated as the {values["phase"]} phase; '
        f'calculated vapor pressure = {values["pressure_torr"]:.4e} Torr. '
        "The rate is nσv̄rel."
    )


def _buffer_tab(isotope):
    atom = atomic_property_record(isotope)
    element = atom["element"]
    reference_temperature_C = 20.0
    line = st.radio("Optical transition", ["D1", "D2"], horizontal=True, key="atomic_properties_line")
    st.caption(
        "H4 is interpreted here as helium-4 (⁴He). Broadening is Lorentzian FWHM. "
        "Electron-randomization rates are quoted at the independent 20 °C reference "
        "temperature. Cross sections are editable; 0 means no built-in value is available."
    )

    columns = st.columns(3)
    rows = []
    missing_er = []
    for column, gas_name in zip(columns, ("N2", "He4", "CH4")):
        gas = BUFFER_GASES[gas_name]
        pressure_key = f"atomic_pressure_{gas_name}_torr"
        sigma_key = f"atomic_er_sigma_{isotope}_{gas_name}"
        st.session_state.setdefault(pressure_key, 0.0)
        reference_sigma = ELECTRON_RANDOMIZATION_CROSS_SECTION_CM2[element][gas_name]
        st.session_state.setdefault(sigma_key, reference_sigma or 0.0)
        with column:
            pressure = st.number_input(
                f'{gas["label"]} pressure (Torr)',
                min_value=0.0,
                step=1.0,
                key=pressure_key,
            )
            sigma = st.number_input(
                f'{gas["label"]} σER (cm²)',
                min_value=0.0,
                step=1e-24,
                format="%.4e",
                key=sigma_key,
            )
        coefficient = PRESSURE_COEFFICIENTS_MHZ_TORR.get(element, {}).get(line, {}).get(gas_name)
        broadening = coefficient[0] * pressure if coefficient else None
        shift = coefficient[1] * pressure if coefficient else None
        randomization_rate = buffer_gas_collision_rate_s(
            isotope, gas_name, reference_temperature_C, pressure, sigma
        )
        if reference_sigma is None:
            missing_er.append(gas["label"])
        rows.append(
            {
                "Buffer gas": gas["label"],
                "Pressure (Torr)": pressure,
                "FWHM coeff. (MHz/Torr)": coefficient[0] if coefficient else None,
                "Broadening (MHz)": broadening,
                "Shift coeff. (MHz/Torr)": coefficient[1] if coefficient else None,
                "Shift (MHz)": shift,
                "Electron randomization at 20 °C (s⁻¹)": randomization_rate,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if missing_er:
        st.info(
            "No built-in ground-state electron-randomization cross section is available "
            f'for {atom["label"]} with {", ".join(missing_er)}. Enter a measured value '
            "above to calculate its rate."
        )
    st.caption(
        "A blank table entry means that this compact reference set has no verified "
        "pressure coefficient for that atom, line, and gas."
    )


def _transition_tab(isotope):
    atom = atomic_property_record(isotope)
    control_panel, graph_panel = st.columns([0.28, 0.72], gap="large")
    with control_panel:
        st.markdown("#### Grotrian controls")
        line = st.selectbox("Fine-structure line", ["D1", "D2"], key="grotrian_line")
        Je = 0.5 if line == "D1" else 1.5
        ground_F = allowed_hyperfine_F(atom["I"], 0.5)
        excited_F = allowed_hyperfine_F(atom["I"], Je)
        if st.session_state.get("grotrian_Fg") not in ground_F:
            st.session_state["grotrian_Fg"] = ground_F[0]
        if st.session_state.get("grotrian_Fe") not in excited_F:
            st.session_state["grotrian_Fe"] = excited_F[0]
        Fg = st.selectbox("Ground F", ground_F, format_func=_format_F, key="grotrian_Fg")
        Fe = st.selectbox("Excited F′", excited_F, format_func=_format_F, key="grotrian_Fe")
        st.markdown("**Polarization**")
        sigma_minus = st.checkbox("σ−", value=True, key="grotrian_sigma_minus_v522")
        pi_enabled = st.checkbox("π", value=True, key="grotrian_pi_v522")
        sigma_plus = st.checkbox("σ+", value=True, key="grotrian_sigma_plus_v522")
        st.markdown("**Display**")
        scale_width = st.checkbox(
            "Scale line width by strength",
            value=True,
            key="grotrian_scale_width_v522",
        )
        labels = st.checkbox(
            "Show fractional strengths",
            value=True,
            key="grotrian_labels_v522",
        )
    selected = []
    if sigma_minus:
        selected.append(-1)
    if pi_enabled:
        selected.append(0)
    if sigma_plus:
        selected.append(1)
    with graph_panel:
        figure, transitions = _grotrian_figure(
            isotope, line, Fg, Fe, selected, scale_width, labels
        )
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        summed_strength = sum(row["strength"] for row in transitions)
        st.caption(
            f'{len(transitions)} allowed transition(s); summed displayed strength = '
            f'{format_transition_strength_vertical_fraction(summed_strength)}. '
            "Colors and strength normalization follow the supplied Mathematica notebook."
        )


@st.dialog("Atomic properties", width="large")
def atomic_properties_dialog():
    """Render the isotope selector and three atomic-property tabs."""
    isotope_names = list(ATOMIC_PROPERTY_DATA)
    st.session_state.setdefault("atomic_properties_isotope", "Rb87")
    isotope = st.selectbox(
        "Alkali isotope",
        isotope_names,
        format_func=lambda name: ATOMIC_PROPERTY_DATA[name]["label"],
        key="atomic_properties_isotope",
    )
    atom = atomic_property_record(isotope)
    st.caption(
        f'Atomic mass {atom["mass_amu"]:.10g} u · nuclear spin I = {_format_F(atom["I"])}'
    )
    thermal_tab, buffer_tab, transition_tab = st.tabs(
        ["Thermal & spin exchange", "Buffer-gas collisions", "Transition strengths"],
        key="atomic_properties_tab",
        on_change="rerun",
    )
    if thermal_tab.open:
        with thermal_tab:
            _thermal_tab(isotope)
    elif buffer_tab.open:
        with buffer_tab:
            _buffer_tab(isotope)
    elif transition_tab.open:
        with transition_tab:
            _transition_tab(isotope)
    st.caption(
        "Reference data: [D. A. Steck alkali data](https://steck.us/alkalidata/); "
        "[Happer, Jau & Walker, *Optically Pumped Atoms*]"
        "(https://doi.org/10.1002/9783527629646); "
        "[Pitz et al. pressure coefficients]"
        "(https://www.sciencedirect.com/science/article/pii/S0022407311004195); "
        "and the supplied transition-strength notebook."
    )
