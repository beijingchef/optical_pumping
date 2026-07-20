"""Compact captions for the main application outputs."""


def _number(value):
    """Format a numeric condition compactly without losing useful precision."""
    return f"{float(value):g}"


def input_conditions_caption(
    *,
    atom_name,
    temperature_C,
    n2_pressure_torr,
    R_ER,
    include_spin_exchange,
    R_SE,
    q_axis,
    bias_larmor_hz,
    beam_inputs,
):
    """Return a compact sidebar-input summary, omitting zero-rate beams."""
    spin_exchange = (
        f"SE=on (R_SE={_number(R_SE)} s⁻¹)"
        if include_spin_exchange
        else "SE=off"
    )
    cell_conditions = (
        f"Sidebar inputs — {atom_name}; T={_number(temperature_C)} °C; "
        f"N₂={_number(n2_pressure_torr)} Torr; R_ER={_number(R_ER)} s⁻¹; "
        f"{spin_exchange}; q={q_axis}; "
        f"bias Larmor={_number(bias_larmor_hz)} Hz."
    )

    active_beams = []
    for beam in beam_inputs:
        if float(beam.get("rate", 0.0)) <= 0.0:
            continue
        rate_reference = str(beam.get("rate_reference", "")).strip().lower()
        active_beams.append(
            f"{beam['name']}: {beam['line']} {beam['transition_label']}, "
            f"Δ_rel={_number(beam['detuning_relative'])} MHz, "
            f"R_pump,total={_number(beam['rate'])} s⁻¹ ({rate_reference}), "
            f"k={beam['k_axis']}, {beam['pol']}"
        )

    beam_conditions = (
        " Active pumps — " + "; ".join(active_beams) + "."
        if active_beams
        else " Active pumps — none."
    )
    return cell_conditions + beam_conditions
