"""HTML and dataframe presentation helpers."""

import numpy as np
import pandas as pd

def apply_two_line_column_headers(df, header_map):
    """Use real pandas MultiIndex headers so Streamlit shows units on a second header row."""
    display_df = df.copy()
    display_df.columns = pd.MultiIndex.from_tuples([
        header_map.get(col, (str(col), "")) for col in display_df.columns
    ])
    return display_df


def render_transition_table_html(df):
    """Render transition table with controlled two-line headers and HTML subscripts."""
    import html

    columns = [
        ("Line", "Line", None, "text"),
        ("Fg", "F<sub>g</sub>", None, "text"),
        ("F'", "F′", None, "text"),
        ("nu_D_absolute", "ν<sub>D</sub> absolute", "MHz", "1f"),
        ("detuning_zero_pressure", "ν<sub>FF′</sub> − ν<sub>D</sub>", "MHz", "1f"),
        ("N2_shift", "N<sub>2</sub> shift", "MHz", "1f"),
        ("transition_frequency_with_N2", "ν<sub>FF′</sub>, with N<sub>2</sub>", "MHz", "1f"),
        ("pump_1_frequency", "ν<sub>pump1</sub>", "MHz", "1f"),
        ("pump_2_frequency", "ν<sub>pump2</sub>", "MHz", "1f"),
        ("pump_3_frequency", "ν<sub>pump3</sub>", "MHz", "1f"),
        ("lorentz_FWHM_total", "Lorentz FWHM total", "MHz", "1f"),
        ("doppler_FWHM", "Doppler FWHM", "MHz", "1f"),
    ]

    def fmt_value(value, kind):
        if pd.isna(value):
            return ""
        if kind == "text":
            return html.escape(str(value))
        if kind == "9f":
            return f"{float(value):.9f}"
        if kind == "1f":
            return f"{float(value):.1f}"
        return html.escape(str(value))

    header_cells = []
    for _col, title, unit, _kind in columns:
        if unit:
            header_cells.append(
                f"<th><div class='transition-header-quantity'>{title}</div><div class='transition-header-unit'>({html.escape(unit)})</div></th>"
            )
        else:
            header_cells.append(f"<th><div class='transition-header-quantity'>{title}</div></th>")

    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col, _title, _unit, kind in columns:
            cells.append(f"<td>{fmt_value(row[col], kind)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
<style>
.transition-table-wrap {{
    max-height: 420px;
    overflow: auto;
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.35rem;
}}
.transition-table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.86rem;
}}
.transition-table th {{
    position: sticky;
    top: 0;
    background: var(--background-color, #ffffff);
    color: var(--text-color, #31333f);
    z-index: 1;
    border-bottom: 1px solid rgba(128, 128, 128, 0.45);
    padding: 0.35rem 0.45rem;
    text-align: left;
    white-space: nowrap;
}}
.transition-table td {{
    border-bottom: 1px solid rgba(49, 51, 63, 0.12);
    padding: 0.30rem 0.45rem;
    white-space: nowrap;
    text-align: right;
}}
.transition-table td:first-child,
.transition-table td:nth-child(2),
.transition-table td:nth-child(3) {{
    text-align: left;
}}
.transition-table .transition-header-quantity {{
    line-height: 1.15;
    color: inherit !important;
    -webkit-text-fill-color: currentColor !important;
    opacity: 1 !important;
    visibility: visible !important;
}}
.transition-table .transition-header-unit {{
    line-height: 1.15;
    font-size: 0.78rem;
    font-weight: 400;
    color: var(--text-color, #31333f);
    opacity: 0.72;
}}
</style>
<div class='transition-table-wrap'>
<table class='transition-table'>
<thead><tr>{''.join(header_cells)}</tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table>
</div>
"""


def render_zeeman_properties_table_html(df):
    """Render the Zeeman-sublevel table with a guaranteed hyperfine separator.

    A custom HTML table is used because Streamlit's dataframe renderer does not
    reliably preserve pandas Styler border rules. The first row of every new F
    manifold receives a 2 px top border, making the boundary between
    F=I-1/2 and F=I+1/2 clearly visible. Column widths are determined
    automatically from the rendered header and cell contents.
    """
    import html

    columns = [
        ("F", "F", None, "g"),
        ("m", "m", None, "g"),
        ("P_F", "P<sub>F</sub>", None, ".3f"),
        ("Pₘ", "P<sub>m</sub>", None, ".3f"),
        ("Dₘ", "D<sub>m</sub>", None, ".3f"),
        ("ν^{LS} (Hz)", "ν<sup>LS</sup>", "Hz", ".1f"),
        ("ν^{B} (Hz)", "ν<sup>B</sup>", "Hz", ".1f"),
        ("ν_m (Hz)", "ν<sub>m</sub>", "Hz", ".1f"),
        ("Λ (s⁻¹)", "Λ", "s<sup>−1</sup>", ".1f"),
        ("G^{OP} (s^-1)", "G<sup>OP</sup>", "s<sup>−1</sup>", ".1f"),
        ("Γ^{OP} (s^-1)", "Γ<sup>OP</sup>", "s<sup>−1</sup>", ".1f"),
        ("Γ^{OP}/2π (Hz)", "Γ<sup>OP</sup>/2π", "Hz", ".1f"),
        ("G^{ER} (s^-1)", "G<sup>ER</sup>", "s<sup>−1</sup>", ".2f"),
        ("G^{SE} (s^-1)", "G<sup>SE</sup>", "s<sup>−1</sup>", ".2f"),
        ("Γ^{ER} (s^-1)", "Γ<sup>ER</sup>", "s<sup>−1</sup>", ".2f"),
        ("Γ^{SE} (s^-1)", "Γ<sup>SE</sup>", "s<sup>−1</sup>", ".2f"),
    ]

    def fmt(value, spec):
        if pd.isna(value):
            return ""
        if spec == "g":
            return f"{float(value):g}"
        return format(float(value), spec)

    header_cells = []
    for _key, title, unit, _spec in columns:
        unit_html = f"<div class='zeeman-header-unit'>({unit})</div>" if unit else ""
        header_cells.append(
            f"<th><div class='zeeman-header-quantity'>{title}</div>{unit_html}</th>"
        )
    headers = "".join(header_cells)

    body_rows = []
    previous_F = None
    for _, row in df.iterrows():
        current_F = float(row["F"])
        separator = previous_F is not None and not np.isclose(current_F, previous_F)
        row_class = " class='hyperfine-separator'" if separator else ""
        cells = "".join(
            f"<td>{html.escape(fmt(row[key], spec))}</td>"
            for key, _title, _unit, spec in columns
        )
        body_rows.append(f"<tr{row_class}>{cells}</tr>")
        previous_F = current_F

    return f"""
<style>
.zeeman-properties-wrap {{
    max-height: 315px;
    overflow: auto;
    border: 1px solid rgba(49, 51, 63, 0.20);
    border-radius: 0.35rem;
}}
.zeeman-properties-table {{
    border-collapse: collapse;
    width: max-content;
    min-width: 0;
    table-layout: auto;
    font-size: 0.86rem;
}}
.zeeman-properties-table th {{
    position: sticky;
    top: 0;
    z-index: 2;
    background: var(--background-color, #ffffff);
    color: var(--text-color, #31333f);
    border-bottom: 1px solid rgba(128, 128, 128, 0.50);
    padding: 0.35rem 0.45rem;
    text-align: right;
    white-space: nowrap;
}}
.zeeman-properties-table th .zeeman-header-quantity {{
    line-height: 1.15;
    color: inherit !important;
    -webkit-text-fill-color: currentColor !important;
    opacity: 1 !important;
    visibility: visible !important;
}}
.zeeman-properties-table th .zeeman-header-unit {{
    line-height: 1.15;
    font-size: 0.78rem;
    font-weight: 400;
    color: var(--text-color, #31333f);
    opacity: 0.72;
}}
.zeeman-properties-table td {{
    border-bottom: 1px solid rgba(49, 51, 63, 0.12);
    padding: 0.30rem 0.45rem;
    text-align: right;
    white-space: nowrap;
}}
.zeeman-properties-table tr.hyperfine-separator td {{
    border-top: 2px solid rgba(49, 51, 63, 0.90) !important;
}}
</style>
<div class="zeeman-properties-wrap">
<table class="zeeman-properties-table">
<thead><tr>{headers}</tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table>
</div>
"""

