"""CSV export helpers for calculated app results."""

import numpy as np
import pandas as pd


def dataframe_to_csv_bytes(dataframe):
    """Serialize a dataframe as an Excel-friendly UTF-8 CSV download."""
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def weak_rf_export_dataframe(
    frequencies_hz,
    susceptibility_amplitude,
    susceptibility_in_phase,
    susceptibility_quadrature,
    plotted_amplitude,
    plotted_in_phase,
    plotted_quadrature,
    *,
    relaxation_normalized,
    normalization_gamma_s_inv=None,
):
    """Return raw and plotted weak-RF susceptibility samples for export.

    Raw susceptibility values retain the calculation's phase convention. The
    plotted signed components include the common -1 display factor, and the
    plotted values include relaxation normalization when it is active.
    """
    arrays = {
        "frequency_Hz": np.asarray(frequencies_hz, dtype=float),
        "amplitude_raw_hbar_s_per_atom": np.asarray(
            susceptibility_amplitude, dtype=float
        ),
        "in_phase_raw_hbar_s_per_atom": np.asarray(
            susceptibility_in_phase, dtype=float
        ),
        "quadrature_raw_hbar_s_per_atom": np.asarray(
            susceptibility_quadrature, dtype=float
        ),
        "amplitude_plotted": np.asarray(plotted_amplitude, dtype=float),
        "in_phase_plotted": np.asarray(plotted_in_phase, dtype=float),
        "quadrature_plotted": np.asarray(plotted_quadrature, dtype=float),
    }
    sample_counts = {len(values) for values in arrays.values()}
    if len(sample_counts) != 1:
        raise ValueError("All weak-RF export arrays must have the same length.")

    plotted_units = (
        "hbar/atom" if relaxation_normalized else "hbar s/atom"
    )
    normalization_gamma = (
        float(normalization_gamma_s_inv)
        if relaxation_normalized and normalization_gamma_s_inv is not None
        else np.nan
    )

    dataframe = pd.DataFrame(arrays)
    dataframe["plotted_units"] = plotted_units
    dataframe["signed_component_plot_factor"] = -1.0
    dataframe["relaxation_normalized"] = bool(relaxation_normalized)
    dataframe["normalization_gamma_s_inv"] = normalization_gamma
    return dataframe
