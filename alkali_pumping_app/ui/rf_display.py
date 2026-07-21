"""Display-only transformations for weak-RF susceptibility curves."""

import numpy as np


def prepare_weak_rf_plot_values(
    amplitude,
    in_phase,
    quadrature,
    *,
    relaxation_gamma_s_inv=None,
    density_cm3=None,
):
    """Apply the plot sign convention and optional scalar factors."""
    factor = 1.0
    if relaxation_gamma_s_inv is not None:
        factor *= float(relaxation_gamma_s_inv)
    if density_cm3 is not None:
        factor *= float(density_cm3)

    plotted_amplitude = np.asarray(amplitude, dtype=float).copy() * factor
    plotted_in_phase = -np.asarray(in_phase, dtype=float).copy() * factor
    plotted_quadrature = -np.asarray(quadrature, dtype=float).copy() * factor
    return plotted_amplitude, plotted_in_phase, plotted_quadrature
