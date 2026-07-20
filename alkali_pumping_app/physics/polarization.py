"""Beam geometry and polarization-basis transformations."""

from math import sqrt

import numpy as np

# ============================================================
# 4. Light geometry and polarization
# ============================================================

def unit_vector(axis):
    if axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=complex)
    if axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=complex)
    if axis == "z":
        return np.array([0.0, 0.0, 1.0], dtype=complex)
    raise ValueError(axis)


def transverse_basis_for_k(k_axis):
    """
    Return two transverse unit vectors a,b such that a x b = k.
    This defines the local beam frame.
    """
    if k_axis == "z":
        return unit_vector("x"), unit_vector("y")
    if k_axis == "x":
        return unit_vector("y"), unit_vector("z")
    if k_axis == "y":
        return unit_vector("z"), unit_vector("x")
    raise ValueError(k_axis)


def allowed_polarizations(k_axis):
    """
    For each propagation direction, allow sigma+, sigma-, and the two lab-linear
    directions perpendicular to k.
    """
    if k_axis == "z":
        return ["sigma+", "sigma-", "linear x", "linear y"]
    if k_axis == "x":
        return ["sigma+", "sigma-", "linear y", "linear z"]
    if k_axis == "y":
        return ["sigma+", "sigma-", "linear z", "linear x"]
    raise ValueError(k_axis)


def lab_e_field(k_axis, pol):
    """
    Complex electric field in lab x,y,z coordinates.

    Convention:
      If k = z and quantization axis = z, sigma+ gives q=+1.
    """
    a, b = transverse_basis_for_k(k_axis)

    if pol == "sigma+":
        return -(a + 1j * b) / sqrt(2)
    if pol == "sigma-":
        return (a - 1j * b) / sqrt(2)
    if pol.startswith("linear"):
        ax = pol.split()[-1]
        return unit_vector(ax)

    raise ValueError(pol)


def local_components(E_lab, q_axis):
    """
    Components in a local frame with local z along the chosen quantization axis.
    """
    if q_axis == "z":
        ux, uy, uz = unit_vector("x"), unit_vector("y"), unit_vector("z")
    elif q_axis == "x":
        ux, uy, uz = unit_vector("y"), unit_vector("z"), unit_vector("x")
    elif q_axis == "y":
        ux, uy, uz = unit_vector("z"), unit_vector("x"), unit_vector("y")
    else:
        raise ValueError(q_axis)

    return np.array([
        np.vdot(ux, E_lab),
        np.vdot(uy, E_lab),
        np.vdot(uz, E_lab),
    ], dtype=complex)


def spherical_weights_relative_to_quant_axis(k_axis, pol, q_axis):
    """
    Return |E_q|^2 for q=-1,0,+1 relative to the selected quantization axis.
    """
    E_lab = lab_e_field(k_axis, pol)
    Ex, Ey, Ez = local_components(E_lab, q_axis)

    E_plus = -(Ex - 1j * Ey) / sqrt(2)
    E_zero = Ez
    E_minus = (Ex + 1j * Ey) / sqrt(2)

    weights = {
        -1: abs(E_minus)**2,
        0: abs(E_zero)**2,
        +1: abs(E_plus)**2,
    }

    s = sum(weights.values())
    if s <= 0:
        return {-1: 0.0, 0: 0.0, +1: 0.0}
    return {q: float(w / s) for q, w in weights.items()}

