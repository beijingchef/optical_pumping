"""Physical-consistency diagnostics for density matrices and populations."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DensityMatrixDiagnostics:
    trace: complex
    trace_error: float
    hermiticity_error: float
    minimum_eigenvalue: float
    trace_ok: bool
    hermitian_ok: bool
    positive_semidefinite_ok: bool

    @property
    def valid(self):
        return self.trace_ok and self.hermitian_ok and self.positive_semidefinite_ok

    def as_dict(self):
        result = asdict(self)
        result["valid"] = self.valid
        return result


def density_matrix_diagnostics(rho, atol=1e-10):
    """Return trace, Hermiticity, and positivity diagnostics for a square matrix."""
    matrix = np.asarray(rho, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("rho must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rho must contain only finite values")

    trace = complex(np.trace(matrix))
    trace_error = float(abs(trace - 1.0))
    hermiticity_error = float(np.linalg.norm(matrix - matrix.conj().T, ord="fro"))
    hermitian_part = 0.5 * (matrix + matrix.conj().T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian_part)).real)

    return DensityMatrixDiagnostics(
        trace=trace,
        trace_error=trace_error,
        hermiticity_error=hermiticity_error,
        minimum_eigenvalue=minimum_eigenvalue,
        trace_ok=trace_error <= atol,
        hermitian_ok=hermiticity_error <= atol,
        positive_semidefinite_ok=minimum_eigenvalue >= -atol,
    )


def population_diagnostics(populations, atol=1e-10):
    """Apply density-matrix checks to the diagonal population model used by v4.23."""
    vector = np.asarray(populations, dtype=float)
    if vector.ndim != 1:
        raise ValueError("populations must be a one-dimensional vector")
    return density_matrix_diagnostics(np.diag(vector), atol=atol)

