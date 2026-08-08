# Alkali Pumping v6.1

This Streamlit application models optical pumping, electron randomization,
self spin exchange, and unlike-alkali spin exchange in a one- or two-alkali
vapor.

Run from this directory:

```powershell
streamlit run alkali_pumping.py
```

## Dual-alkali behavior

- **Alkali A** is always active. **Alkali B** defaults to `None`.
- Selecting the same isotope for A and B leaves B inactive and ignores its
  pump settings.
- Selecting different isotopes enables a coupled A/B steady-state solve and
  separate result tabs.
- Density can use independent saturated-vapor curves or a user-entered
  relative concentration, `n(B)/n(A)`.
- PumpA1, PumpA2, PumpB1, and PumpB2 are configured in persistent sidebar
  tabs. Every active laser frequency is evaluated against both species.
- The optional rate-matrix display includes each local map and the full block
  population Jacobian `[[J_AA, J_AB], [J_BA, J_BB]]`.
- The sidebar contains the shared static-field direction and strength in nT.
- Each result tab has its own quantization axis and RF susceptibility controls.
- RF-A and RF-B have independent frequency ranges. In the A tab only RF-A is
  applied and all observables belong to A; in the B tab only RF-B is applied
  and all observables belong to B.
- The dual-alkali RF solver retains coherence feedback through self and cross
  spin exchange even though the other RF drive is zero.

The condition-file schema is version 6.1. Complete v6.1 files are required;
v6.0 and v5.0 files are migrated automatically. Legacy Pump1 and Pump2 become
PumpA1 and PumpA2, while the retired Pump3 is discarded. A legacy A Larmor
frequency is converted to the corresponding static field in nT.

The population model is diagonal in each selected quantization basis. When a
nonzero static-field direction is transverse to a tab's quantization axis, the
app warns that transverse static-field mixing is omitted and does not present
that RF curve. Choose the static-field direction as the quantization axis for
the supported secular weak-response calculation.

## Unlike-alkali spin-exchange defaults

| Pair | Cross section (cm²) |
| --- | ---: |
| Rb85–Rb87 | 1.70 × 10⁻¹⁴ |
| Rb–Cs | 2.30 × 10⁻¹⁴ |
| K–Rb | 2.00 × 10⁻¹⁴ |
| K–Cs | 2.24 × 10⁻¹⁴ |

The Rb-isotope value follows [Jarrett, *Phys. Rev.* 133, A111
(1964)](https://doi.org/10.1103/PhysRev.133.A111), and the Rb–Cs value follows
[Gibbs and Hull, *Phys. Rev.* 153, 132
(1967)](https://doi.org/10.1103/PhysRev.153.132). The K–Rb value is the
approximately 200 Å² hybrid-vapor convention; the K–Cs value converts the
approximately 800 a₀² thermal result from [Kartoshkin, *Optics and
Spectroscopy* 113, 235
(2012)](https://doi.org/10.1134/S0030400X12090081). Rates use the Maxwellian
mean relative speed and the collision partner's number density.

Native Streamlit theme settings can be added in `.streamlit/config.toml`.
See `CHANGELOG.md` for user-visible and physics-model updates.

## Layout

- `alkali_pumping.py`: direct Streamlit page script.
- `alkali_pumping_app/physics/`: single- and dual-species numerical models.
- `alkali_pumping_app/ui/`: condition-file and table-rendering helpers.
- `tests/`: regression and physical-consistency tests.
- `alkali_pumping_v4_23.py`: retained source snapshot for comparison only.
- `../archive/alkali_pumping_v5.2.16/`: pre-v6 archived application.

Run the tests with:

```powershell
python -m unittest discover -s tests -v
```
