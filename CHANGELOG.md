# Changelog

## 6.1.7 - 2026-08-08

### Fixed

- Restricted each alkali's reported light shift and light-shift diagonality
  check to pumps targeting that alkali. A nonzero Pump A can no longer blank
  Alkali B's light-shift columns, and the converse is also true.
- Preserved off-resonant optical-pumping contributions and all self/cross spin
  exchange physics; only cross-alkali AC-Stark shifts are neglected.
- Archived the previous 6.1.6 source without copying the incomplete `.venv`.

## 6.1.6 - 2026-08-08

### Fixed

- Made Alkali A/B result tabs lazy and strictly isolated. The selected tab now
  contains its own quantization-axis control, population graph, Zeeman table,
  RF settings, and RF susceptibility plot as one complete result section.
- Added persistent backing state for all lazy result-tab controls so switching
  tabs cannot reset or mix A/B quantization and RF configurations.
- Preserved the selected result tab through quantization-axis and RF-control
  reruns, including when the selected isotope changes.
- Archived the previous 6.1.5 source without copying the incomplete `.venv`.

## 6.1.5 - 2026-08-08

### Fixed

- Moved the pending full-rerun check to the end of the pump fragment so an
  intensity change reliably updates the physical solution and rate caption.
- Confirmed that, after PumpA1 or PumpA2 reaches zero intensity, direction,
  polarization, line, transition, and detuning edits remain fragment-only and
  do not rerun the full physical system.
- Archived the previous 6.1.4 source without copying the incomplete `.venv`.

## 6.1.4 - 2026-08-08

### Fixed

- Removed the `Calling st.rerun() within a callback is a no-op` warning from
  pump controls. Callbacks now set a pending full-rerun flag, and the pump
  fragment consumes that flag before requesting the supported app rerun.
- Preserved fragment-only reruns for zero-intensity beam settings and full
  physical recomputation for intensity changes or active-beam edits.
- Archived the previous 6.1.3 source without copying the incomplete `.venv`.

## 6.1.3 - 2026-08-08

### Fixed

- Kept the selected Alkali A/B result tab active when a quantization-axis or
  RF control triggers a rerun, so Alkali B controls no longer appear to alter
  or jump back to Alkali A.
- Isolated zero-intensity pump configuration edits in a sidebar fragment.
  Direction, polarization, line, transition, and detuning changes now avoid a
  full app rerun until that beam has nonzero intensity.
- Cached physical-system solutions and excluded zero-intensity beams from the
  solver input, while retaining their stored UI configuration.
- Archived the previous 6.1.2 source without copying the incomplete `.venv`.

## 6.1.2 - 2026-08-08

### Fixed

- Preserved every Alkali A and Alkali B pump setting when switching between
  the lazy pump-configuration tabs. Visible widget values are now copied into
  persistent condition state instead of being lost during hidden-widget
  cleanup.
- Kept condition-file loading and saving connected to the persistent pump
  settings while using separate temporary keys for visible tab controls.
- Archived the previous 6.1.1 application before applying this update.

## 6.1.1 - 2026-08-08

### Fixed

- Kept the Alkali A and Alkali B pump configurations in persistent keyed tabs
  when a pump transition or another tab-local setting triggers a rerun.
- Rendered only the open pump tab so the two pump configurations cannot appear
  stacked in the sidebar after a rerun.

### Changed

- Moved the `n(B) / n(A)` input onto the same sidebar row as the mixture density
  model selector.
- Archived the previous 6.1.0 application before applying this update.

## 6.1.0 - 2026-08-08

### Added

- Added a shared static-field direction and signed field strength in nT.
- Added independent Alkali A and Alkali B quantization-axis controls above
  their population and Zeeman-result regions.
- Added independent RF-A and RF-B axes, observables, frequency ranges, curve
  selections, and normalization settings beside their susceptibility plots.
- Added a coupled coherence-response generator. Each result applies only its
  own RF drive while retaining self- and cross-species spin-exchange feedback.
- Added automatic v6.0 condition migration, including conversion of the old
  A upper-manifold Larmor frequency to static-field strength.

### Changed

- Moved all RF-response controls out of the sidebar and to the left of the
  corresponding Alkali A or Alkali B susceptibility plot.
- Archived the pre-update application in `archive/alkali_pumping_v6.0.0`.

## 6.0.0 - 2026-08-07

### Added

- Added optional **Alkali B** selection beside **Alkali A**. `None` is the
  default, and a B selection identical to A is intentionally inactive.
- Added independent saturated-vapor and relative-concentration density modes.
- Added persistent sidebar pump tabs with PumpA1, PumpA2, PumpB1, and PumpB2.
- Added coupled unlike-alkali spin exchange, including two-species fixed-point
  populations and the full block small-signal population Jacobian.
- Added separate Alkali A and Alkali B result tabs when B is active.
- Added self- and cross-spin-exchange contributions to the Zeeman table.
- Added v5.0 condition migration to the v6.0 condition schema.

### Changed

- Retired the third A pump. PumpA1 and PumpA2 default to 5.0 µW/cm²; PumpB1
  and PumpB2 default to 0.0 µW/cm².
- Every active pump is evaluated at its absolute optical frequency for both
  species, including isotope cross-pumping.
- Rate-matrix output now includes the coupled A/B Jacobian and its local maps.
- Archived the pre-upgrade application in `archive/alkali_pumping_v5.2.16`.

## 5.2.16 - 2026-07-23

### Added

- Added a beam-intensity input in µW/cm² for each of the three pump beams.
- Added calculated sidebar captions for the selected ground manifold's total
  pump rate at the reference resonance and at the specified detuning.
- Added automatic conversion of legacy v5 rate-referenced condition files to
  the equivalent physical beam intensities.

### Changed

- Replaced the pump-rate reference selector and pump-rate input with an
  absolute weak-light rate calculation based on photon flux, D-line
  wavelength, natural linewidth, and pressure/Doppler broadening.

## 5.2.15 - 2026-07-21

### Added

- Added a fully commented `.streamlit/config.toml` template for future native
  Streamlit theme, font, color, border, and sidebar customization.

### Changed

- Documented the theme configuration file in the README.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.14`.

## 5.2.14 - 2026-07-21

### Changed

- Moved the calculated `R_SE` caption directly below the **Include spin
  exchange** checkbox.
- Removed alkali density and spin-exchange cross-section values from that
  sidebar caption.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.13`.

## 5.2.13 - 2026-07-21

### Changed

- Removed all injected sidebar widget CSS, including custom heights, fonts,
  file-uploader sizing, and number-input step-button sizing.
- Restored Streamlit's native appearance and dimensions for every sidebar
  input, selection, checkbox, uploader, and button.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.12`.

## 5.2.12 - 2026-07-21

### Changed

- Removed the editable N₂ pressure-broadening and shift coefficient section
  from the sidebar while retaining the stored coefficients in calculations and
  condition files.
- Set every sidebar input field and selection box to a uniform 25-pixel height.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.11`.

## 5.2.11 - 2026-07-21

### Changed

- Made Streamlit's native decrease and increase controls visible for N₂
  pressure, temperature, and every beam's detuning and pump-rate fields.
- Reduced the requested native step-button width and input height for a compact
  sidebar appearance.
- Moved N₂ pressure and temperature into a two-column row so each field is
  wide enough for Streamlit to render its native controls.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.10`.

## 5.2.10 - 2026-07-21

### Changed

- Changed the fresh-app pump-rate reference default to **At resonance** for
  all three beams.
- Updated the bundled default condition accordingly; explicitly loaded saved
  conditions continue to retain their own pump-rate-reference choices.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.9`.

## 5.2.9 - 2026-07-21

### Added

- Added vector light shift $\nu^{\mathrm{VS}}$ and tensor light shift
  $\nu^{\mathrm{TS}}$ columns immediately before the total light shift in the
  Zeeman-sublevel properties table and its CSV export.
- Decomposed each diagonal total AC-Stark shift within its hyperfine manifold
  into scalar, vector, and tensor state contributions.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.8`.

## 5.2.8 - 2026-07-21

### Changed

- Shortened the Grotrian hyperfine-level labels to `F=…` and `F′=…`.
- Right-aligned each label with a fixed gap to the left of its nearest Zeeman
  level segment, preventing overlap with the level-denoting line.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.7`.

## 5.2.7 - 2026-07-21

### Changed

- Increased the Grotrian stacked-fraction annotation font from 10 to 15
  points, exactly 50%, while leaving magnetic-quantum-number labels unchanged.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.6`.

## 5.2.6 - 2026-07-21

### Changed

- Replaced linear transition-strength labels such as `1/12` with vertically
  stacked MathText fractions.
- Applied the same vertical fraction format to the summed displayed strength.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.5`.

## 5.2.5 - 2026-07-21

### Changed

- Lowered the common σ−/σ+ transition-strength label level from 0.40 to 0.30
  while retaining exact placement on each slanted transition line.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.4`.

## 5.2.4 - 2026-07-21

### Changed

- Raised all π-transition strength labels to a common upper level.
- Placed σ− and σ+ strength labels at the same lower level and calculated each
  horizontal label coordinate from its transition line, keeping every label
  centered on the corresponding slanted line.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.3`.

## 5.2.3 - 2026-07-21

### Changed

- Increased Grotrian fractional-strength annotations to the same 10-point
  font size as the magnetic-quantum-number labels.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.2`.

## 5.2.2 - 2026-07-21

### Changed

- Moved every Grotrian-diagram control into a vertical panel to the left of
  the graph.
- Made every polarization and display option checked by default, including
  transition-strength labels.
- Changed individual and summed transition-strength labels from decimals to
  reduced fractions.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.1`.

## 5.2.1 - 2026-07-21

### Fixed

- Moved the atomic-properties temperature slider inside the first tab.
- Decoupled the buffer-gas tab from that slider by reporting its calculated
  electron-randomization rates at a stated 20 °C reference temperature.
- Made the modal tabs stateful and render only the active tab, so a
  Grotrian-control rerun preserves the third tab without replaying the first
  two tabs above it.
- Archived the pre-update application in `archive/alkali_pumping_v5.2.0`.

## 5.2.0 - 2026-07-21

### Added

- Added a **Settings** button that opens a large modal atomic-properties dialog.
- Added isotope selection for ²³Na, ³⁹K, ⁴¹K, ⁸⁵Rb, ⁸⁷Rb, and ¹³³Cs.
- Added temperature-dependent saturated-vapor density, RMS velocity, mean
  relative velocity, and self spin-exchange rate calculations.
- Added optical pressure broadening and shift tables for N₂, ⁴He, and CH₄,
  together with editable ground-state electron-randomization cross sections
  and calculated collision rates.
- Added an interactive hyperfine–Zeeman Grotrian diagram whose selection rules,
  colors, and line strengths reproduce the supplied Mathematica notebook.
- Archived the pre-update application in `archive/alkali_pumping_v5.1.0`.

- Added an optional **Density factor** for the weak-RF susceptibility plot.
  It multiplies every plotted component by the calculated saturated alkali
  vapor density in cm⁻³.
- Added density-factor status, density, and resulting plotted units to the
  weak-RF CSV export.
- Added backward-compatible loading of v5.0 condition files that predate the
  density-factor field; the new option defaults to off for those files.

### Changed

- Replaced the RF relaxation-normalization caption with an always-visible
  summary of scientific sidebar inputs and active pump beams.
- Shortened the pump-rate input label to **Pump rate** while retaining the
  total Zeeman-summed definition introduced in v5.1.

## 5.1.0 - 2026-07-20

### Changed

- Changed the pump-rate input from an average over ground Zeeman sublevels to
  the total selected-transition rate summed over ground and excited Zeeman
  sublevels.
- Updated the pump-rate documentation, normalization tests, and application
  metadata for the new definition.
