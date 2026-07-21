# Changelog

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
