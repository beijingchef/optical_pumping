# Alkali Pumping v5.2

This is the modular successor to `alkali_pumping.py` v4.23. The pump-rate
input is defined as the selected-transition absorption rate summed over all
ground and excited Zeeman sublevels. It is not averaged over the ground
hyperfine manifold.

Run from this directory:

```powershell
streamlit run alkali_pumping.py
```

The condition-file schema is version 5.0. Condition files must contain every
current field; legacy condition-file compatibility is intentionally not provided.
App/package metadata is versioned as 5.2.14. The condition-file field layout
remains schema version 5.0.

See `CHANGELOG.md` for user-visible and physics-model updates.

## Layout

- `alkali_pumping.py`: direct Streamlit page script, preserving top-to-bottom reruns.
- `alkali_pumping_app/physics/`: numerical model and validation helpers.
- `alkali_pumping_app/ui/`: Streamlit, condition-file, and table rendering code.
- `tests/`: foundation and physical-consistency tests.
- `alkali_pumping_v4_23.py`: retained source snapshot for comparison only.

Run the tests with:

```powershell
python -m unittest discover -s tests -v
```
