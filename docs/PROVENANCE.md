# Provenance and Packaging Notes

This public package was normalized from the supplied experimental simulation branch.

## Published surface

- `simulation.py` — final smooth/traced 4D latent-state model and GIF exporter.
- `assets/boundary_continuity_phase_reconstruction.gif` — supplied rendered animation.
- `controls/no_boundary.py` — explicit boundary removed.
- `controls/no_latent_phase.py` — active `w` phase removed.
- `controls/random_boundary.py` — randomized smooth constraint field.
- `controls/square_boundary.py` — square-shell boundary.

## Earlier development stages not included

Earlier 2D/3D particle prototypes and intermediate 4D trace versions were used during development but are not required to reproduce the final public experiment. They are omitted from the clean package to keep the repository centered on the final model and its falsification controls.

## Duplicate supplied control

The supplied files `sim_project_4d_control_no_boundary` and `sim_project_4d_control_no_coupling` were byte-for-byte identical (SHA-256 `15b5eac3129abfa6316d7b4c92dee5404afbf0a3a3786ac8c98522e4cd00e754`). The latter therefore does not constitute an independent no-coupling ablation and has been excluded rather than mislabeled.

No replacement no-coupling result was invented during packaging.
