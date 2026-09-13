# Boundary Continuity in Projected Latent State

A controlled simulation of how a latent state evolves near an explicit boundary, how hidden phase is exposed through lower-dimensional projection, and which observable structures persist through distortion and reconstruction.

<p align="center"><img src="assets/boundary_continuity_phase_reconstruction.gif" alt="Boundary continuity and phase reconstruction simulation" width="100%"></p>

> **Experiment:** When a latent dynamical state encounters a local constraint, which aspects of observable structure persist, distort, disappear, or reconstruct under projection?

The animation shows three views of the same evolving system: an observable projection with particle trajectories, a rotating projected state view, and diagnostic measures over time.

## Scope

This repository contains the executable simulation, controlled variants, visualization pipeline, and measurement tooling for a synthetic numerical experiment. Its labels and thresholds are conventions of the harness, not claims about physical systems or universal phase transitions.

## Run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python simulation.py --gif
```

Use `--frames` and `--particles` for smaller or larger exports. Controls in `controls/` change one structural assumption at a time for comparison.

## Research lineage

This artifact belongs to the Boundary Continuity / Phase Reconstruction branch of the broader DIFFRACTION work on representation, observability, and boundary-conditioned state evolution. See `docs/EXPERIMENT.md` and `docs/PROVENANCE.md` for the experiment record and provenance.

## License

MIT. Fork it, modify it, and build your own version. This repository is provided as-is and carries no maintenance or support commitment.
