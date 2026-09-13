# Experiment Notes

## State and projection

The full model evolves an ensemble of particles with latent state `(x, y, z, w)`. The directly observable readout is `(x, y)`. A second visualization rotates the state in the `x-w`, `y-w`, and `z-w` planes and displays the resulting first three coordinates.

The projection is therefore part of the experiment, not merely a camera effect: it provides a controlled way to ask how hidden-state variation appears after a representation change.

## Boundary field

For the full model, the boundary is a Gaussian ring centered at radius `r0`. Boundary strength gates nonlinear coupling and local tangential/radial terms. The random- and square-boundary controls preserve the broader experimental structure while changing the geometry of that constraint.

## Diagnostics

The harness records five primary diagnostics:

1. **Global continuity** — similarity of the current and reference radial distributions in observable `(x, y)`.
2. **Boundary distortion** — mean latent-state displacement for particles currently near the boundary.
3. **Boundary occupancy** — fraction of particles inside the boundary zone.
4. **Latent phase spread** — mean absolute magnitude of `w`.
5. **Orbit/trap fraction** — fraction of particles near the boundary whose tangential velocity dominates radial velocity by the configured ratio.

The state names shown in the animation are threshold-based summaries of these diagnostics. They are useful for navigation through a run, not independent empirical categories.

## Interpretation guardrail

The simulation can demonstrate behavior **inside its defined model** and can compare that behavior under controlled ablations. It cannot establish that visually similar structures in unrelated systems share the same mechanism. Projection-induced structure, boundary-induced structure, and model-specific numerical behavior must remain distinguishable.
