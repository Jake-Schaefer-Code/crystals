# Devlog

## 2026-04-29

- Added direct complex finite-dimensional representation support to `src/symmetry.py`.
- Implemented `Representation` validation plus character, projector, and Reynolds-projector helpers for ordered matrix data.
- Stopped `AffineOperation` from discarding complex entries so `FiniteGroupAction.linear_projector(...)` now works for complex linear actions with zero translation, including the induced `Hom(C^2_σ, C^2_τ)` `Z/4` example.
