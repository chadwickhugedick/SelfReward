# Changelog

## Unreleased

### Changed
- Standardized environment API to follow Gymnasium conventions:
  - `reset()` now returns `(observation, info)` instead of a single observation.
  - `step(action)` now returns `(observation, reward, terminated, truncated, info)` instead of `(observation, reward, done, info)`.

### Added
- Backwards-compatible helpers on `TradingEnvironment`:
  - `reset_legacy()` — returns only the observation for legacy callers.
  - `step_legacy(action)` — returns `(observation, reward, done, info)` where `done = terminated or truncated`.

### Notes
- Updated internal call sites, scripts, and tests to use the new Gymnasium-style API.
- If external integrations call `env.reset()` expecting a single return value or unpack `env.step()` into `(obs, reward, done, info)`, update them to the new signatures or use the legacy helpers until you migrate.

---

For migration guidance and a suggested PR description, see `PR_DESCRIPTION.md`.