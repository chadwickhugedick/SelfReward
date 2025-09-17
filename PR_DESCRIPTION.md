Title: Standardize environment API to Gymnasium-style reset()/step() and add legacy wrappers

Summary:
This change standardizes the `TradingEnvironment` API to follow Gymnasium conventions across the repository. The goal is to align with modern Gymnasium-based vectorized environments and make the codebase easier to integrate with other projects that use Gymnasium.

Key changes:
- `reset()` now returns `(observation, info)`.
- `step(action)` now returns `(observation, reward, terminated, truncated, info)`.
- Added `reset_legacy()` and `step_legacy(action)` on `TradingEnvironment` to assist downstream users during migration.
- Updated call-sites and tests across the repository to use Gymnasium-style signatures.

Files updated:
- `src/environment/trading_env.py` (rewritten/cleaned implementation, added `reset_legacy()` and `step_legacy()`)
- `src/environment/vectorized_env.py` (already Gymnasium-compatible, confirmed)
- `src/pretrain_reward_network.py`
- `src/evaluate.py`
- `src/train.py`
- `src/baselines/run_baselines.py`
- `test_srddqn_fix.py` (test harness)
- Various tests under `tests/` (already Gymnasium-style or adjusted)

Migration notes for downstream users:
1. Update your code to use the Gymnasium signatures:
   - `obs, info = env.reset()`
   - `obs, reward, terminated, truncated, info = env.step(action)` and derive `done = bool(terminated or truncated)` if you still rely on a single `done` boolean.
2. If you cannot migrate immediately, use the compatibility helpers:
   - `obs = env.reset_legacy()` — returns only observation
   - `obs, reward, done, info = env.step_legacy(action)` — merges `terminated`/`truncated` into `done`
3. Verify integrations (scripts, training pipelines, evaluation scripts) that call `reset()` and `step()`.

Testing:
- Ran the full test suite: `pytest -q` — all tests passed (32 passed, 4 warnings).

Suggested commit message (short):
"Standardize TradingEnvironment API to Gymnasium reset/step; add legacy helpers; update call-sites and tests"

Suggested PR description:
- Purpose: Align environment API with Gymnasium to enable better interoperability with vectorized environments and modern RL tooling.
- Scope: Environment API changes + compatibility helpers + codebase updates.
- Impact: Minor API change for downstream users; legacy wrappers provided for transition.
- Tests: Full test-suite passes locally.

If you'd like, I can open a PR branch and create the GitHub PR with this description and the changes applied. Let me know if you want that.