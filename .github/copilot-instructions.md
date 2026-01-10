# GitHub Copilot / AI Agent Instructions for Bi-Pedal-Walker ✅

Purpose: Short, actionable guidance to help AI coding agents be immediately productive in this repo. Focus is on training/eval workflows for BipedalWalker (Hardcore), reproducibility, and common pitfalls observed in the notebooks and scripts.

---

## Quick start (what to run)
- Install dependencies: run the repository's helper: `python dependencies.py` (installs gymnasium[box2d], stable-baselines3, torch, numpy, matplotlib, etc.).
- Reproduce a training run: open `walker_hardcore_3.ipynb` and run the cells. Key training blocks are labeled `# --- TRAINING LOOP ---` and `# === TRAIN ===` (PPO / SB3 flows also exist in the notebook).
- Run a short smoke test: set `N_ENVS = 1` and `TOTAL_TIMESTEPS = 10_000` (or tune `START_STEPS` in the SAC cell) to validate changes quickly.

---

## Important repo-specific patterns & conventions
- Gymnasium (not Gym): notebooks use Gymnasium API (reset returns `(obs, info)`; step returns `(obs, reward, terminated, truncated, info)`). Use `done = terminated or truncated` consistently.
- Rendering for GIFs: create env with `render_mode="rgb_array"` and call `env.render()` (no `mode` kwarg) to append frames (see evaluation cell in `walker_hardcore_3.ipynb`).
- VecNormalize usage (SB3): training wraps envs with `VecNormalize`. The training code saves `vec_normalize.pkl` (via `train_env.save`) — always load this when doing deterministic evaluation to maintain consistent observation normalization.
- Two saving conventions:
  - Custom SAC agent: actor weights are saved with `torch.save(agent.actor.state_dict(), "hardcore_solved.pth")`.
  - Stable-Baselines3 (PPO): use `model.save(path)` and store `vec_normalize.pkl` alongside.
- Seeding: envs are seeded in `make_env(rank)` with `SEED + rank` and eval envs use `1000 + i`. Keep this pattern to preserve reproducibility across distributed experiments.
- Evaluation artifacts: `logs/hardcore_5/eval` holds `evaluations.npz` used by plotting cells (`timesteps`, `results`, `ep_lengths`). Use these arrays to compute mean/std of eval runs.
- Early stopping & goal: some training stops when `np.mean(scores_window) >= 300` and saves `hardcore_solved.pth` (SAC notebook). This is the local training success heuristic.

---

## Files & locations to reference (useful starting points)
- `dependencies.py` — single-step dependency installation.
- `walker_hardcore_3.ipynb` — primary training and evaluation code (contains both custom SAC and SB3 PPO examples).
- `logs/`, `hardcore_5/`, `hardsore_3/` — run folders and `run_info.json` + `evaluations.npz`.
- `arch_sweep/` and `ablation_runs/` — configurations and result archives for architecture/ablation experiments.

---

## Common fixes or low-friction PR ideas (explicit examples)
- Ensure VecNormalize is loaded before evaluation: add a helper that loads `vec_normalize.pkl` alongside SB3 `model.load()` and pairs it with the eval env.
- Add a small CLI wrapper (e.g. `train.py`) that accepts `--env`, `--total-steps`, and `--seed` so runs can be started non-interactively (notebook-first currently).
- Make a short smoke-test GitHub Action that runs a single-episode evaluation with `N_ENVS=1` and `TOTAL_TIMESTEPS=10000` to detect obvious regressions.

---

## Testing & debugging tips
- Quick local verification: set `N_ENVS=1`, `TOTAL_TIMESTEPS` small, and `device='cpu'` to ensure deterministic behavior during debugging.
- When evaluating, check `evaluations.npz` shape and that `results.mean(axis=1)` is used — misinterpreting axes has caused incorrect summary plots before.
- If rendering returns `None`, re-check that the env was created with `render_mode='rgb_array'` and that Gymnasium vs Gym APIs are not mixed.

---

## Prompts & examples for AI agent tasks
- "Add a utility to load SB3 model + its `VecNormalize` wrapper for evaluation; update evaluation notebook to use it and document usage in README or notes." 
- "Add `train.py` CLI that mirrors the notebook `# --- TRAINING LOOP ---` behaviour and preserves seeding/vec-normalize saving." 
- "Add unit-like smoke tests (very short runs) and a minimal GitHub Action to run them on PRs." 

---

## Pitfalls / gotchas to watch for
- Mixing Gym and Gymnasium render/step/reset signatures.
- Forgetting to save or load `VecNormalize` causing large distributional shifts at evaluation time.
- Two parallel training styles (custom SAC vs SB3 PPO) — make sure changes to core naming (e.g., env ID, obs shape) are implemented consistently in both places.

---

If you'd like, I can now:
1) Add this file to the repo (.github/copilot-instructions.md), and
2) Create a short PR that also adds a one-file `train.py` smoke-run wrapper and a minimal GitHub Action for smoke tests.

What would you prefer I do next? 👇
