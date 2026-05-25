# Train-Loss Plateau And Longer Market Comparisons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal train-only stopping mode so market training does not depend on validation loss, then run longer comparable experiments for the raw baseline and cross-sectional target setup.

**Architecture:** Keep the existing training loop and checkpoint flow. Extend training control with a small helper that monitors smoothed train loss and saves checkpoints on improvement, similar to the current validation-based early stopping. Reuse the same model/data path and only vary training control and experiment configs.

**Tech Stack:** Python, PyTorch, unittest, existing Time-Series-Library market pipeline

---

### Task 1: Add Focused Tests For Train-Only Stopping

**Files:**
- Modify: `tests/test_market_research.py`
- Test: `tests/test_market_research.py`

- [ ] **Step 1: Write a failing test for train-loss plateau stop state**
- [ ] **Step 2: Run the focused test and verify it fails for missing helper**

### Task 2: Implement Train-Loss Plateau Stopping

**Files:**
- Modify: `utils/tools.py`
- Modify: `run.py`
- Modify: `exp/exp_long_term_forecasting.py`
- Test: `tests/test_market_research.py`

- [ ] **Step 1: Add a train-loss plateau checkpoint helper**
- [ ] **Step 2: Add CLI args for smoothing, delta, and patience**
- [ ] **Step 3: Wire the new train mode into the long-term forecast loop**
- [ ] **Step 4: Re-run focused tests until green**

### Task 3: Verify And Run Longer Comparisons

**Files:**
- Modify: none

- [ ] **Step 1: Run `tests.test_market_research`**
- [ ] **Step 2: Run one short smoke training with `train_mode=train_loss_plateau`**
- [ ] **Step 3: Launch longer `iTransformer` comparisons for `raw` vs `cross_section_rank`**
- [ ] **Step 4: Summarize top1, rank_ic, concentration, and stop behavior**
