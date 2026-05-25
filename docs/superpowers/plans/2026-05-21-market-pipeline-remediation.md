# Market Pipeline Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the market research pipeline so experiments are reproducible, market backbone uses time marks, tradability is visible to training objectives, and offline re-evaluation is consistent with saved prediction outputs.

**Architecture:** Keep the current market daily pipeline and cross-section model, but repair four high-impact misalignments: sample-cache isolation, backbone information flow, tradability-aware training supervision, and deterministic evaluation diagnostics. Changes stay local to the dataset, model, training loss helpers, and tests.

**Tech Stack:** Python, PyTorch, pandas, unittest

---

### Task 1: Cache Key Isolation

**Files:**
- Modify: `data_provider/data_loader.py`
- Test: `tests/test_market_research.py`

- [ ] Add a failing test that two datasets with different `market_test_end` or filtering args produce different sample cache paths.
- [ ] Run the focused unittest and confirm it fails for the current cache key.
- [ ] Extend `_sample_cache_path()` to include the split/filtering args that affect sample construction.
- [ ] Re-run the focused unittest and confirm it passes.

### Task 2: Market Backbone Time Marks

**Files:**
- Modify: `models/Transformer.py`
- Test: `tests/test_market_research.py`

- [ ] Add a failing regression test that `encode_market_sequence(x_enc, x_mark_enc)` changes when `x_mark_enc` changes.
- [ ] Run the focused unittest and confirm it fails under the current implementation.
- [ ] Wire `x_mark_enc` into `encode_market_sequence()`.
- [ ] Re-run the focused unittest and confirm it passes.

### Task 3: Tradable-Aware Market Losses

**Files:**
- Modify: `data_provider/data_loader.py`
- Modify: `exp/exp_long_term_forecasting.py`
- Modify: `utils/market_multitask.py`
- Modify: `run.py`
- Test: `tests/test_market_research.py`

- [ ] Add failing tests for mask-aware top-k/rank proxy behavior and for the experiment helper that builds tradable masks from sample metadata.
- [ ] Run the focused unittest set and confirm the new expectations fail.
- [ ] Implement tradable-mask helpers and apply them to rank loss, top-k loss, and training monitor proxies.
- [ ] Add a CLI flag for optionally applying the tradable mask to the regression loss as well, defaulting off for the minimal safe version.
- [ ] Re-run the focused unittests and confirm they pass.

### Task 4: Offline Re-evaluation Consistency

**Files:**
- Modify: `utils/market_research.py`
- Test: `tests/test_market_research.py`

- [ ] Add a failing test that a saved prediction frame can be re-evaluated and returns the same top-k metrics as direct evaluation.
- [ ] Run the focused unittest and confirm the gap or missing helper fails.
- [ ] Add a small helper to evaluate a prediction CSV/DataFrame deterministically through the same path used by market evaluation.
- [ ] Re-run the focused unittest and confirm it passes.

### Task 5: Verification

**Files:**
- Test: `tests/test_market_research.py`

- [ ] Run the full market research unittest file.
- [ ] Run one end-to-end diagnostic command that exercises dataset construction and reports cache path isolation.
- [ ] Summarize residual risks: tradable mask currently applied to ranking/top-k/proxy by default, but not forced onto regression unless the new flag is enabled.
