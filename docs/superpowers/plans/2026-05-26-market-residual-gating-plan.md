# Market Residual Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal post-model evaluation path that neutralizes simple cross-sectional style exposures and gates bad market states without changing the backbone or training loop.

**Architecture:** Keep the existing prediction files unchanged. Add small utilities that 1) residualize daily scores against a few style features and 2) build a state-gated daily top1 strategy from those residualized scores plus market-state thresholds. Expose both through a standalone script that writes CSV/JSON diagnostics.

**Tech Stack:** Python, pandas, existing market research/live proxy utilities, pytest

---

### Task 1: Add failing tests for residual ranking and state gating

**Files:**
- Modify: `tests/test_market_research.py`
- Test: `tests/test_market_research.py`

- [ ] **Step 1: Write the failing tests**

```python
    def test_residualize_prediction_scores_removes_linear_style_bias(self):
        ...

    def test_build_state_gated_top1_strategy_can_hold_cash_on_bad_state(self):
        ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_market_research.py -k 'residualize_prediction_scores or state_gated_top1'`
Expected: FAIL with import or attribute errors for new helpers.

- [ ] **Step 3: Write minimal implementation**

Add the smallest helper set needed in utility modules.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_market_research.py -k 'residualize_prediction_scores or state_gated_top1'`
Expected: PASS

### Task 2: Implement residual ranking and state-gated strategy helpers

**Files:**
- Modify: `utils/market_research.py`
- Modify: `utils/market_live_proxy.py`

- [ ] **Step 1: Add residual-score helper**

```python
def residualize_prediction_scores(prediction_frame, feature_frame, feature_columns):
    ...
```

- [ ] **Step 2: Add state-gated top1 helper**

```python
def build_state_gated_top1_strategy_frame(...):
    ...
```

- [ ] **Step 3: Verify targeted tests**

Run: `python -m pytest -q tests/test_market_research.py -k 'residualize_prediction_scores or state_gated_top1'`
Expected: PASS

### Task 3: Add executable research script and verify on 2025

**Files:**
- Create: `scripts/market_daily/evaluate_residual_gating.py`
- Test: `scripts/market_daily/evaluate_residual_gating.py`

- [ ] **Step 1: Implement script**

The script should:
- load prediction files
- join market cache features
- evaluate baseline / residualized / residualized+gated
- write summaries and daily CSVs

- [ ] **Step 2: Smoke run**

Run: `python scripts/market_daily/evaluate_residual_gating.py --year 2025 --model Transformer --variant single_head_csrank_topq_v1`
Expected: a summary CSV and per-strategy daily outputs under `logs/residual_gating_eval/...`

### Task 4: Run validation and summarize outcome

**Files:**
- Output only under `logs/residual_gating_eval`

- [ ] **Step 1: Run 2025 focused validation**

Run the script for:
- `single_head_csrank_topq_v1`
- `stage2topheavy_topk_csrank_topq_headcand_v2`

- [ ] **Step 2: Run one broad-year comparison**

Run at least `2021` and `2025` for the mainline variant to compare good year vs failure year.

- [ ] **Step 3: Record the conclusion**

Summarize whether:
- residual ranking reduces repeated bad picks
- state gating cuts 2025 drawdown / mean loss
- the combined approach is worth promoting to the next research stage
