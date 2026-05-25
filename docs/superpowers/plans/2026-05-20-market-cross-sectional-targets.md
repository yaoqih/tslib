# Market Cross-Sectional Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal cross-sectional market training mode that keeps existing time-series backbones but trains them with rank-percentile targets, lightweight pairwise ranking, and richer market history features.

**Architecture:** Keep `Dataset_MarketDaily` and existing forecasting models intact, but extend market feature preparation and loss computation. Add optional cross-sectional target transformation and stricter same-day pairwise sampling via metadata-aware loss computation.

**Tech Stack:** Python, pandas, PyTorch, unittest, existing Time-Series-Library market pipeline

---

### Task 1: Extend Tests For New Market Supervision

**Files:**
- Modify: `tests/test_market_research.py`
- Test: `tests/test_market_research.py`

- [ ] **Step 1: Write failing tests for cross-sectional target transform and feature exposure**
- [ ] **Step 2: Run targeted tests to verify they fail**
- [ ] **Step 3: Write failing tests for metadata-aware pairwise loss filtering by date**
- [ ] **Step 4: Run targeted tests to verify they fail**

### Task 2: Add Market Feature And Target Utilities

**Files:**
- Modify: `utils/market_research.py`
- Test: `tests/test_market_research.py`

- [ ] **Step 1: Implement market-level rolling cross-sectional features in feature preparation**
- [ ] **Step 2: Implement optional cross-sectional rank-percentile target transform**
- [ ] **Step 3: Re-run targeted tests**

### Task 3: Wire Dataset Metadata

**Files:**
- Modify: `data_provider/data_loader.py`
- Test: `tests/test_market_research.py`

- [ ] **Step 1: Store per-sample date metadata and transformed target labels**
- [ ] **Step 2: Keep prediction export on original label scale**
- [ ] **Step 3: Re-run targeted tests**

### Task 4: Upgrade Market Loss Computation

**Files:**
- Modify: `utils/market_multitask.py`
- Modify: `exp/exp_long_term_forecasting.py`
- Modify: `run.py`
- Test: `tests/test_market_research.py`

- [ ] **Step 1: Add same-day filtered pairwise rank loss helper**
- [ ] **Step 2: Add CLI args for cross-sectional target mode and pairwise minimum target gap**
- [ ] **Step 3: Use dataset metadata in market loss path**
- [ ] **Step 4: Re-run targeted tests**

### Task 5: Verify Behavior

**Files:**
- Modify: `scripts/market_daily/diagnose_fold_stability.py` (only if needed)

- [ ] **Step 1: Run focused unit tests**
- [ ] **Step 2: Run a small smoke diagnosis on existing artifacts**
- [ ] **Step 3: Summarize behavioral changes and remaining risks**
