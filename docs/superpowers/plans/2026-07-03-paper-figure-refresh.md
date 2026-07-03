# Paper Figure Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stale template image references with six correctly mapped paper figures.

**Architecture:** Keep the single static HTML page and local CSS. Rename image assets to stable English filenames, update tests to enforce the figure mapping, and update page sections to present figures in paper order.

**Tech Stack:** HTML, CSS, Python standard-library `unittest`, local PNG assets.

---

### Task 1: Update Validation Test First

**Files:**
- Modify: `tests/test_project_page.py`

- [ ] **Step 1: Require renamed paper figure assets**

Update `test_references_current_assets` to require `figure1-teaser.png` through `figure6-real-world-setup.png` and remove `system.png` / `experiment.png`.

- [ ] **Step 2: Reject stale image paths**

Add `system.png`, `experiment.png`, `PRISM_logo.png`, `PRISM2.png`, and Chinese `图片` paths to forbidden content.

- [ ] **Step 3: Run red test**

Run `python3 -m unittest tests/test_project_page.py -v`. Expected: FAIL because the renamed assets and HTML references are not implemented yet.

### Task 2: Rename Figure Assets

**Files:**
- Move: `static/images/图片 1.png` -> `static/images/figure1-teaser.png`
- Move: `static/images/图片 2.png` -> `static/images/figure2-method-overview.png`
- Move: `static/images/图片 3.png` -> `static/images/figure3-real-world-tasks.png`
- Move: `static/images/图片 4.png` -> `static/images/figure4-feature-visualization.png`
- Move: `static/images/图片 5.png` -> `static/images/figure5-simulation-tasks.png`
- Move: `static/images/图片 6.png` -> `static/images/figure6-real-world-setup.png`

- [ ] **Step 1: Move files with non-destructive checks**

Verify source files exist and target files do not already exist, then rename them.

### Task 3: Update HTML Figure Storyline

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Update metadata image references**

Use `static/images/figure1-teaser.png` for Open Graph, Twitter, and favicon references.

- [ ] **Step 2: Add Core Idea figure**

Add a section after Abstract using Figure 1 and the extracted PDF caption.

- [ ] **Step 3: Replace Method figure**

Use Figure 2 in Method Overview with the extracted PDF caption.

- [ ] **Step 4: Add Representation Analysis figure**

Use Figure 4 under Results after the metric cards.

- [ ] **Step 5: Add Tasks & Setup section**

Add Figures 5, 3, and 6 with extracted captions before Real-World Videos.

### Task 4: Update Styling and Verify

**Files:**
- Modify: `static/css/index.css`

- [ ] **Step 1: Add figure-grid styles**

Style multi-figure sections so large paper figures remain readable on desktop and mobile.

- [ ] **Step 2: Run verification**

Run the validation test, local asset scan, stale path search, and headless Chrome desktop/mobile screenshots.
