# Task Plan: CoRL Paper Website

## Goal
Build a polished static project website for the CoRL paper using the provided PDF, images, and videos, suitable to link from arXiv.

## Current Phase
Phase 5

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand user intent
- [x] Inventory repository files
- [x] Extract content from paper and media assets
- [x] Document findings in findings.md
- **Status:** complete

### Phase 2: Design Approval
- [x] Propose website approaches
- [x] Present concise design for user approval
- [x] Write approved design spec
- **Status:** complete

### Phase 3: Implementation
- [x] Add or update static website files
- [x] Preserve existing assets and avoid unrelated reversions
- [x] Keep layout responsive and video-focused
- **Status:** complete

### Phase 4: Testing & Verification
- [x] Verify HTML/CSS renders locally
- [x] Check asset paths and video embeds
- [x] Review responsive layout with headless Chrome screenshots
- **Status:** complete

### Phase 5: Delivery
- [x] Review all output files
- [x] Ensure deliverables are complete
- [ ] Deliver to user
- **Status:** in_progress

## Key Questions
1. Should the site be a compact arXiv-style project page or a richer lab/demo showcase? Answer: compact formal arXiv/project page.
2. Which content should be featured above the fold? Answer: title/authors/links followed immediately by the main video.
3. What final paper title/authors/links should appear? Answer: title and authors extracted from the PDF; Paper links to local PDF; Code/Data cards are present for release links.

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep this as a static site | Existing repo already has static HTML/CSS/JS and arXiv project pages are easiest to host as static files. |
| Replace the old template content wholesale | The previous page contained stale content and broken paths from another paper. |
| Add a Python unittest for page validation | Provides a repeatable check for paper identity, required assets, required sections, and stale template content. |
| Proceed in-place on `gh-pages` | The provided PDF/videos are untracked in this workspace; a new worktree would not contain them without copying. |
| Rename uploaded paper figures to English filenames | Avoid URL encoding issues from Chinese filenames and spaces on GitHub Pages. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Sandboxed shell failed with `bwrap: loopback: Failed RTM_NEWADDR` | 1 | Re-ran required commands with escalation, per sandbox instructions. |
| `python3 -m unittest tests/test_project_page.py -v` initially could not import `tests.test_project_page` | 1 | Added empty `tests/__init__.py`; rerun failed for the expected stale-page assertions. |
| `view_image` could not load local PNGs due to the sandbox helper failure | 1 | Used PDF captions, asset metadata, HTTP checks, and headless Chrome screenshots instead. |

### Phase 6: Paper Figure Refresh
- [x] Map newly uploaded images to PDF figures
- [x] Rename image assets to stable English filenames
- [x] Update validation tests for the new figure set
- [x] Update page sections and CSS
- [x] Verify tests, assets, stale path scan, and render screenshots
- **Status:** complete

## Notes
- Existing worktree was dirty before this task; unrelated pre-existing deletes and `.idea/` were not reverted.
- Local preview server is running at `http://127.0.0.1:8000/`.
