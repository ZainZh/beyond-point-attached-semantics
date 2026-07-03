# Findings & Decisions

## Requirements
- Create a website for a paper intended for CoRL/arXiv.
- Use the provided materials in the repository, especially PDFs and videos.
- Existing project appears to be a static website skeleton.

## Research Findings
- Repository root contains `index.html`, `README.md`, `static/css`, `static/js`, `static/images`, `static/pdfs`, and `static/videos`.
- Paper PDF path: `static/pdfs/Beyond_Point_Attached_Semantics__Object_Centric_Semantic_Fields_for_Generalizable_Manipulation (2).pdf`.
- Image assets include `static/images/system.png`, `static/images/experiment.png`, `static/images/PRISM_logo.png`, and `static/images/PRISM2.png`.
- Video assets include `full_video.mp4` plus task clips for mug, hammer, pour_water, and stirring.
- Git status showed pre-existing modifications/deletions in `index.html`, old PDFs, and old placeholder videos before this task began.
- Existing `index.html` is based on an academic project page template but still contains stale hyperspectral/PRISM content, broken placeholder video paths (`carousel1.mp4`, etc.), a PDF link that does not match the current PDF filename, and an outdated BibTeX block.
- Current image dimensions: `system.png` 4826x1663, `experiment.png` 1818x1506, `PRISM_logo.png` 616x529, `PRISM2.png` 4435x1011.
- Current video inventory contains one large `full_video.mp4` (~122 MB) and task clips: hammer, mug, pour_water, and stirring.
- Extracted PDF title: Beyond Point-Attached Semantics: Object-Centric Semantic Fields for Generalizable Manipulation.
- Extracted PDF authors: Zheng SUN, Lerong ZHANG, Quentin ROUXEL, Zhihao LI, and Fei CHEN, all from The Chinese University of Hong Kong.
- Extracted abstract describes an object-centric continuous semantic field conditioned on object point clouds and queried at explicit 3D locations to produce part-aware embeddings.
- Main paper contributions: queryable 3D semantic embeddings for functional object parts; category-level training with part anchoring, cross-instance alignment, and augmentation stability; semantic point clouds for policy conditioning validated in RoboTwin and real bimanual tasks.
- Simulation tasks/results from Table 1: Hang Mug 37%, Beat Hammer 84%, Open Microwave 35%, Put Cabinet 83% for the proposed method.
- Real-world tasks/results from Table 2: Grasp Mug 17/20, Beat Cube 17/20, Stir Mug 10/20, Pour Water 10/20 for the proposed method.
- `full_video.mp4` duration is 361.25 seconds and size is about 122 MB.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Use existing static site structure | It is already present and suitable for arXiv/project-page hosting. |
| Build a formal arXiv/project page | User approved this direction and it fits the paper-submission use case. |
| Include all provided task clips in grouped video galleries | The user said all materials are in `videos`; grouping keeps the page navigable while preserving the assets. |
| Use `preload="metadata"` for videos | Avoids forcing full video downloads on page load, especially for the large main and stirring clips. |
| Keep Code/Data as resource cards without external release URLs | No reliable current code/data URLs were present in the provided materials. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Sandboxed shell failed to launch with bwrap loopback error | Re-ran required read-only commands with escalation. |
| Shell backticks in a Python one-liner were interpreted by bash | Switched to single-quoted shell command wrapping Python code. |
| `view_image` could not load PNGs due to the sandbox helper bwrap failure | Use PDF captions, file metadata, and browser verification instead. |

## Resources
- `index.html`
- `README.md`
- `static/pdfs/Beyond_Point_Attached_Semantics__Object_Centric_Semantic_Fields_for_Generalizable_Manipulation (2).pdf`
- `static/images/system.png`
- `static/images/experiment.png`
- `static/videos/full_video.mp4`

## Visual/Browser Findings
- Local image viewer could not load PNGs because the sandbox helper failed, so visual interpretation relied on PDF captions and render checks.
- Headless Chrome rendered desktop and mobile screenshots successfully: `/tmp/corl_project_page_desktop.png` 1440x1600 and `/tmp/corl_project_page_mobile.png` 390x1200.

## 2026-07-03 Figure Refresh Findings
- User replaced old template images with six paper figures under `static/images`: `图片 1.png` through `图片 6.png`.
- Old image files `system.png`, `experiment.png`, `PRISM_logo.png`, and `PRISM2.png` are deleted, so the current page image references are broken.
- New image dimensions: `图片 1.png` 4239x2213, `图片 2.png` 4138x2461, `图片 3.png` 4207x2332, `图片 4.png` 3020x2488, `图片 5.png` 3231x1902, `图片 6.png` 2915x2218.
- PDF captions map naturally by filename order: `图片 1.png` = Figure 1 teaser, `图片 2.png` = Figure 2 method overview, `图片 3.png` = Figure 3 real-world tasks/object splits, `图片 4.png` = Figure 4 cross-instance feature visualization, `图片 5.png` = Figure 5 simulation task examples, `图片 6.png` = Figure 6 real-world experimental setup.
- Current validation test fails only on deleted old image assets, which is the correct red-state for this update.
