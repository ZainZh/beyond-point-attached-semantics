# CoRL Project Page Design

## Goal
Create a polished static project website for "Beyond Point-Attached Semantics: Object-Centric Semantic Fields for Generalizable Manipulation" using the provided PDF, figures, and videos, suitable for linking from arXiv.

## Approved Direction
Use a formal academic project page rather than a lab showcase or marketing page. The first screen should show the paper title, authors, institution, primary links, and the main experiment video.

## Page Structure
1. Hero with title, author list, CUHK affiliation, CoRL 2026 label, and links for Paper, Code, Dataset, and Video.
2. Main teaser video using `static/videos/full_video.mp4` with concise caption about object-centric continuous semantic fields for manipulation.
3. Abstract copied from the paper text, with stale template content removed.
4. Method overview using `static/images/system.png`, explaining support points, query locations, semantic fields, and semantic point clouds.
5. Results section with simulation and real-world result cards based on the extracted PDF tables.
6. Real-world video gallery grouped into Grasp Mug, Beat Hammer/Cube, Stir Mug, and Pour Water using the provided task clips.
7. Additional figures using `static/images/experiment.png` for real-world task setup when relevant; omit figures whose captions cannot be made accurate from the current paper.
8. BibTeX with a provisional arXiv-style citation keyed as `sun2026beyondpointattached`, using the extracted title, authors, and year.
9. Footer crediting the academic project page template.

## Constraints
- Keep the site static: edit `index.html`, `static/css/index.css`, and small local tests only.
- Do not revert pre-existing user changes or deleted old assets.
- Avoid broken placeholder paths such as `carousel1.mp4`, `sample.pdf`, or the old non-existent PDF filename.
- Use direct local assets; no new external runtime dependencies.
- Keep the design responsive, readable, and video-forward.

## Verification
- Add a Python unittest that checks required paper text, links, media asset paths, absence of stale placeholder text, and duplicate carousel IDs.
- Run the test once before implementation and confirm it fails on the stale template.
- Run the test after implementation and confirm it passes.
- Start a local static server for preview and basic manual inspection if feasible.
