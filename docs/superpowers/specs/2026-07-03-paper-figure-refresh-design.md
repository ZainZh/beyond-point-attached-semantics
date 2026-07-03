# Paper Figure Refresh Design

## Goal
Replace all stale template image references on the project page with the six newly uploaded paper figures from `static/images`, preserving the formal arXiv/project-page structure.

## Approved Direction
Use the paper-order figure mapping approved by the user. Rename uploaded images from Chinese filenames with spaces to stable English filenames for GitHub Pages compatibility.

## Figure Mapping
1. `图片 1.png` -> `figure1-teaser.png`: teaser of object-centric continuous semantic field.
2. `图片 2.png` -> `figure2-method-overview.png`: method overview with support points, tri-plane cache, query locations, and semantic point clouds.
3. `图片 3.png` -> `figure3-real-world-tasks.png`: real-world tasks and object splits.
4. `图片 4.png` -> `figure4-feature-visualization.png`: cross-instance feature visualization.
5. `图片 5.png` -> `figure5-simulation-tasks.png`: simulation task examples.
6. `图片 6.png` -> `figure6-real-world-setup.png`: real-world experimental setup.

## Page Structure Changes
- Use `figure1-teaser.png` for social preview/favicon and add a Core Idea figure section near the top.
- Use `figure2-method-overview.png` in Method Overview.
- Keep numeric result cards, then show `figure4-feature-visualization.png` under representation analysis.
- Add a Tasks & Setup section with `figure5-simulation-tasks.png`, `figure3-real-world-tasks.png`, and `figure6-real-world-setup.png`.
- Keep real-world video galleries unchanged.

## Verification
- Update the Python unittest to require all six renamed figure assets and reject old template image names.
- Run the updated test before implementation and confirm it fails.
- Rename assets, update HTML/CSS, rerun tests, run the local asset scan, and render smoke-test desktop/mobile screenshots.
