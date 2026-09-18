# Anonymous supplementary website

Open `index.html` directly in a browser. No build step or external services are required.

The page contains the current method, experimental results, paper figures, a 2:44 narrated overview, and 16 task videos. Author names, affiliations, personal links, author citations, paper downloads, and code links are intentionally absent. All assets, fonts, and scripts are local; no analytics or remote services are loaded.

The design uses a full-bleed experiment hero, task tabs, grouped playback, expandable results, and zoomable figures. Task clips play at 3x speed and are muted. The English overview includes male narration and burned-in subtitles. Each demonstration keeps its own playback state, so one clip ending does not stop the others. Hidden task videos are paused and load only when played.

The latest method figure is sourced from the official ICRA manuscript. The real-world protocol uses held-out physical objects; the simulation protocol uses the same task/object setup in training and evaluation. Keep these settings distinct when editing results.

Run checks with `python3 -m unittest discover -s tests -v`.

## Anonymous hosting

Publish only `index.html` and its referenced `static/` assets to an anonymous host. Unused assets from earlier versions remain in this repository and need not be deployed. Do not submit a personal hosting URL, repository URL, Git history, editor settings, or local development files. The robots directive requests no indexing but is not access control. Previously public versions may remain cached or discoverable; this local update cannot retract them.

Video frames have been sampled for review, not exhaustively inspected frame by frame. Check the complete clips for visible people, institution marks, and identifying labels before submission.

## Template attribution

The original site was based on the Academic Project Page Template and Nerfies. This redesign takes layout inspiration from the WholeBodyWAM project website, with original markup and styling and this project's own research content and media. Retain applicable third-party licenses, including those for the bundled Font Awesome icons.
