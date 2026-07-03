# CoRL Project Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale academic template with a polished static project page for the CoRL paper.

**Architecture:** The site remains a single static page with local CSS and existing local media assets. A small Python unittest validates the generated HTML for required content, valid referenced local assets, and removal of stale template content.

**Tech Stack:** HTML, CSS, Bulma assets already present in `static/css`, Font Awesome assets already present in `static/js`, Python standard-library `unittest`.

---

### Task 1: Add Static Site Validation Test

**Files:**
- Create: `tests/test_project_page.py`

- [ ] **Step 1: Write the failing test**

```python
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "index.html"


class ProjectPageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = HTML.read_text(encoding="utf-8")

    def test_contains_current_paper_identity(self):
        required = [
            "Beyond Point-Attached Semantics",
            "Object-Centric Semantic Fields for Generalizable Manipulation",
            "Zheng SUN",
            "Lerong ZHANG",
            "Quentin ROUXEL",
            "Zhihao LI",
            "Fei CHEN",
            "The Chinese University of Hong Kong",
            "CoRL 2026",
        ]
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, self.html)

    def test_contains_required_sections(self):
        required = [
            "Abstract",
            "Method Overview",
            "Results",
            "Real-World Videos",
            "Resources",
            "BibTeX",
        ]
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, self.html)

    def test_references_current_assets(self):
        required_assets = [
            "static/pdfs/Beyond_Point_Attached_Semantics__Object_Centric_Semantic_Fields_for_Generalizable_Manipulation (2).pdf",
            "static/videos/full_video.mp4",
            "static/images/system.png",
            "static/images/experiment.png",
            "static/videos/mug0.mp4",
            "static/videos/hammer1.mp4",
            "static/videos/stirring1.mp4",
            "static/videos/pour_water.mp4",
        ]
        for asset in required_assets:
            with self.subTest(asset=asset):
                self.assertIn(asset, self.html)
                self.assertTrue((ROOT / asset).exists(), asset)

    def test_removes_stale_template_content(self):
        forbidden = [
            "DESCRIPTION META TAG",
            "SOCIAL MEDIA TITLE TAG",
            "hyperspectral",
            "SpectralGrasp",
            "RPISM",
            "sample.pdf",
            "carousel1.mp4",
            "carousel2.mp4",
            "carousel3.mp4",
        ]
        lower = self.html.lower()
        for text in forbidden:
            with self.subTest(text=text):
                self.assertNotIn(text.lower(), lower)

    def test_no_duplicate_ids(self):
        ids = re.findall(r'id="([^"]+)"', self.html)
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        self.assertEqual([], duplicates)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_project_page.py -v`
Expected: FAIL because the existing page still contains stale template content and broken placeholder paths.

### Task 2: Rewrite Page Content

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Replace metadata and hero**

Set title, description, keyword, Open Graph, Twitter, author, institution, and action links to match the current paper and local assets. Keep local CSS/JS imports already used by the static template.

- [ ] **Step 2: Replace body sections**

Create these body sections in order: hero, teaser video, abstract, method overview, results, real-world videos, resources, BibTeX, footer. Use only current local asset paths from `static/pdfs`, `static/images`, and `static/videos`.

### Task 3: Refresh Styling

**Files:**
- Modify: `static/css/index.css`

- [ ] **Step 1: Add page-specific layout styles**

Add responsive styles for hero actions, teaser media, figure panels, result cards, resource cards, and video gallery.

- [ ] **Step 2: Keep compatibility with existing template JS/CSS**

Preserve class names that are still useful from the academic template, including `.publication-title`, `.publication-authors`, `.publication-video`, and `.author-block`.

### Task 4: Verify and Preview

**Files:**
- Read: `index.html`
- Read: `static/css/index.css`

- [ ] **Step 1: Run validation test**

Run: `python3 -m unittest tests/test_project_page.py -v`
Expected: PASS.

- [ ] **Step 2: Check asset paths**

Run this Python scan:

```python
from html.parser import HTMLParser
from pathlib import Path

class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"src", "href", "content"} and value and value.startswith("static/"):
                self.assets.append(value)

root = Path(".")
parser = AssetParser()
parser.feed(Path("index.html").read_text(encoding="utf-8"))
missing = sorted({asset for asset in parser.assets if not (root / asset).exists()})
if missing:
    raise SystemExit("Missing assets: " + ", ".join(missing))
print(f"Checked {len(parser.assets)} local static asset references")
```

Expected: all local assets referenced by `src`, `href`, and `content` exist.

- [ ] **Step 3: Start local preview server**

Run: `python3 -m http.server 8000`
Expected: page is available at `http://localhost:8000/`.
