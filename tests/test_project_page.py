import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "index.html"


class AnchorParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.anchors = []
        self._stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a":
            self._stack.append({"href": attrs.get("href", ""), "text": []})

    def handle_data(self, data):
        if self._stack:
            self._stack[-1]["text"].append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._stack:
            item = self._stack.pop()
            item["text"] = " ".join(part.strip() for part in item["text"] if part.strip())
            self.anchors.append(item)


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
            "Core Idea",
            "Tasks & Setup",
            "Representation Analysis",
        ]
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, self.html)

    def test_references_current_assets(self):
        required_assets = [
            "static/pdfs/Beyond_Point_Attached_Semantics__Object_Centric_Semantic_Fields_for_Generalizable_Manipulation (2).pdf",
            "static/videos/full_video.mp4",
            "static/images/figure1-teaser.png",
            "static/images/figure2-method-overview.png",
            "static/images/figure3-real-world-tasks.png",
            "static/images/figure4-feature-visualization.png",
            "static/images/figure5-simulation-tasks.png",
            "static/images/figure6-real-world-setup.png",
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
            "system.png",
            "experiment.png",
            "PRISM_logo.png",
            "PRISM2.png",
            "图片",
        ]
        lower = self.html.lower()
        for text in forbidden:
            with self.subTest(text=text):
                self.assertNotIn(text.lower(), lower)


    def test_code_link_is_public_anchor_and_dataset_is_hidden(self):
        parser = AnchorParser()
        parser.feed(self.html)
        code_url = "https://github.com/ZainZh/beyond-point-attached-semantics"
        code_anchors = [item for item in parser.anchors if item["href"] == code_url]
        self.assertGreaterEqual(len(code_anchors), 2)
        self.assertTrue(any("Code" in item["text"] for item in code_anchors))
        self.assertNotIn('<span>Data</span>', self.html)
        self.assertNotIn('<strong>Data</strong>', self.html)
        self.assertNotIn('Dataset link can be added here', self.html)

    def test_no_duplicate_ids(self):
        ids = re.findall(r'id="([^"]+)"', self.html)
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        self.assertEqual([], duplicates)


if __name__ == "__main__":
    unittest.main()
