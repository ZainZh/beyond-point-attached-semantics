import unittest
from html.parser import HTMLParser
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.assets, self.ids, self.meta = [], [], [], []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs: self.ids.append(attrs['id'])
        if tag == 'a': self.links.append(attrs.get('href', ''))
        if 'src' in attrs: self.assets.append(attrs['src'])
        if tag == 'link': self.assets.append(attrs['href'])
        if tag == 'meta': self.meta.append(attrs)

class ProjectPageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / 'index.html').read_text()
        cls.page = PageParser()
        cls.page.feed(cls.html)
    def test_anonymous_resources(self):
        self.assertIn('Anonymous supplementary material', self.html)
        for value in ['publication-authors', 'publication-affiliation', 'BibTeX', 'full_video.mp4', '.pdf']:
            self.assertNotIn(value, self.html)
        self.assertFalse(any(m.get('name') == 'author' or m.get('property') == 'og:url' for m in self.page.meta))
        self.assertTrue(all(h.startswith('#') for h in self.page.links))
    def test_local_assets_and_anchors(self):
        for asset in self.page.assets: self.assertTrue((ROOT / asset).is_file(), asset)
        for link in self.page.links: self.assertIn(link[1:], self.page.ids)
        self.assertEqual(len(self.page.ids), len(set(self.page.ids)))
    def test_latest_results_and_figures(self):
        for value in ['69.3', '67.5', '73.0', 'G3Flow', 'Beat Cube', 'Stable Object-Centric Semantic Fields for Robust Manipulation']:
            self.assertIn(value, self.html)
        for name in ['first.png', 'real-world-objects.png', 'simulation-feature.png']:
            self.assertTrue(any(a.endswith(name) for a in self.page.assets))
        self.assertEqual(17, sum(a.endswith('.mp4') for a in self.page.assets))

if __name__ == '__main__':
    unittest.main()
