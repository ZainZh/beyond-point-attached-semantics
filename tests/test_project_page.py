import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
ROOT = Path(__file__).resolve().parents[1]

class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.assets, self.ids, self.meta, self.controls, self.videos = [], [], [], [], [], []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs: self.ids.append(attrs['id'])
        if tag == 'a': self.links.append(attrs.get('href', ''))
        if 'src' in attrs: self.assets.append(attrs['src'])
        if 'poster' in attrs: self.assets.append(attrs['poster'])
        if tag == 'link': self.assets.append(attrs['href'])
        if tag == 'meta': self.meta.append(attrs)
        if 'aria-controls' in attrs: self.controls.append(attrs['aria-controls'])
        if tag == 'video': self.videos.append(attrs)

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
        for asset in self.page.assets:
            self.assertTrue(asset.startswith('static/'), asset)
            self.assertTrue((ROOT / urlsplit(asset).path).is_file(), asset)
        for link in self.page.links: self.assertIn(link[1:], self.page.ids)
        for target in self.page.controls: self.assertIn(target, self.page.ids)
        self.assertEqual(len(self.page.ids), len(set(self.page.ids)))
    def test_latest_results_and_figures(self):
        for value in ['69.3', '67.5', '73.0', 'G3Flow', 'Beat Cube', 'Stable Object-Centric Semantic Fields for Robust Manipulation']:
            self.assertIn(value, self.html)
        for name in ['first.png', 'real-world-objects.png', 'simulation-feature.png']:
            self.assertTrue(any(a.endswith(name) for a in self.page.assets))
        self.assertEqual(18, sum(a.endswith('.mp4') for a in self.page.assets))

    def test_stylesheet_cache_version(self):
        stylesheets = [urlsplit(asset) for asset in self.page.assets
                       if urlsplit(asset).path == 'static/css/index.css']
        self.assertEqual(1, len(stylesheets))
        self.assertTrue(parse_qs(stylesheets[0].query).get('v'))

    def test_video_accessibility_and_loading(self):
        self.assertEqual(18, len(self.page.videos))
        for video in self.page.videos:
            self.assertIn('aria-label', video)
            self.assertIn('poster', video)
            self.assertIn('playsinline', video)
            if video.get('id') != 'cover-video':
                self.assertEqual('none', video.get('preload'))
                self.assertIn('controls', video)
            if video.get('id') != 'overview-film':
                self.assertIn('muted', video)

    def test_independent_video_groups(self):
        for task in ['mug', 'hammer', 'stir', 'pour']:
            for number in range(1, 5):
                self.assertIn(f'static/videos/{task}_{number}.mp4', self.page.assets)
            self.assertIn(f'task-{task}', self.page.ids)
        script = (ROOT / 'static/js/index.js').read_text()
        self.assertIn('Promise.allSettled', script)
        self.assertNotIn('jquery', script.lower())
        self.assertNotIn('http://', script)
        self.assertNotIn('https://', script)

    def test_protocol_distinctions(self):
        for phrase in ['same task and object setup', '20 trials per task',
                       'five training seeds', 'does not isolate query resampling alone',
                       'Part-supervised point-wise']:
            self.assertIn(phrase, self.html)

if __name__ == '__main__':
    unittest.main()
