import os
import unittest


try:
    from playwright.sync_api import sync_playwright  # type: ignore
except Exception:  # pragma: no cover
    sync_playwright = None


class TestPlaywrightUISmoke(unittest.TestCase):
    def setUp(self):
        if os.getenv('PW_RUN') != '1':
            self.skipTest('Set PW_RUN=1 to run Playwright UI tests')
        if sync_playwright is None:
            self.skipTest('playwright not installed')

    def test_page_loads_and_images_present(self):
        url = os.getenv('PW_URL', 'http://localhost:8597')
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until='domcontentloaded')
            # Expect the app title and main title
            page.wait_for_selector("text=Optimize Latents by Preference")
            # Click Generate pair to force images (stub app auto-generates too)
            try:
                page.get_by_role("button", name="Generate pair").click(timeout=2000)
            except Exception:
                pass
            # Wait a bit for images to render
            page.wait_for_timeout(500)
            imgs = page.locator("img").count()
            self.assertGreaterEqual(imgs, 2)
            browser.close()


if __name__ == '__main__':
    unittest.main()

