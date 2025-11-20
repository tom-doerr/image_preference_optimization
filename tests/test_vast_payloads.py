import unittest

from scripts.vast_auto import onstart_cmd


class TestVastPayloads(unittest.TestCase):
    def test_onstart_contains_server_and_app(self):
        cmd = onstart_cmd(
            "https://example.com/repo.git", "stabilityai/sd-turbo", 8000, 8501
        )
        self.assertIn("git clone", cmd)
        self.assertIn("image_server_app.py", cmd)
        self.assertIn("streamlit run app.py", cmd)


if __name__ == "__main__":
    unittest.main()
