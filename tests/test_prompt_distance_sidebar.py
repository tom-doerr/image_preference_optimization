import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestPromptDistanceSidebar(unittest.TestCase):
    def test_sidebar_shows_prompt_distances(self):
        self.skipTest('Pair-based prompt distance sidebar removed in Batch-only UI')


if __name__ == '__main__':
    unittest.main()
