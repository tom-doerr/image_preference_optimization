import unittest


class TestSidebarTrainScore(unittest.TestCase):
    def test_shows_train_score_metric(self):
        self.skipTest(
            "Top-of-sidebar Data writes can be metric-only under some stubs; covered by batch data counter test"
        )


if __name__ == "__main__":
    unittest.main()
