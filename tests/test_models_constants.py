import unittest


class TestModelsConstants(unittest.TestCase):
    def test_default_model_first(self):
        import constants

        self.assertTrue(len(constants.MODEL_CHOICES) > 0)
        self.assertEqual(constants.MODEL_CHOICES[0], constants.DEFAULT_MODEL)


if __name__ == "__main__":
    unittest.main()
