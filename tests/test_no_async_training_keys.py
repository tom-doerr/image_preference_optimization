import unittest


class TestNoAsyncTrainingKeys(unittest.TestCase):
    def test_constants_has_no_async_training_keys(self):
        from ipo.infra import constants
        self.assertFalse(hasattr(constants.Keys, "XGB_TRAIN_ASYNC"))
        self.assertFalse(hasattr(constants.Keys, "XGB_FIT_FUTURE"))


if __name__ == "__main__":
    unittest.main()

