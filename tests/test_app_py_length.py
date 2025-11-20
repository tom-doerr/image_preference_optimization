import unittest


class TestAppPyLength(unittest.TestCase):
    def test_app_under_400_lines(self):
        with open('app.py', 'r', encoding='utf-8') as f:
            n = sum(1 for _ in f)
        self.assertLessEqual(n, 400, msg=f'app.py too long: {n} lines')


if __name__ == '__main__':
    unittest.main()

