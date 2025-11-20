#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"

if [[ "$MODE" == "smoke" ]]; then
  echo "Running SMOKE tests..."
  echo "Python: $(python -V)"
  python -m unittest discover -s tests/smoke -p 'test_*.py'
  exit $?
fi

echo "Running unit tests (verbose)..."
echo "Python: $(python -V)"

# Isolate datasets to a temp folder for deterministic counts
export IPO_DATA_ROOT="$(mktemp -d .tmp_testdata.XXXXXX)"
echo "IPO_DATA_ROOT=${IPO_DATA_ROOT}"

python - <<'PY'
import unittest, time, sys

suite = unittest.defaultTestLoader.discover('tests', pattern='test_*.py')

class TimingResult(unittest.TextTestResult):
    def startTest(self, test):
        self._t0 = time.perf_counter()
        super().startTest(test)
    def _dur(self):
        return time.perf_counter() - getattr(self, '_t0', time.perf_counter())
    def addSuccess(self, test):
        self.stream.write(f"OK    {test}  ({self._dur():.3f}s)\n")
        super().addSuccess(test)
    def addFailure(self, test, err):
        self.stream.write(f"FAIL  {test}  ({self._dur():.3f}s)\n")
        super().addFailure(test, err)
    def addError(self, test, err):
        self.stream.write(f"ERROR {test}  ({self._dur():.3f}s)\n")
        super().addError(test, err)
    def addSkip(self, test, reason):
        self.stream.write(f"SKIP  {test}  ({self._dur():.3f}s) — {reason}\n")
        super().addSkip(test, reason)

class PurgeAppRunner(unittest.TextTestRunner):
    def _makeResult(self):
        res = super()._makeResult()
        _orig_start = res.startTest
        import sys as _sys
        def _start_test(test):
            _sys.modules.pop('app', None)
            _orig_start(test)
        res.startTest = _start_test
        return res

runner = PurgeAppRunner(verbosity=2, resultclass=TimingResult)
start = time.perf_counter()
result = runner.run(suite)
end = time.perf_counter()
print("\nSummary: ran", result.testsRun, "tests in", f"{end-start:.2f}s",
      f"; failures={len(result.failures)} errors={len(result.errors)} skipped={len(result.skipped)}")
sys.exit(0 if result.wasSuccessful() else 1)
PY
