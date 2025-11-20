import os
import sys


# Ensure repository root is on sys.path for pytest runs
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import-time app render for tests that assert sidebar strings without
# explicitly importing the app. We import the app after the test module is
# imported so any Streamlit stub the test installs is active.
# Note: Avoid auto-importing the app here; several tests stub streamlit
# before import to assert import-time writes.
