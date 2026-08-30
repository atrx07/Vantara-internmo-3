"""Vantara customer behavior prediction platform."""

import os

# Restricted Windows runners may lack WMIC; make joblib's CPU fallback explicit.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

__version__ = "0.1.0"
