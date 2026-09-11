#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compatibility entrypoint for the shared, manifest-validated gap report."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.validated_tables import build, main, paired_counts

if __name__ == "__main__":
    main()
