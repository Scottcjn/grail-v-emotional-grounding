# SPDX-License-Identifier: MIT
"""Make code/ importable for the test suite (the scripts are not a package)."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, "code")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
