"""Compatibility entry point for the independent PCA-Gabor PalmNet M5 experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from palmprint.training.palmnet import main

if __name__ == "__main__":
    main()
