#!/usr/bin/env python3
"""Convenience script to run the spaceship navigator server."""
import sys
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from server.app import main

if __name__ == "__main__":
    main()
