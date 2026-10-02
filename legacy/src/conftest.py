"""Make src/ a package root for imports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
