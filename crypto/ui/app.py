"""Streamlit UI wrapper redirecting to main application entry point."""
from pathlib import Path
import runpy
import sys

# Ensure crypto root is in sys.path
CRYPTO_DIR = Path(__file__).resolve().parent.parent
if str(CRYPTO_DIR) not in sys.path:
    sys.path.insert(0, str(CRYPTO_DIR))

# Execute main app
main_app = CRYPTO_DIR / "app.py"
runpy.run_path(str(main_app), run_name="__main__")
