"""Entry point for Streamlit Cloud — redirects to web/app.py"""
import importlib.util
import sys
from pathlib import Path

# web/app.py をロード
app_path = Path(__file__).parent / "web" / "app.py"
spec = importlib.util.spec_from_file_location("app", app_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["app"] = mod
spec.loader.exec_module(mod)
