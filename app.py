"""
FlowCast Streamlit Cloud Entrypoint
Redirects to dashboard/app.py for seamless Streamlit Cloud and local execution.
"""

import runpy
from pathlib import Path

app_path = Path(__file__).parent / "dashboard" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
