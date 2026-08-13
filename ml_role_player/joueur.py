"""Compatibility entrypoint for the unified ERP Club FastAPI service.

Player routes now live in `app.main` so deployment can run a single FastAPI app:

    uvicorn app.main:app --host 0.0.0.0 --port 8000

This module keeps older local commands such as `python ml_role_player/joueur.py`
working by re-exporting the unified app.
"""

from pathlib import Path
import sys


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.main import app  # noqa: E402


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
